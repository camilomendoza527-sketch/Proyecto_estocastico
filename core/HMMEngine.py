"""HMMEngine: Implementación nativa de Modelos Ocultos de Markov (HMM).

Restricciones estrictas:
- Solo NumPy para operaciones matemáticas y algebraicas.
- Cero librerías externas para lógica HMM o estadística.
- Cada función matemática incluye un comentario con la ecuación que implementa.
- Forward y Backward utilizan factores de escala c_t (Rabiner, 1989).
- Viterbi opera en espacio logarítmico (log-space).
- Serialización y carga compatible con models/*.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class HMM:
    """Modelo Oculto de Markov discreto λ = (A, B, π).

    Attributes:
        letter: Carácter asociado al modelo (ej. 'a', 'b', 'ñ').
        n_states: Número de estados ocultos N.
        n_symbols: Tamaño del alfabeto de observaciones M (|V| = K + 4).
        A: Matriz de transición de estados de tamaño (N, N).
           a_ij = P(q_{t+1} = S_j | q_t = S_i)
        B: Matriz de emisión de tamaño (N, M).
           b_j(k) = P(O_t = v_k | q_t = S_j)
        pi: Vector de distribución inicial de estados de tamaño (N,).
            π_i = P(q_1 = S_i)
    """

    letter: str
    n_states: int
    n_symbols: int
    A: np.ndarray
    B: np.ndarray
    pi: np.ndarray

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        self.pi = np.asarray(self.pi, dtype=np.float64)
        self._validate_shapes_and_stochastic()

    def _validate_shapes_and_stochastic(self) -> None:
        """Verifica dimensiones y que las filas sumen 1 (propiedad estocástica)."""
        N = self.n_states
        M = self.n_symbols
        if self.A.shape != (N, N):
            raise ValueError(f"A debe tener forma ({N}, {N}), tiene {self.A.shape}")
        if self.B.shape != (N, M):
            raise ValueError(f"B debe tener forma ({N}, {M}), tiene {self.B.shape}")
        if self.pi.shape != (N,):
            raise ValueError(f"pi debe tener forma ({N},), tiene {self.pi.shape}")

        if not np.allclose(self.A.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Las filas de la matriz A deben sumar 1: sumas={self.A.sum(axis=1)}")
        if not np.allclose(self.B.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Las filas de la matriz B deben sumar 1: sumas={self.B.sum(axis=1)}")
        if not np.isclose(self.pi.sum(), 1.0, atol=1e-5):
            raise ValueError(f"El vector inicial pi debe sumar 1: suma={self.pi.sum()}")

    # =========================================================================
    # 1. EVALUACIÓN: Algoritmo Forward con Escalado
    # =========================================================================
    def forward(self, observations: Sequence[int]) -> tuple[np.ndarray, np.ndarray, float]:
        """Algoritmo Forward con coeficientes de escala c_t (Rabiner, 1989).

        Resuelve P(O | λ) evitando underflow numérico mediante escalado en cada paso.

        Ecuaciones:
        1. Inicialización (t = 1):
           α̃_1(i) = π_i · b_i(O_1)
           c_1 = 1 / ( Σ_{i=1}^N α̃_1(i) )
           α̂_1(i) = c_1 · α̃_1(i)

        2. Inducción (t = 1 ... T-1):
           α̃_{t+1}(j) = [ Σ_{i=1}^N α̂_t(i) · a_{ij} ] · b_j(O_{t+1})
           c_{t+1} = 1 / ( Σ_{j=1}^N α̃_{t+1}(j) )
           α̂_{t+1}(j) = c_{t+1} · α̃_{t+1}(j)

        3. Log-Verosimilitud:
           log P(O | λ) = - Σ_{t=1}^T log(c_t)

        Args:
            observations: Secuencia de símbolos O = (o_1, o_2, ..., o_T) con o_t ∈ [0, M-1].

        Returns:
            alpha_hat: Matriz de variables forward escaladas de tamaño (T, N).
            c: Vector de coeficientes de escala de tamaño (T,).
            log_prob: Log-verosimilitud log P(O | λ).
        """
        T = len(observations)
        N = self.n_states
        if T == 0:
            return np.empty((0, N)), np.empty(0), float("-inf")

        alpha_hat = np.zeros((T, N), dtype=np.float64)
        c = np.zeros(T, dtype=np.float64)

        # 1. Inicialización
        o_0 = observations[0]
        alpha_raw_0 = self.pi * self.B[:, o_0]
        sum_0 = np.sum(alpha_raw_0)
        c[0] = 1.0 / (sum_0 + 1e-300)
        alpha_hat[0] = c[0] * alpha_raw_0

        # 2. Inducción
        for t in range(T - 1):
            o_next = observations[t + 1]
            # [ Σ_{i=1}^N α̂_t(i) · a_{ij} ] · b_j(O_{t+1})
            alpha_raw_next = np.dot(alpha_hat[t], self.A) * self.B[:, o_next]
            sum_next = np.sum(alpha_raw_next)
            c[t + 1] = 1.0 / (sum_next + 1e-300)
            alpha_hat[t + 1] = c[t + 1] * alpha_raw_next

        # 3. Log-Verosimilitud
        log_prob = -float(np.sum(np.log(c + 1e-300)))
        return alpha_hat, c, log_prob

    def forward_step(
        self, alpha_hat_prev: np.ndarray, next_obs: int
    ) -> tuple[np.ndarray, float]:
        """Paso incremental del algoritmo Forward para tiempo real en la GUI.

        Ecuación:
            α̃_{t+1}(j) = [ Σ_{i=1}^N α̂_t(i) · a_{ij} ] · b_j(O_{t+1})
            c_{t+1} = 1 / ( Σ_{j=1}^N α̃_{t+1}(j) )
            α̂_{t+1}(j) = c_{t+1} · α̃_{t+1}(j)
            Δ_log = -log(c_{t+1})

        Args:
            alpha_hat_prev: Vector α̂_t de tamaño (N,).
            next_obs: Nuevo símbolo observado O_{t+1}.

        Returns:
            alpha_hat_next: Vector α̂_{t+1} de tamaño (N,).
            log_c_inv: Aporte a la log-probabilidad -log(c_{t+1}).
        """
        alpha_raw = np.dot(alpha_hat_prev, self.A) * self.B[:, next_obs]
        c_inv = np.sum(alpha_raw)
        c = 1.0 / (c_inv + 1e-300)
        alpha_hat_next = c * alpha_raw
        return alpha_hat_next, float(np.log(c_inv + 1e-300))

    # =========================================================================
    # 2. EVALUACIÓN: Algoritmo Backward con Escalado
    # =========================================================================
    def backward(self, observations: Sequence[int], c: np.ndarray) -> np.ndarray:
        """Algoritmo Backward con los coeficientes de escala c_t de Forward.

        Ecuaciones (Rabiner, 1989):
        1. Inicialización (t = T):
           β̂_T(i) = c_T · 1.0  (o 1.0 normalizado por c_T para mantener coherencia)
           (Siguiendo la convención estándar donde β̂_T(i) = c_T)

        2. Inducción (t = T-1, T-2, ..., 1):
           β̃_t(i) = Σ_{j=1}^N a_{ij} · b_j(O_{t+1}) · β̂_{t+1}(j)
           β̂_t(i) = c_t · β̃_t(i)

        Args:
            observations: Secuencia de símbolos O_{1:T}.
            c: Coeficientes de escala calculados en Forward para esta secuencia.

        Returns:
            beta_hat: Matriz de variables backward escaladas de tamaño (T, N).
        """
        T = len(observations)
        N = self.n_states
        if T == 0:
            return np.empty((0, N))

        beta_hat = np.zeros((T, N), dtype=np.float64)

        # 1. Inicialización
        beta_hat[T - 1] = c[T - 1]

        # 2. Inducción hacia atrás
        for t in range(T - 2, -1, -1):
            o_next = observations[t + 1]
            # Σ_{j=1}^N a_{ij} · b_j(O_{t+1}) · β̂_{t+1}(j)
            beta_raw = np.dot(self.A, self.B[:, o_next] * beta_hat[t + 1])
            beta_hat[t] = c[t] * beta_raw

        return beta_hat

    # =========================================================================
    # 3. DECODIFICACIÓN: Algoritmo de Viterbi en Log-Espacio
    # =========================================================================
    def viterbi(self, observations: Sequence[int]) -> tuple[list[int], float]:
        """Algoritmo de Viterbi en espacio logarítmico.

        Encuentra la secuencia de estados ocultos más probable Q* = (q_1*, ..., q_T*).

        Ecuaciones:
        1. Inicialización (t = 1):
           δ_1(i) = log(π_i) + log(b_i(O_1))
           ψ_1(i) = 0

        2. Inducción (t = 2 ... T):
           δ_t(j) = max_{1 ≤ i ≤ N} [ δ_{t-1}(i) + log(a_{ij}) ] + log(b_j(O_t))
           ψ_t(j) = argmax_{1 ≤ i ≤ N} [ δ_{t-1}(i) + log(a_{ij}) ]

        3. Terminación:
           q_T* = argmax_{1 ≤ i ≤ N} [ δ_T(i) ]
           P* = max_{1 ≤ i ≤ N} [ δ_T(i) ]

        4. Reconstrucción del camino (Backtracking, t = T-1 ... 1):
           q_t* = ψ_{t+1}(q_{t+1}*)

        Args:
            observations: Secuencia de observaciones O_{1:T}.

        Returns:
            best_path: Lista con los índices de estados más probables [q_1*, ..., q_T*].
            best_log_prob: Log-probabilidad del mejor camino P*.
        """
        T = len(observations)
        N = self.n_states
        if T == 0:
            return [], float("-inf")

        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        delta = np.zeros((T, N), dtype=np.float64)
        psi = np.zeros((T, N), dtype=int)

        # 1. Inicialización
        delta[0] = log_pi + log_B[:, observations[0]]

        # 2. Inducción
        for t in range(1, T):
            o_t = observations[t]
            # matriz (N, N) con fila i = delta[t-1, i] + log_A[i, j]
            # delta[:, None] transmite a columnas
            transitions = delta[t - 1, :, None] + log_A
            psi[t] = np.argmax(transitions, axis=0)
            delta[t] = np.max(transitions, axis=0) + log_B[:, o_t]

        # 3. Terminación
        best_last_state = int(np.argmax(delta[T - 1]))
        best_log_prob = float(delta[T - 1, best_last_state])

        # 4. Backtracking
        best_path = [0] * T
        best_path[T - 1] = best_last_state
        for t in range(T - 2, -1, -1):
            best_path[t] = int(psi[t + 1, best_path[t + 1]])

        return best_path, best_log_prob

    # =========================================================================
    # 4. APRENDIZAJE: Algoritmo Baum-Welch (EM) con Escalado
    # =========================================================================
    def fit(
        self,
        sequences: Sequence[Sequence[int]],
        max_iter: int = 50,
        tol: float = 1e-4,
        pseudo_count: float = 1e-3,
    ) -> list[float]:
        """Entrenamiento offline por Esperanza-Maximización (Baum-Welch) multisequencia.

        Ecuaciones (Rabiner, 1989 con escalado c_t):
        Para cada secuencia k con longitud T_k y factores c_t^k:
        1. Variables a posteriori:
           γ_t(i) = [ α̂_t(i) · β̂_t(i) ] / c_t
           ξ_t(i, j) = α̂_t(i) · a_{ij} · b_j(O_{t+1}) · β̂_{t+1}(j)

        2. Reestimación M-Step (agregando todas las secuencias k=1..K):
           π̄_i = [ Σ_k γ_1^k(i) ] / K
           ā_{ij} = [ Σ_k Σ_{t=1}^{T_k-1} ξ_t^k(i, j) ] / [ Σ_k Σ_{t=1}^{T_k-1} γ_t^k(i) ]
           b̄_j(m) = [ Σ_k Σ_{t: O_t^k = v_m} γ_t^k(j) ] / [ Σ_k Σ_{t=1}^{T_k} γ_t^k(j) ]

        3. Suavizado de Laplace / pseudo-conteos:
           Se agrega pseudo_count para garantizar b_j(m) > 0 y evitar probabilidades cero.

        Args:
            sequences: Lista de secuencias de observaciones de entrenamiento.
            max_iter: Máximo número de iteraciones de Baum-Welch.
            tol: Tolerancia de cambio en log-verosimilitud para convergencia.
            pseudo_count: Suavizado para emisiones y transiciones alcanzables.

        Returns:
            log_likelihood_history: Historial de log-verosimilitudes totales.
        """
        N = self.n_states
        M = self.n_symbols
        K_seqs = len(sequences)
        if K_seqs == 0:
            return []

        history: list[float] = []

        for iteration in range(max_iter):
            # Acumuladores para el paso M
            pi_num = np.zeros(N, dtype=np.float64)
            A_num = np.zeros((N, N), dtype=np.float64)
            A_den = np.zeros(N, dtype=np.float64)
            B_num = np.zeros((N, M), dtype=np.float64)
            B_den = np.zeros(N, dtype=np.float64)

            total_log_prob = 0.0

            for seq in sequences:
                T = len(seq)
                if T == 0:
                    continue

                alpha_hat, c, log_prob = self.forward(seq)
                total_log_prob += log_prob
                beta_hat = self.backward(seq, c)

                # 1. γ_t(i) = (α̂_t(i) * β̂_t(i)) / c_t
                gamma = (alpha_hat * beta_hat) / (c[:, None] + 1e-300)
                # Normalizar γ_t para estabilidad numérica: Σ_i γ_t(i) = 1
                gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-300

                # Acumulador π: γ_1(i)
                pi_num += gamma[0]

                # 2. ξ_t(i, j) = α̂_t(i) * a_ij * b_j(O_{t+1}) * β̂_{t+1}(j)
                for t in range(T - 1):
                    o_next = seq[t + 1]
                    xi_t = (
                        alpha_hat[t, :, None]
                        * self.A
                        * (self.B[:, o_next] * beta_hat[t + 1])[None, :]
                    )
                    xi_sum = np.sum(xi_t)
                    if xi_sum > 0:
                        xi_t /= xi_sum
                    A_num += xi_t
                    A_den += gamma[t]

                # Acumuladores de B
                for t in range(T):
                    o_t = seq[t]
                    B_num[:, o_t] += gamma[t]
                    B_den += gamma[t]

            history.append(total_log_prob)

            # Paso M: Reestimación
            # Reestimar pi (en topología izquierda-derecha fija: π = [1, 0, ...])
            new_pi = pi_num / K_seqs
            new_pi = np.maximum(new_pi, 1e-6)
            self.pi = new_pi / np.sum(new_pi)

            # Reestimar A
            new_A = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                if A_den[i] > 0:
                    new_A[i] = A_num[i] / A_den[i]
                else:
                    new_A[i] = self.A[i]

            # Preservar estructura de soporte (máscara de ceros estructurales de Bakis)
            mask_A = self.A > 0
            new_A = new_A * mask_A + pseudo_count * mask_A
            row_sums_A = np.sum(new_A, axis=1, keepdims=True)
            self.A = new_A / (row_sums_A + 1e-300)

            # Reestimar B
            new_B = np.zeros((N, M), dtype=np.float64)
            for j in range(N):
                if B_den[j] > 0:
                    new_B[j] = (B_num[j] + pseudo_count) / (B_den[j] + M * pseudo_count)
                else:
                    new_B[j] = (1.0 / M)
            row_sums_B = np.sum(new_B, axis=1, keepdims=True)
            self.B = new_B / (row_sums_B + 1e-300)

            # Criterio de parada
            if iteration > 0 and abs(history[-1] - history[-2]) < tol:
                break

        return history

    # =========================================================================
    # 5. PROYECCIÓN DINÁMICA DE TRAYECTORIA FUTURA
    # =========================================================================
    def project_future_symbols(
        self,
        current_state: int,
        max_symbols: int = 50,
        spent_in_current: int = 0,
    ) -> list[int]:
        """Proyecta la secuencia futura de símbolos esperados a partir del estado actual.

        Ecuaciones:
        1. Permanencia esperada en cada estado j (distribución geométrica):
           E[d_j] = 1 / (1 - a_{jj})
           En el estado actual q_t*, la permanencia restante es max(0, round(E[d_j]) - spent_in_current).
        2. Símbolo más probable emitido en el estado j:
           símbolo_esperado(j) = argmax_k [ b_j(k) ]
        3. Recorrido Bakis de izquierda a derecha desde current_state hasta N-1:
           Se emite el símbolo representativo de j durante los pasos esperados
           y luego se transiciona al siguiente estado más probable:
           j_next = argmax_{k > j} [ a_{jk} ].

        Args:
            current_state: Estado oculto actual q_t* estimado por Viterbi.
            max_symbols: Cota máxima de símbolos generados para evitar bucles.
            spent_in_current: Pasos ya consumidos en current_state al final del camino Viterbi.

        Returns:
            Lista de símbolos proyectados para reconstruir la trayectoria.
        """
        N = self.n_states
        if current_state < 0 or current_state >= N:
            raise ValueError(f"Estado inicial inválido: {current_state}")

        projected: list[int] = []
        state = current_state
        first_state = True

        while state < N and len(projected) < max_symbols:
            # 1. Permanencia esperada
            a_stay = self.A[state, state]
            if a_stay >= 0.999:
                dwell = 2
            else:
                dwell = int(round(1.0 / (1.0 - a_stay)))
                dwell = max(1, min(dwell, 10))

            if first_state and spent_in_current > 0:
                dwell = max(0, dwell - spent_in_current)
            first_state = False

            # 2. Símbolo característico del estado
            sym = int(np.argmax(self.B[state]))
            for _ in range(dwell):
                projected.append(sym)
                if len(projected) >= max_symbols:
                    break

            if state == N - 1:
                break

            # 3. Avanzar al siguiente estado
            next_candidates = self.A[state, state + 1 :]
            if len(next_candidates) == 0 or np.sum(next_candidates) < 1e-6:
                state += 1
            else:
                state = state + 1 + int(np.argmax(next_candidates))

        return projected

    # =========================================================================
    # 5. SERIALIZACIÓN (JSON)
    # =========================================================================
    def to_dict(self) -> dict:
        """Convierte los parámetros a un diccionario serializable."""
        return {
            "letter": self.letter,
            "n_states": self.n_states,
            "n_symbols": self.n_symbols,
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "pi": self.pi.tolist(),
        }

    def save_json(self, path: str | Path) -> None:
        """Guarda los parámetros del modelo en un archivo JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> HMM:
        """Instancia un HMM a partir de un diccionario."""
        return cls(
            letter=data["letter"],
            n_states=int(data["n_states"]),
            n_symbols=int(data["n_symbols"]),
            A=np.array(data["A"], dtype=np.float64),
            B=np.array(data["B"], dtype=np.float64),
            pi=np.array(data["pi"], dtype=np.float64),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> HMM:
        """Carga un modelo HMM desde un archivo JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# 6. CLASIFICADOR MULTICLASE (27 MODELOS)
# =============================================================================
class HMMClassifier:
    """Clasificador multicolección de 27 modelos HMM {λ_c}."""

    def __init__(self, models: dict[str, HMM] | None = None) -> None:
        self.models: dict[str, HMM] = models if models is not None else {}

    def add_model(self, hmm: HMM) -> None:
        self.models[hmm.letter] = hmm

    def classify(
        self, observations: Sequence[int]
    ) -> list[tuple[str, float, float]]:
        """Calcula probabilidades a posteriori P(c | O) para cada letra c.

        Ecuaciones:
        1. Verosimilitud por clase: L_c = log P(O | λ_c) (obtenida con Forward)
        2. Log-Sum-Exp para la constante de normalización:
           m = max_c L_c
           log Σ = m + log( Σ_c exp(L_c - m) )
        3. Probabilidad a posteriori (suponiendo prior uniforme P(c) = 1/27):
           P(c | O) = exp( L_c - log Σ )

        Returns:
            Lista ordenada descendentemente: [(letra, prob_posteriori, log_verosimilitud), ...]
        """
        if not self.models or len(observations) == 0:
            return [(letter, 1.0 / max(1, len(self.models)), float("-inf")) for letter in self.models]

        log_likes: dict[str, float] = {}
        for letter, model in self.models.items():
            _, _, log_p = model.forward(observations)
            log_likes[letter] = log_p

        max_log = max(log_likes.values())
        if np.isneginf(max_log):
            uniform = 1.0 / len(self.models)
            return [(letter, uniform, float("-inf")) for letter in sorted(self.models.keys())]

        sum_exp = sum(np.exp(lp - max_log) for lp in log_likes.values())
        log_denom = max_log + float(np.log(sum_exp))

        results = []
        for letter, log_p in log_likes.items():
            post = float(np.exp(log_p - log_denom))
            results.append((letter, post, log_p))

        # Ordenar descendente por probabilidad posterior
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def predict_and_project(
        self, observations: Sequence[int], max_symbols: int = 45
    ) -> tuple[list[tuple[str, float, float]], str | None, int | None, list[int]]:
        """Clasifica O_{1:t}, decodifica el estado actual q_t* con Viterbi y proyecta símbolos futuros.

        Ecuaciones (SPEC §3.5):
        1. Ranking posterior con Forward: P(c | O_{1:t}) ∀ c ∈ C
        2. Carácter más probable: ĉ = argmax_{c ∈ C} P(c | O_{1:t})
        3. Estado oculto actual por Viterbi en λ_{ĉ}:
           Q* = argmax_Q P(Q | O_{1:t}, λ_{ĉ}),   q_t* = Q*_t
        4. Símbolos futuros proyectados desde q_t*: {Ô_{t+1}, ..., Ô_T}.
        """
        ranking = self.classify(observations)
        if not ranking or len(observations) == 0:
            return ranking, None, None, []

        top_letter = ranking[0][0]
        top_model = self.models[top_letter]
        best_path, _ = top_model.viterbi(observations)
        if not best_path:
            return ranking, top_letter, None, []

        current_state = int(best_path[-1])
        spent = 0
        for s in reversed(best_path):
            if s == current_state:
                spent += 1
            else:
                break

        future_symbols = top_model.project_future_symbols(
            current_state=current_state,
            max_symbols=max_symbols,
            spent_in_current=spent,
        )
        return ranking, top_letter, current_state, future_symbols

    @classmethod
    def load_from_dir(cls, directory: str | Path, pattern: str = "2d_*.json") -> HMMClassifier:
        """Carga todos los modelos de un directorio según el patrón."""
        p = Path(directory)
        classifier = cls()
        for file in sorted(p.glob(pattern)):
            hmm = HMM.load_json(file)
            classifier.add_model(hmm)
        return classifier

