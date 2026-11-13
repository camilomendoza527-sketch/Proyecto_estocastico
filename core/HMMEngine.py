"""HMMEngine: Implementación nativa de Modelos Ocultos de Markov (HMM).

Restricciones estrictas (SPEC.pdf §3):
- Solo NumPy para operaciones matemáticas y algebraicas.
- Cero librerías externas para lógica HMM o estadística.
- Forward y Backward utilizan factores de escala c_t (Rabiner, 1989).
- Viterbi opera en espacio logarítmico (log-space).
- Topología Bakis izquierda-derecha con estado inicial fijo π = [1, 0, ..., 0].
- Modelado conjunto de secuencia (Forward), duración (μ_c, σ_c) y morfología global (g).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_SHAPE_MIN_STD = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10], dtype=np.float64)


@dataclass
class HMM:
    """Modelo Oculto de Markov discreto λ = (A, B, π) con modelo de duración y forma."""

    letter: str
    n_states: int
    n_symbols: int
    A: np.ndarray
    B: np.ndarray
    pi: np.ndarray
    expected_length: float = 0.0
    length_std: float = 0.0
    expected_jumps: float = -1.0
    shape_mean: np.ndarray | None = None
    shape_std: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        self.pi = np.asarray(self.pi, dtype=np.float64)
        self.expected_length = float(self.expected_length)
        self.length_std = float(self.length_std)
        self.expected_jumps = float(self.expected_jumps)
        if self.shape_mean is not None:
            self.shape_mean = np.asarray(self.shape_mean, dtype=np.float64)
        if self.shape_std is not None:
            self.shape_std = np.asarray(self.shape_std, dtype=np.float64)
        self._validate_shapes_and_stochastic()
        self._refresh_log_matrices()

    def _refresh_log_matrices(self) -> None:
        """Precalcula log(A), log(B) y log(π) para Viterbi en log-space."""
        self._log_A = np.log(self.A + 1e-300)
        self._log_B = np.log(self.B + 1e-300)
        self._log_pi = np.log(self.pi + 1e-300)

    def _validate_shapes_and_stochastic(self) -> None:
        """Verifica dimensiones y que las filas sumen 1 (propiedad estocástica)."""
        N, M = self.n_states, self.n_symbols
        if self.A.shape != (N, N):
            raise ValueError(f"A debe tener forma ({N}, {N}), tiene {self.A.shape}")
        if self.B.shape != (N, M):
            raise ValueError(f"B debe tener forma ({N}, {M}), tiene {self.B.shape}")
        if self.pi.shape != (N,):
            raise ValueError(f"pi debe tener forma ({N},), tiene {self.pi.shape}")
        if not np.allclose(self.A.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Las filas de A deben sumar 1: {self.A.sum(axis=1)}")
        if not np.allclose(self.B.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Las filas de B deben sumar 1: {self.B.sum(axis=1)}")
        if not np.isclose(self.pi.sum(), 1.0, atol=1e-5):
            raise ValueError(f"El vector pi debe sumar 1: {self.pi.sum()}")

    def length_deviation_penalty(self, n_obs: int, complete: bool = False) -> float:
        """Penalización estocástica de duración según longitud de arco observada T.

        Ecuación:
        - En trazo finalizado (complete=True):
            P_dur(T) = 0.5 · ((T - μ_c) / max(σ_c, 4.0))²
        - En trazo parcial en curso (complete=False):
            Solo penaliza si T > μ_c + 1.5·σ_c para no castigar prefijos cortos.
        """
        if self.expected_length <= 0.0 or n_obs <= 0:
            return 0.0
        sigma = max(self.length_std, 4.0)
        if complete:
            z = (float(n_obs) - self.expected_length) / sigma
            return 0.5 * (z * z)
        threshold = self.expected_length + 1.5 * sigma
        if n_obs <= threshold:
            return 0.0
        z = (float(n_obs) - threshold) / sigma
        return 0.5 * (z * z)

    def shape_penalty(
        self, shape: np.ndarray | None, n_obs: int, complete: bool = False
    ) -> float:
        """Penalización morfológica gaussiana sobre el vector geométrico global g (7D).

        Ecuación:
            P_shape(g) = w · 0.5 · Σ_{d=1}^7 ( (g_d - μ_{shape,c,d}) / σ_{shape,c,d} )²
        """
        if shape is None or self.shape_mean is None or self.shape_std is None or n_obs <= 0:
            return 0.0
        g = np.asarray(shape, dtype=np.float64)
        if g.shape != self.shape_mean.shape:
            return 0.0

        if complete:
            w = 1.4
        elif self.expected_length > 0.0 and n_obs >= 0.85 * self.expected_length:
            progress = min(
                1.0,
                (float(n_obs) - 0.85 * self.expected_length)
                / max(1.0, 0.15 * self.expected_length),
            )
            w = 0.45 * progress
        else:
            return 0.0

        z = (g - self.shape_mean) / np.maximum(self.shape_std, 1e-6)
        return float(w * 0.5 * np.sum(z * z))

    def jump_penalty(self, observations: Sequence[int], complete: bool = False) -> float:
        """Penalización estructural por cantidad de saltos (pluma levantada) entre sub-trazos."""
        if self.expected_jumps < 0.0 or self.n_symbols < 12 or len(observations) == 0:
            return 0.0
        K = self.n_symbols - 4
        obs_jumps = sum(1 for o in observations if int(o) >= K)
        # Si el usuario ya levantó la pluma (≥1 salto) y el modelo es de 1 solo trazo (ej. l, v, n, h)
        if obs_jumps >= 1 and self.expected_jumps <= 0.12:
            return 16.0 * float(obs_jumps)
        # Si el trazo terminó con 0 saltos y el modelo exige ≥2 sub-trazos (ej. i, j, ñ, t, x, f, k)
        if obs_jumps == 0 and self.expected_jumps >= 0.85:
            if complete:
                return 24.0
            if self.expected_length > 0.0 and len(observations) > 0.72 * self.expected_length:
                return 12.0
        return 0.0

    # =========================================================================
    # 1. EVALUACIÓN: Algoritmo Forward con Escalado (Rabiner, 1989)
    # =========================================================================
    def forward(self, observations: Sequence[int]) -> tuple[np.ndarray, np.ndarray, float]:
        """Resuelve P(O | λ) con factores de escala c_t: log P(O | λ) = -Σ_t log(c_t)."""
        T = len(observations)
        N = self.n_states
        if T == 0:
            return np.empty((0, N)), np.empty(0), float("-inf")

        alpha_hat = np.zeros((T, N), dtype=np.float64)
        c = np.zeros(T, dtype=np.float64)

        alpha_raw_0 = self.pi * self.B[:, observations[0]]
        c[0] = 1.0 / (np.sum(alpha_raw_0) + 1e-300)
        alpha_hat[0] = c[0] * alpha_raw_0

        for t in range(T - 1):
            alpha_raw_next = np.dot(alpha_hat[t], self.A) * self.B[:, observations[t + 1]]
            c[t + 1] = 1.0 / (np.sum(alpha_raw_next) + 1e-300)
            alpha_hat[t + 1] = c[t + 1] * alpha_raw_next

        log_prob = -float(np.sum(np.log(c + 1e-300)))
        return alpha_hat, c, log_prob

    # =========================================================================
    # 2. EVALUACIÓN: Algoritmo Backward con Escalado
    # =========================================================================
    def backward(self, observations: Sequence[int], c: np.ndarray) -> np.ndarray:
        """Algoritmo Backward usando los coeficientes de escala c_t de Forward."""
        T = len(observations)
        N = self.n_states
        if T == 0:
            return np.empty((0, N))

        beta_hat = np.zeros((T, N), dtype=np.float64)
        beta_hat[T - 1] = c[T - 1]

        for t in range(T - 2, -1, -1):
            beta_raw = np.dot(self.A, self.B[:, observations[t + 1]] * beta_hat[t + 1])
            beta_hat[t] = c[t] * beta_raw

        return beta_hat

    # =========================================================================
    # 3. DECODIFICACIÓN: Algoritmo de Viterbi en Log-Espacio
    # =========================================================================
    def viterbi(self, observations: Sequence[int]) -> tuple[list[int], float]:
        """Encuentra la secuencia de estados ocultos más probable Q* en log-space."""
        T = len(observations)
        N = self.n_states
        if T == 0:
            return [], float("-inf")

        delta = np.zeros((T, N), dtype=np.float64)
        psi = np.zeros((T, N), dtype=int)

        delta[0] = self._log_pi + self._log_B[:, observations[0]]
        for t in range(1, T):
            transitions = delta[t - 1, :, None] + self._log_A
            psi[t] = np.argmax(transitions, axis=0)
            delta[t] = np.max(transitions, axis=0) + self._log_B[:, observations[t]]

        best_last_state = int(np.argmax(delta[T - 1]))
        best_log_prob = float(delta[T - 1, best_last_state])

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
        shapes: Sequence[np.ndarray] | None = None,
    ) -> list[float]:
        """Entrenamiento offline por Esperanza-Maximización (Baum-Welch) multisequencia."""
        N, M = self.n_states, self.n_symbols
        if not sequences:
            return []

        history: list[float] = []
        mask_A = self.A > 0

        for iteration in range(max_iter):
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

                gamma = (alpha_hat * beta_hat) / (c[:, None] + 1e-300)
                gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-300

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

                if T > 1:
                    A_den += np.sum(gamma[:-1], axis=0)

                np.add.at(B_num.T, np.asarray(seq, dtype=np.intp), gamma)
                B_den += np.sum(gamma, axis=0)

            history.append(total_log_prob)

            # En topología Bakis izquierda-derecha, el estado inicial es fijo: π = [1, 0, ..., 0]
            self.pi = np.zeros(N, dtype=np.float64)
            self.pi[0] = 1.0

            valid_A = A_den[:, None] > 0
            new_A = np.where(valid_A, A_num / np.maximum(A_den[:, None], 1e-300), self.A)
            new_A = (new_A + pseudo_count) * mask_A
            self.A = new_A / (np.sum(new_A, axis=1, keepdims=True) + 1e-300)

            valid_B = B_den[:, None] > 0
            new_B = np.where(
                valid_B,
                (B_num + pseudo_count) / (B_den[:, None] + M * pseudo_count),
                1.0 / M,
            )
            self.B = new_B / (np.sum(new_B, axis=1, keepdims=True) + 1e-300)
            self._refresh_log_matrices()

            if iteration > 0 and abs(history[-1] - history[-2]) < tol:
                break

        valid_lengths = [len(seq) for seq in sequences if len(seq) > 0]
        if valid_lengths:
            self.expected_length = float(np.mean(valid_lengths))
            self.length_std = float(max(np.std(valid_lengths), 4.0))

        # Suavizado circular de sectores angulares vecinos (tolerancia angular ±1 sector) y conteo de saltos
        if M >= 12:
            K = M - 4
            dir_B = self.B[:, :K].copy()
            smoothed_dir = (
                0.78 * dir_B
                + 0.11 * np.roll(dir_B, 1, axis=1)
                + 0.11 * np.roll(dir_B, -1, axis=1)
            )
            self.B[:, :K] = smoothed_dir
            self.B /= np.sum(self.B, axis=1, keepdims=True) + 1e-300
            jump_counts = [sum(1 for o in seq if int(o) >= K) for seq in sequences if len(seq) > 0]
            if jump_counts:
                self.expected_jumps = float(np.mean(jump_counts))

        if shapes:
            arr_shapes = np.asarray(shapes, dtype=np.float64)
            if arr_shapes.ndim == 2 and arr_shapes.shape[0] > 0:
                self.shape_mean = np.mean(arr_shapes, axis=0)
                min_std = DEFAULT_SHAPE_MIN_STD[: arr_shapes.shape[1]]
                self.shape_std = np.maximum(np.std(arr_shapes, axis=0), min_std)

        self._refresh_log_matrices()
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
        """Proyecta la secuencia futura de símbolos esperados desde current_state."""
        N = self.n_states
        if current_state < 0 or current_state >= N:
            raise ValueError(f"Estado inicial inválido: {current_state}")

        projected: list[int] = []
        state = current_state
        first_state = True

        while state < N and len(projected) < max_symbols:
            a_stay = self.A[state, state]
            if a_stay >= 0.999:
                dwell = 2
            else:
                dwell = max(1, min(int(round(1.0 / (1.0 - a_stay))), 10))

            if first_state and spent_in_current > 0:
                dwell = max(0, dwell - spent_in_current)
            first_state = False

            sym = int(np.argmax(self.B[state]))
            for _ in range(dwell):
                projected.append(sym)
                if len(projected) >= max_symbols:
                    break

            if state == N - 1:
                break

            next_candidates = self.A[state, state + 1 :]
            if len(next_candidates) == 0 or np.sum(next_candidates) < 1e-6:
                state += 1
            else:
                state = state + 1 + int(np.argmax(next_candidates))

        return projected

    # =========================================================================
    # 6. SERIALIZACIÓN (JSON)
    # =========================================================================
    def to_dict(self) -> dict:
        """Convierte los parámetros a un diccionario serializable."""
        d: dict = {
            "letter": self.letter,
            "n_states": self.n_states,
            "n_symbols": self.n_symbols,
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "pi": self.pi.tolist(),
        }
        if self.expected_length > 0.0:
            d["expected_length"] = round(self.expected_length, 3)
            d["length_std"] = round(self.length_std, 3)
        if self.expected_jumps >= 0.0:
            d["expected_jumps"] = round(self.expected_jumps, 3)
        if self.shape_mean is not None and self.shape_std is not None:
            d["shape_mean"] = [round(float(x), 4) for x in self.shape_mean]
            d["shape_std"] = [round(float(x), 4) for x in self.shape_std]
        return d

    def save_json(self, path: str | Path) -> None:
        """Guarda los parámetros del modelo en un archivo JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> HMM:
        """Instancia un HMM a partir de un diccionario."""
        shape_mean = (
            np.array(data["shape_mean"], dtype=np.float64)
            if "shape_mean" in data and data["shape_mean"] is not None
            else None
        )
        shape_std = (
            np.array(data["shape_std"], dtype=np.float64)
            if "shape_std" in data and data["shape_std"] is not None
            else None
        )
        return cls(
            letter=data["letter"],
            n_states=int(data["n_states"]),
            n_symbols=int(data["n_symbols"]),
            A=np.array(data["A"], dtype=np.float64),
            B=np.array(data["B"], dtype=np.float64),
            pi=np.array(data["pi"], dtype=np.float64),
            expected_length=float(data.get("expected_length", 0.0)),
            length_std=float(data.get("length_std", 0.0)),
            expected_jumps=float(data.get("expected_jumps", -1.0)),
            shape_mean=shape_mean,
            shape_std=shape_std,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> HMM:
        """Carga un modelo HMM desde un archivo JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# 7. INICIALIZACIÓN DE TOPOLOGÍA BAKIS (IZQUIERDA-DERECHA)
# =============================================================================
def init_bakis_hmm(letter: str, n_states: int, n_symbols: int) -> HMM:
    """Crea un HMM con topología Bakis izquierda-derecha (a_ij > 0 solo para j ∈ {i, i+1, i+2})."""
    pi = np.zeros(n_states, dtype=np.float64)
    pi[0] = 1.0

    A = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        if i == n_states - 1:
            A[i, i] = 1.0
        elif i == n_states - 2:
            A[i, i] = 0.65
            A[i, i + 1] = 0.35
        else:
            A[i, i] = 0.6
            A[i, i + 1] = 0.3
            A[i, i + 2] = 0.1

    B = np.full((n_states, n_symbols), 1.0 / n_symbols, dtype=np.float64)
    return HMM(letter=letter, n_states=n_states, n_symbols=n_symbols, A=A, B=B, pi=pi)


def init_segmental_bakis_hmm(
    letter: str,
    n_states: int,
    n_symbols: int,
    sequences: list[list[int]] | list[np.ndarray],
    pseudo_count: float = 0.1,
) -> HMM:
    """Inicializa un HMM Bakis partiendo cada secuencia en N segmentos uniformes."""
    hmm = init_bakis_hmm(letter, n_states, n_symbols)
    if not sequences:
        return hmm

    mask_A = hmm.A > 0
    A_counts = pseudo_count * mask_A.astype(np.float64)
    B_counts = np.full((n_states, n_symbols), pseudo_count, dtype=np.float64)

    for seq in sequences:
        T = len(seq)
        if T == 0:
            continue
        states = np.minimum(n_states - 1, (np.arange(T) * n_states) // T)
        for t, s in enumerate(states):
            B_counts[s, int(seq[t])] += 1.0
        for t in range(T - 1):
            s_cur, s_nxt = int(states[t]), int(states[t + 1])
            if mask_A[s_cur, s_nxt]:
                A_counts[s_cur, s_nxt] += 1.0

    A = A_counts / A_counts.sum(axis=1, keepdims=True)
    B = B_counts / B_counts.sum(axis=1, keepdims=True)
    return HMM(letter=letter, n_states=n_states, n_symbols=n_symbols, A=A, B=B, pi=hmm.pi)


# =============================================================================
# 8. CLASIFICADOR MULTICLASE (27 MODELOS)
# =============================================================================
class HMMClassifier:
    """Clasificador multiclase de 27 modelos HMM {λ_c}."""

    def __init__(self, models: dict[str, HMM] | None = None) -> None:
        self.models: dict[str, HMM] = models if models is not None else {}

    def add_model(self, hmm: HMM) -> None:
        self.models[hmm.letter] = hmm

    def classify(
        self,
        observations: Sequence[int],
        shape: np.ndarray | None = None,
        complete: bool = False,
    ) -> list[tuple[str, float, float]]:
        """Calcula probabilidades a posteriori P(c | O) integrando Forward, duración y forma."""
        if not self.models or len(observations) == 0:
            return [
                (letter, 1.0 / max(1, len(self.models)), float("-inf"))
                for letter in self.models
            ]

        log_likes: dict[str, float] = {}
        n_obs = len(observations)
        for letter, model in self.models.items():
            _, _, log_p = model.forward(observations)
            pen_len = model.length_deviation_penalty(n_obs, complete=complete)
            pen_shape = model.shape_penalty(shape, n_obs, complete=complete)
            pen_jump = model.jump_penalty(observations, complete=complete)
            log_likes[letter] = log_p - pen_len - pen_shape - pen_jump

        max_log = max(log_likes.values())
        if np.isneginf(max_log):
            uniform = 1.0 / len(self.models)
            return [(letter, uniform, float("-inf")) for letter in sorted(self.models.keys())]

        sum_exp = sum(np.exp(lp - max_log) for lp in log_likes.values())
        log_denom = max_log + float(np.log(sum_exp))

        results = [
            (letter, float(np.exp(log_p - log_denom)), log_p)
            for letter, log_p in log_likes.items()
        ]
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def predict_and_project(
        self,
        observations: Sequence[int],
        max_symbols: int = 45,
        shape: np.ndarray | None = None,
        complete: bool = False,
    ) -> tuple[list[tuple[str, float, float]], str | None, int | None, list[int]]:
        """Clasifica O_{1:t}, decodifica q_t* con Viterbi y proyecta símbolos futuros."""
        ranking = self.classify(observations, shape=shape, complete=complete)
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
            classifier.add_model(HMM.load_json(file))
        return classifier
