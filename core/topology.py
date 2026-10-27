"""Inicializador de HMM con topología Bakis (Izquierda-Derecha)."""

import numpy as np
from core.HMMEngine import HMM


def init_bakis_hmm(letter: str, n_states: int, n_symbols: int) -> HMM:
    """Crea un HMM con topología Bakis izquierda-derecha (a_ij > 0 solo para j in {i, i+1, i+2}).

    Ecuaciones:
    1. π = [1, 0, 0, ..., 0]  (siempre inicia en el primer estado)
    2. a_ii = 0.6, a_{i, i+1} = 0.3, a_{i, i+2} = 0.1  (normalizado por fila)
    3. B uniforme con perturbación suave o inicialización uniforme b_j(k) = 1 / M.
    """
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

    # Emisiones iniciales uniformes
    B = np.full((n_states, n_symbols), 1.0 / n_symbols, dtype=np.float64)

    return HMM(
        letter=letter,
        n_states=n_states,
        n_symbols=n_symbols,
        A=A,
        B=B,
        pi=pi,
    )


def init_segmental_bakis_hmm(
    letter: str,
    n_states: int,
    n_symbols: int,
    sequences: list[list[int]] | list[np.ndarray],
    pseudo_count: float = 0.1,
) -> HMM:
    """Inicializa un HMM Bakis partiendo cada secuencia en N tramos uniformes.

    Esta inicialización (segmental uniform alignment) rompe la simetría entre
    estados: si B fuera uniforme con A Bakis homogénea, Baum-Welch quedaría en
    un punto silla con todos los estados idénticos.

    Ecuaciones:
        Estado asignado al instante t ∈ {0, ..., T-1}:
            q_t = min( N - 1,  ⌊ t · N / T ⌋ )
        Conteos empíricos con suavizado de Laplace α = pseudo_count:
            b_j(k) = ( #{(r, t): q_t^r = j, o_t^r = k} + α ) / ( #{q_t^r = j} + M·α )
            a_{ij} = ( #{(r, t): q_t^r = i, q_{t+1}^r = j} + α·1_{j∈{i,i+1,i+2}} ) / Z_i
    """
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

