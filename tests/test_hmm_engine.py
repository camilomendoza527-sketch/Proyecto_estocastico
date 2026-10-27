"""Pruebas unitarias matemáticas para HMMEngine.py.

Verifica:
- Propiedad estocástica y validación de matrices.
- Algoritmo Forward escalado y comparación de log-verosimilitud.
- Algoritmo Backward escalado y consistencia con Forward.
- Algoritmo Viterbi en log-space contra cálculo manual exacto.
- Proyección futura de estados y símbolos.
- Log-Sum-Exp en el clasificador multicolección.
- Serialización y carga JSON.
"""

import numpy as np
import pytest

from core.HMMEngine import HMM, HMMClassifier


@pytest.fixture
def simple_hmm():
    """HMM de juguete con 2 estados y 3 observaciones verificable a mano.

    S1 = Estado 0, S2 = Estado 1
    Observaciones: 0, 1, 2
    """
    pi = np.array([0.6, 0.4])
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6],
    ])
    B = np.array([
        [0.5, 0.4, 0.1],
        [0.1, 0.3, 0.6],
    ])
    return HMM(letter="a", n_states=2, n_symbols=3, A=A, B=B, pi=pi)


def test_validation_stochastic_matrices():
    # Matriz con suma != 1 debe fallar
    bad_A = np.array([[0.5, 0.2], [0.4, 0.6]])
    pi = np.array([0.5, 0.5])
    B = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError):
        HMM(letter="x", n_states=2, n_symbols=2, A=bad_A, B=B, pi=pi)


def test_forward_algorithm_against_manual_calculation(simple_hmm):
    """Cálculo manual para O = (0, 1):
    t = 1 (obs=0):
      α̃_1(0) = π_0 * B_0(0) = 0.6 * 0.5 = 0.30
      α̃_1(1) = π_1 * B_1(0) = 0.4 * 0.1 = 0.04
      c_1 = 1 / (0.30 + 0.04) = 1 / 0.34
      α̂_1(0) = 0.30 / 0.34 = 15/17
      α̂_1(1) = 0.04 / 0.34 = 2/17

    t = 2 (obs=1):
      α̃_2(0) = [α̂_1(0)*A_00 + α̂_1(1)*A_10] * B_0(1)
             = [(15/17)*0.7 + (2/17)*0.4] * 0.4
             = [(10.5 + 0.8)/17] * 0.4 = (11.3 / 17) * 0.4 = 4.52 / 17 ≈ 0.26588235
      α̃_2(1) = [α̂_1(0)*A_01 + α̂_1(1)*A_11] * B_1(1)
             = [(15/17)*0.3 + (2/17)*0.6] * 0.3
             = [(4.5 + 1.2)/17] * 0.3 = (5.7 / 17) * 0.3 = 1.71 / 17 ≈ 0.10058823
      Suma_2 = (4.52 + 1.71) / 17 = 6.23 / 17 ≈ 0.36647059
      c_2 = 17 / 6.23
      P(O) sin escalar = 0.34 * (6.23 / 17) = 0.34 * 0.36647059 = 0.1246
      log P(O) = log(0.1246) ≈ -2.08269
    """
    obs = [0, 1]
    alpha_hat, c, log_prob = simple_hmm.forward(obs)

    # Las filas de alpha_hat deben sumar 1
    np.testing.assert_allclose(alpha_hat.sum(axis=1), [1.0, 1.0])

    # Log probabilidad teórica
    expected_prob = 0.1246
    np.testing.assert_allclose(np.exp(log_prob), expected_prob, rtol=1e-5)


def test_forward_step_consistency(simple_hmm):
    obs = [0, 1, 2, 0]
    alpha_hat, c, full_log_prob = simple_hmm.forward(obs)

    # Replicar con forward_step
    # Paso 0
    o0 = obs[0]
    alpha_raw_0 = simple_hmm.pi * simple_hmm.B[:, o0]
    c0 = 1.0 / np.sum(alpha_raw_0)
    current_alpha = c0 * alpha_raw_0
    accum_log_prob = float(np.log(np.sum(alpha_raw_0)))

    for t in range(1, len(obs)):
        current_alpha, delta_log = simple_hmm.forward_step(current_alpha, obs[t])
        accum_log_prob += delta_log
        np.testing.assert_allclose(current_alpha, alpha_hat[t], rtol=1e-5)

    np.testing.assert_allclose(accum_log_prob, full_log_prob, rtol=1e-5)


def test_viterbi_against_manual_calculation(simple_hmm):
    """Para O = (0, 2):
    Caminos posibles:
    0 -> 0: π_0 * B_0(0) * A_00 * B_0(2) = 0.6 * 0.5 * 0.7 * 0.1 = 0.0210
    0 -> 1: π_0 * B_0(0) * A_01 * B_1(2) = 0.6 * 0.5 * 0.3 * 0.6 = 0.0540
    1 -> 0: π_1 * B_1(0) * A_10 * B_0(2) = 0.4 * 0.1 * 0.4 * 0.1 = 0.0016
    1 -> 1: π_1 * B_1(0) * A_11 * B_1(2) = 0.4 * 0.1 * 0.6 * 0.6 = 0.0144

    El mejor camino es 0 -> 1 con probabilidad 0.0540.
    """
    obs = [0, 2]
    best_path, best_log_p = simple_hmm.viterbi(obs)

    assert best_path == [0, 1]
    np.testing.assert_allclose(np.exp(best_log_p), 0.0540, rtol=1e-5)


def test_backward_variables_consistency(simple_hmm):
    """En cualquier t: Σ_i α̂_t(i) * β̂_t(i) / c_t = 1 (propiedad de escala Rabiner)."""
    obs = [0, 1, 2]
    alpha_hat, c, _ = simple_hmm.forward(obs)
    beta_hat = simple_hmm.backward(obs, c)

    for t in range(len(obs)):
        gamma_sum = np.sum(alpha_hat[t] * beta_hat[t]) / c[t]
        np.testing.assert_allclose(gamma_sum, 1.0, rtol=1e-5)


def test_projection_future_symbols(simple_hmm):
    projected = simple_hmm.project_future_symbols(current_state=0, max_symbols=5)
    assert len(projected) > 0
    # Cada símbolo emitido debe estar en [0, n_symbols-1]
    assert all(0 <= sym < simple_hmm.n_symbols for sym in projected)


def test_classifier_multiclass_and_log_sum_exp(simple_hmm):
    # Crear un segundo modelo con diferente preferencia
    pi2 = np.array([0.1, 0.9])
    A2 = np.array([[0.5, 0.5], [0.2, 0.8]])
    B2 = np.array([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
    hmm_b = HMM(letter="b", n_states=2, n_symbols=3, A=A2, B=B2, pi=pi2)

    clf = HMMClassifier()
    clf.add_model(simple_hmm)
    clf.add_model(hmm_b)

    # Observación con alta probabilidad en simple_hmm (0, 1)
    results = clf.classify([0, 1])
    assert len(results) == 2
    top_letter, top_prob, _ = results[0]
    assert top_letter == "a"
    assert top_prob > 0.5
    # Las probabilidades posteriores deben sumar 1
    assert pytest.approx(sum(r[1] for r in results)) == 1.0


def test_json_serialization(simple_hmm, tmp_path):
    json_path = tmp_path / "model_a.json"
    simple_hmm.save_json(json_path)

    loaded_hmm = HMM.load_json(json_path)
    assert loaded_hmm.letter == simple_hmm.letter
    assert loaded_hmm.n_states == simple_hmm.n_states
    assert loaded_hmm.n_symbols == simple_hmm.n_symbols
    np.testing.assert_allclose(loaded_hmm.A, simple_hmm.A)
    np.testing.assert_allclose(loaded_hmm.B, simple_hmm.B)
    np.testing.assert_allclose(loaded_hmm.pi, simple_hmm.pi)


def test_baum_welch_convergence_increases_likelihood():
    from core.topology import init_bakis_hmm

    # Modelo inicial Bakis de 3 estados y 4 símbolos
    hmm = init_bakis_hmm(letter="c", n_states=3, n_symbols=4)

    # Secuencias consistentes (ej. observan mayormente 0, luego 1, luego 2)
    training_data = [
        [0, 0, 1, 1, 2],
        [0, 1, 1, 2, 2],
        [0, 0, 1, 2],
    ]

    history = hmm.fit(training_data, max_iter=20, tol=1e-5)
    assert len(history) > 1
    # La verosimilitud debe ser monótonamente creciente (o casi idéntica por tolerancia)
    assert history[-1] >= history[0]

