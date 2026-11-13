"""Pruebas unitarias matemáticas para HMMEngine.py."""

import numpy as np
import pytest

from core.HMMEngine import HMM, HMMClassifier, init_bakis_hmm


@pytest.fixture
def simple_hmm():
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    return HMM(letter="a", n_states=2, n_symbols=3, A=A, B=B, pi=pi)


def test_validation_stochastic_matrices():
    bad_A = np.array([[0.5, 0.2], [0.4, 0.6]])
    pi = np.array([0.5, 0.5])
    B = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError):
        HMM(letter="x", n_states=2, n_symbols=2, A=bad_A, B=B, pi=pi)


def test_forward_algorithm_against_manual_calculation(simple_hmm):
    obs = [0, 1]
    alpha_hat, c, log_prob = simple_hmm.forward(obs)
    np.testing.assert_allclose(alpha_hat.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(np.exp(log_prob), 0.1246, rtol=1e-5)


def test_viterbi_against_manual_calculation(simple_hmm):
    obs = [0, 2]
    best_path, best_log_p = simple_hmm.viterbi(obs)
    assert best_path == [0, 1]
    np.testing.assert_allclose(np.exp(best_log_p), 0.0540, rtol=1e-5)


def test_backward_variables_consistency(simple_hmm):
    obs = [0, 1, 2]
    alpha_hat, c, _ = simple_hmm.forward(obs)
    beta_hat = simple_hmm.backward(obs, c)
    for t in range(len(obs)):
        gamma_sum = np.sum(alpha_hat[t] * beta_hat[t]) / c[t]
        np.testing.assert_allclose(gamma_sum, 1.0, rtol=1e-5)


def test_projection_future_symbols(simple_hmm):
    projected = simple_hmm.project_future_symbols(current_state=0, max_symbols=5)
    assert len(projected) > 0
    assert all(0 <= sym < simple_hmm.n_symbols for sym in projected)


def test_classifier_multiclass_and_log_sum_exp(simple_hmm):
    pi2 = np.array([0.1, 0.9])
    A2 = np.array([[0.5, 0.5], [0.2, 0.8]])
    B2 = np.array([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
    hmm_b = HMM(letter="b", n_states=2, n_symbols=3, A=A2, B=B2, pi=pi2)

    clf = HMMClassifier({"a": simple_hmm, "b": hmm_b})
    results = clf.classify([0, 1])
    assert len(results) == 2
    assert results[0][0] == "a"
    assert results[0][1] > 0.5
    assert pytest.approx(sum(r[1] for r in results)) == 1.0


def test_json_serialization(simple_hmm, tmp_path):
    simple_hmm.shape_mean = np.zeros(7)
    simple_hmm.shape_std = np.ones(7) * 0.2
    json_path = tmp_path / "model_a.json"
    simple_hmm.save_json(json_path)

    loaded_hmm = HMM.load_json(json_path)
    assert loaded_hmm.letter == simple_hmm.letter
    np.testing.assert_allclose(loaded_hmm.A, simple_hmm.A)
    np.testing.assert_allclose(loaded_hmm.B, simple_hmm.B)
    np.testing.assert_allclose(loaded_hmm.pi, simple_hmm.pi)
    np.testing.assert_allclose(loaded_hmm.shape_mean, simple_hmm.shape_mean)


def test_baum_welch_keeps_pi_fixed_and_increases_likelihood():
    hmm = init_bakis_hmm(letter="c", n_states=3, n_symbols=4)
    training_data = [[0, 0, 1, 1, 2], [0, 1, 1, 2, 2], [0, 0, 1, 2]]
    history = hmm.fit(training_data, max_iter=20, tol=1e-5)
    assert len(history) > 1
    assert history[-1] >= history[0]
    np.testing.assert_allclose(hmm.pi, [1.0, 0.0, 0.0])


def test_length_and_shape_penalties_disambiguate_letters():
    pi = np.array([1.0, 0.0])
    A = np.array([[0.8, 0.2], [0.0, 1.0]])
    B = np.array([[0.5, 0.5], [0.5, 0.5]])

    hmm_a = HMM(
        letter="a",
        n_states=2,
        n_symbols=2,
        A=A,
        B=B,
        pi=pi,
        expected_length=35.0,
        length_std=4.0,
        shape_mean=np.array([0.0, 0.7, 0.3, 0.9, 0.9, 0.5, 0.5]),
        shape_std=np.full(7, 0.15),
    )
    hmm_o = HMM(
        letter="o",
        n_states=2,
        n_symbols=2,
        A=A,
        B=B,
        pi=pi,
        expected_length=35.0,
        length_std=4.0,
        shape_mean=np.array([0.0, 0.5, 0.1, 0.5, 0.1, 0.5, 0.5]),
        shape_std=np.full(7, 0.15),
    )
    clf = HMMClassifier({"a": hmm_a, "o": hmm_o})
    obs = [0] * 35

    # Morfología cerrada arriba (propia de 'o') -> 'o' gana con >95%
    shape_o = np.array([0.0, 0.5, 0.1, 0.5, 0.1, 0.5, 0.5])
    res_o = dict((c, p) for c, p, _ in clf.classify(obs, shape=shape_o, complete=True))
    assert res_o["o"] > 0.95

    # Morfología terminando abajo a la derecha (propia de 'a') -> 'a' gana con >95%
    shape_a = np.array([0.0, 0.7, 0.3, 0.9, 0.9, 0.5, 0.5])
    res_a = dict((c, p) for c, p, _ in clf.classify(obs, shape=shape_a, complete=True))
    assert res_a["a"] > 0.95
