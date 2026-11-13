"""Pruebas del pipeline de entrenamiento, dataset limpio y evaluación."""

import numpy as np

from core.config import get_config
from core.FeatureExtractor import FeatureExtractor2D, FeatureExtractor3D
from training.augment import canonical_unit_strokes, load_dataset, map_2d_to_sphere
from training.evaluate import evaluate_k_2d, evaluate_k_3d


def test_canonical_templates_and_dataset_exist_for_all_27_letters():
    cfg = get_config()
    fx = FeatureExtractor2D()
    box = cfg.features.expected_letter_size
    for letter in cfg.letters:
        strokes = [s * box for s in canonical_unit_strokes(letter)]
        res = fx.extract(strokes, realtime=False)
        assert len(res.symbols) >= 4, f"Plantilla muy corta para {letter}"
        ds = load_dataset(letter, mode="2d", max_samples=5)
        assert len(ds) >= 5


def test_multistroke_conventions_in_templates():
    fx = FeatureExtractor2D()
    box = fx.cfg.expected_letter_size
    for letter in ("i", "j", "ñ"):
        s = fx.extract([st * box for st in canonical_unit_strokes(letter)]).symbols
        assert (fx.K + 1) in s
    for letter in ("t", "x"):
        strokes = canonical_unit_strokes(letter)
        assert len(strokes) == 2
    assert canonical_unit_strokes("t")[1][-1, 0] > canonical_unit_strokes("t")[1][0, 0]


def test_evaluate_compares_prefixes_2d_and_3d():
    rep2 = evaluate_k_2d(K=16, max_samples=12, augment_factor=0, max_iter=6)
    assert set(rep2.top1_by_prefix.keys()) == set(get_config().prefix_fractions)
    assert rep2.top1_by_prefix[1.0] > 0.85

    fx3 = FeatureExtractor3D()
    box = get_config().features.expected_letter_size
    for letter in ("a", "ñ", "t", "x"):
        strokes_2d = [s * box for s in canonical_unit_strokes(letter)]
        strokes_3d = map_2d_to_sphere(strokes_2d, radius=2.0)
        for s3 in strokes_3d:
            np.testing.assert_allclose(np.linalg.norm(s3, axis=1), 2.0, atol=1e-6)
        assert len(fx3.extract(strokes_3d, realtime=False).symbols) >= 4

    rep3 = evaluate_k_3d(K=16, max_samples=12, augment_factor=0, max_iter=6)
    assert rep3.top1_by_prefix[1.0] > 0.85
