"""Pruebas del pipeline de entrenamiento, plantillas 2D y evaluación."""

from core.config import get_config
from core.FeatureExtractor import FeatureExtractor2D
from training.augment import augment_strokes, canonical_unit_strokes
from training.evaluate import evaluate_k_2d


def test_canonical_templates_exist_for_all_27_letters():
    cfg = get_config()
    fx = FeatureExtractor2D()
    box = cfg.features.expected_letter_size
    for letter in cfg.letters:
        strokes = [s * box for s in canonical_unit_strokes(letter)]
        res = fx.extract(strokes, realtime=False)
        assert len(res.symbols) >= 4, f"Plantilla muy corta para {letter}"


def test_multistroke_conventions_in_templates():
    fx = FeatureExtractor2D()
    box = fx.cfg.expected_letter_size
    # 'i', 'j', 'ñ' deben emitir salto hacia arriba (K+1) después del cuerpo
    for letter in ("i", "j", "ñ"):
        s = fx.extract([st * box for st in canonical_unit_strokes(letter)]).symbols
        assert (fx.K + 1) in s
    # 't' y 'x' tienen dos sub-trazos y el segundo va de izquierda a derecha (sector 0 o diagonal der)
    for letter in ("t", "x"):
        strokes = canonical_unit_strokes(letter)
        assert len(strokes) == 2
        assert strokes[1][-1, 0] > strokes[1][0, 0]


def test_evaluate_compares_prefixes(tmp_path):
    from training.augment import generate_seed_dataset_2d

    generate_seed_dataset_2d(samples_per_letter=6, base_dir=tmp_path, only_if_empty=True)
    rep = evaluate_k_2d(K=16, raw_dir=tmp_path, augment_factor=0, max_iter=10)
    assert set(rep.top1_by_prefix.keys()) == set(get_config().prefix_fractions)
    # Con el trazo completo (100%) la precisión debe superar ampliamente el azar (1/27)
    assert rep.top1_by_prefix[1.0] > 0.75


def test_map_2d_to_sphere_and_evaluate_3d(tmp_path):
    import numpy as np
    from core.FeatureExtractor import FeatureExtractor3D
    from training.augment import generate_seed_dataset_3d, map_2d_to_sphere
    from training.evaluate import evaluate_k_3d

    cfg = get_config()
    fx3 = FeatureExtractor3D()
    box = cfg.features.expected_letter_size
    for letter in ("a", "ñ", "t", "x"):
        strokes_2d = [s * box for s in canonical_unit_strokes(letter)]
        strokes_3d = map_2d_to_sphere(strokes_2d, radius=2.0)
        for s3 in strokes_3d:
            radii = np.linalg.norm(s3, axis=1)
            np.testing.assert_allclose(radii, 2.0, atol=1e-6)
        res3 = fx3.extract(strokes_3d, realtime=False)
        assert len(res3.symbols) >= 4

    generate_seed_dataset_3d(samples_per_letter=6, base_dir=tmp_path, only_if_empty=True)
    rep3 = evaluate_k_3d(K=16, raw_dir=tmp_path, augment_factor=0, max_iter=10)
    assert rep3.top1_by_prefix[1.0] > 0.75

