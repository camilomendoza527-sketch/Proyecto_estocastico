"""Pruebas del FeatureExtractor 2D/3D, descriptores morfológicos y configuración."""

import numpy as np
import pytest

from core.config import get_config
from core.FeatureExtractor import FeatureExtractor2D, FeatureExtractor3D, compute_shape_features


def line(p0, p1, n=50):
    """Segmento recto de p0 a p1 con n puntos, en coordenadas de pantalla."""
    return np.linspace(p0, p1, n)


@pytest.fixture
def fx():
    return FeatureExtractor2D()


def test_alphabet_is_exactly_27_spanish_letters():
    letters = get_config().letters
    assert len(letters) == 27
    assert "ñ" in letters
    assert list(letters) == list("abcdefghijklmn") + ["ñ"] + list("opqrstuvwxyz")


def test_states_defined_for_every_letter():
    cfg = get_config()
    assert set(cfg.states) == set(cfg.letters)
    assert cfg.file_name("ñ") == "enie"


@pytest.mark.parametrize("K", [8, 16])
def test_cardinal_directions(K):
    fx = FeatureExtractor2D(K=K)
    assert set(fx.extract([line((0, 0), (100, 0))]).symbols) == {0}
    assert set(fx.extract([line((0, 100), (0, 0))]).symbols) == {K // 4}
    assert set(fx.extract([line((100, 0), (0, 0))]).symbols) == {K // 2}
    assert set(fx.extract([line((0, 0), (0, 100))]).symbols) == {3 * K // 4}


def test_sector_boundaries_are_centered(fx):
    w = 2 * np.pi / 16
    assert fx.quantize_direction(w / 2 - 1e-6) == 0
    assert fx.quantize_direction(w / 2 + 1e-6) == 1
    assert fx.quantize_direction(-w / 2 + 1e-6) == 0
    assert fx.quantize_direction(np.pi) == 8


def test_scale_and_speed_invariance(fx):
    small = [line((0, 0), (30, 0)), line((30, 0), (30, 40))]
    big = [line((0, 0), (300, 0)), line((300, 0), (300, 400))]
    np.testing.assert_array_equal(fx.extract(small).symbols, fx.extract(big).symbols)

    slow = [line((0, 0), (100, 100), n=400)]
    fast = [line((0, 0), (100, 100), n=12)]
    np.testing.assert_array_equal(fx.extract(slow).symbols, fx.extract(fast).symbols)


def test_resample_constant_step(fx):
    pts = np.array([[0, 0], [0.3, 0], [0.3, 0.7]])
    q = fx.resample(pts)
    steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    assert np.all(steps <= fx.cfg.resample_step + 1e-9)


def test_letter_i_stem_then_dot(fx):
    stem = line((50, 40), (50, 140))
    dot = np.array([[50, 10]])
    s = fx.extract([stem, dot]).symbols
    assert s[-1] == fx.K + 1
    assert set(s[:-1]) == {12}


def test_pen_up_in_same_place_emits_no_jump(fx):
    s = fx.extract([line((0, 0), (100, 0)), line((100, 0), (200, 0))]).symbols
    assert not any(fx.is_jump(o) for o in s)


def test_symbols_within_alphabet_and_shape_vector(fx):
    rng = np.random.default_rng(0)
    strokes = [np.cumsum(rng.normal(size=(80, 2)), axis=0) for _ in range(2)]
    res = fx.extract(strokes)
    assert res.symbols.min() >= 0 and res.symbols.max() < fx.n_symbols == 20
    assert res.shape.shape == (7,)
    # Tall stroke vs wide stroke have opposite log(w/h) signs
    tall = compute_shape_features([np.array([[0.0, 0.0], [0.1, 1.0]])])
    wide = compute_shape_features([np.array([[0.0, 0.0], [1.0, 0.1]])])
    assert tall[0] < 0.0 < wide[0]


def test_empty_and_single_point(fx):
    assert fx.extract([]).symbols.size == 0
    assert fx.extract([[[5, 5]]]).symbols.size == 0


def test_inverse_normalization_roundtrip(fx):
    raw = line((120, 300), (220, 250))
    res = fx.extract([raw])
    back = res.to_input_coords(res.strokes[0])
    np.testing.assert_allclose(back[0], raw[0], atol=1e-9)


def test_realtime_prefix_matches_full_stroke_prefix(fx):
    S = fx.cfg.expected_letter_size
    strokes = [line((0, 0), (S, 0), n=100), line((S, 0), (S, S), n=100)]
    full = fx.extract(strokes, realtime=False).symbols
    for frac in get_config().prefix_fractions:
        pre = fx.extract_prefix(strokes, frac).symbols
        assert 0 < len(pre) <= len(full)
        np.testing.assert_array_equal(pre, full[: len(pre)])


def sphere_arc(theta0, phi0, theta1, phi1, n=60, r=1.0):
    thetas = np.linspace(theta0, theta1, n)
    phis = np.linspace(phi0, phi1, n)
    st = np.sin(thetas)
    return r * np.column_stack([st * np.cos(phis), st * np.sin(phis), np.cos(thetas)])


@pytest.mark.parametrize("K", [8, 16])
def test_3d_cardinal_directions_on_equator(K):
    fx3 = FeatureExtractor3D(K=K)
    eq = np.pi / 2.0
    assert set(fx3.extract([sphere_arc(eq, -0.3, eq, 0.3)]).symbols) == {0}
    assert set(fx3.extract([sphere_arc(eq + 0.3, 0.0, eq - 0.3, 0.0)]).symbols) == {K // 4}
    assert set(fx3.extract([sphere_arc(eq, 0.3, eq, -0.3)]).symbols) == {K // 2}
    assert set(fx3.extract([sphere_arc(eq - 0.3, 0.0, eq + 0.3, 0.0)]).symbols) == {3 * K // 4}


def test_3d_camera_frame_rotation_invariance():
    fx3 = FeatureExtractor3D(K=16)
    stroke_front = sphere_arc(np.pi / 2 + 0.25, -0.25, np.pi / 2 - 0.25, 0.25, r=2.5)
    sym_front = fx3.extract([stroke_front], camera_rotation=(0.0, 0.0)).symbols

    cam_orient = (135.0, 35.0)
    R_cam = fx3.build_camera_matrix(cam_orient)
    stroke_rotated_world = stroke_front @ R_cam
    res_rot = fx3.extract([stroke_rotated_world], camera_rotation=cam_orient)

    np.testing.assert_array_equal(sym_front, res_rot.symbols)
    assert res_rot.shape.shape == (7,)
    world_back = res_rot.to_world_coords(res_rot.strokes[0])
    np.testing.assert_allclose(world_back[0], stroke_rotated_world[0], atol=1e-6)


def test_3d_phi_unwrap_and_geodesic_prefix():
    fx3 = FeatureExtractor3D(K=16)
    arc = sphere_arc(np.pi / 2, np.deg2rad(170.0), np.pi / 2, np.deg2rad(190.0), n=50)
    sph = fx3.cartesian_to_spherical(arc)
    assert np.all(np.diff(sph[:, 1]) > 0)
    assert set(fx3.extract([arc]).symbols) == {0}

    span = fx3.cfg.expected_angle_3d
    eq = np.pi / 2.0
    strokes = [
        sphere_arc(eq, -span / 2, eq, span / 2, n=80),
        sphere_arc(eq, span / 2, eq + span, span / 2, n=80),
    ]
    full = fx3.extract(strokes, realtime=False).symbols
    for frac in get_config().prefix_fractions:
        pre = fx3.extract_prefix(strokes, frac).symbols
        assert 0 < len(pre) <= len(full)
        np.testing.assert_array_equal(pre, full[: len(pre)])
