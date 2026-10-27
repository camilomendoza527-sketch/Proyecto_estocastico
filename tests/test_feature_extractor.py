"""Pruebas del FeatureExtractor 2D y del cargador de configuración."""

import numpy as np
import pytest

from core.config import get_config
from core.FeatureExtractor import FeatureExtractor2D, FeatureExtractor3D


def line(p0, p1, n=50):
    """Segmento recto de p0 a p1 con n puntos, en coordenadas de pantalla."""
    return np.linspace(p0, p1, n)


@pytest.fixture
def fx():
    return FeatureExtractor2D()  # K=16 por config


# --- Configuración ------------------------------------------------------------
def test_alphabet_is_exactly_27_spanish_letters():
    letters = get_config().letters
    assert len(letters) == 27
    assert "ñ" in letters
    assert list(letters) == list("abcdefghijklmn") + ["ñ"] + list("opqrstuvwxyz")
    assert not {"ch", "ll", "á", "é", "í", "ó", "ú"} & set(letters)


def test_states_defined_for_every_letter():
    cfg = get_config()
    assert set(cfg.states) == set(cfg.letters)
    assert cfg.file_name("ñ") == "enie"


# --- Cuantización -------------------------------------------------------------
@pytest.mark.parametrize("K", [8, 16])
def test_cardinal_directions(K):
    fx = FeatureExtractor2D(K=K)
    # Pantalla: y hacia abajo, así que "arriba" es y decreciente.
    assert set(fx.extract([line((0, 0), (100, 0))]).symbols) == {0}           # →
    assert set(fx.extract([line((0, 100), (0, 0))]).symbols) == {K // 4}      # ↑
    assert set(fx.extract([line((100, 0), (0, 0))]).symbols) == {K // 2}      # ←
    assert set(fx.extract([line((0, 0), (0, 100))]).symbols) == {3 * K // 4}  # ↓


def test_sector_boundaries_are_centered(fx):
    w = 2 * np.pi / 16
    assert fx.quantize_direction(w / 2 - 1e-6) == 0
    assert fx.quantize_direction(w / 2 + 1e-6) == 1
    assert fx.quantize_direction(-w / 2 + 1e-6) == 0
    assert fx.quantize_direction(np.pi) == 8


# --- Invariancias -------------------------------------------------------------
def test_scale_invariance(fx):
    small = [line((0, 0), (30, 0)), line((30, 0), (30, 40))]
    big = [line((0, 0), (300, 0)), line((300, 0), (300, 400))]
    np.testing.assert_array_equal(fx.extract(small).symbols, fx.extract(big).symbols)


def test_speed_invariance(fx):
    slow = [line((0, 0), (100, 100), n=400)]
    fast = [line((0, 0), (100, 100), n=12)]
    a, b = fx.extract(slow).symbols, fx.extract(fast).symbols
    assert len(a) == len(b)
    np.testing.assert_array_equal(a, b)


def test_resample_constant_step(fx):
    pts = np.array([[0, 0], [0.3, 0], [0.3, 0.7]])
    q = fx.resample(pts)
    steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    # Los pasos sobre el mismo segmento miden exactamente D; en la esquina, a lo sumo D.
    assert np.all(steps <= fx.cfg.resample_step + 1e-9)


# --- Multitrazo ---------------------------------------------------------------
def test_letter_i_stem_then_dot(fx):
    stem = line((50, 40), (50, 140))   # trazo hacia abajo
    dot = np.array([[50, 10]])         # clic arriba del palo
    s = fx.extract([stem, dot]).symbols
    assert s[-1] == fx.K + 1           # salto ↑
    assert set(s[:-1]) == {12}         # dirección ↓ con K=16


def test_pen_up_in_same_place_emits_no_jump(fx):
    s = fx.extract([line((0, 0), (100, 0)), line((100, 0), (200, 0))]).symbols
    assert not any(fx.is_jump(o) for o in s)


def test_symbols_within_alphabet(fx):
    rng = np.random.default_rng(0)
    strokes = [np.cumsum(rng.normal(size=(80, 2)), axis=0) for _ in range(3)]
    s = fx.extract(strokes).symbols
    assert s.min() >= 0 and s.max() < fx.n_symbols == 20


def test_empty_and_single_point(fx):
    assert fx.extract([]).symbols.size == 0
    assert fx.extract([[[5, 5]]]).symbols.size == 0


def test_inverse_normalization_roundtrip(fx):
    raw = line((120, 300), (220, 250))
    res = fx.extract([raw])
    back = res.to_input_coords(res.strokes[0])
    np.testing.assert_allclose(back[0], raw[0], atol=1e-9)


# --- Tiempo real y prefijos ---------------------------------------------------
def test_realtime_prefix_matches_full_stroke_prefix(fx):
    # Letra de tamaño = expected_letter_size (cabe en el recuadro guía).
    S = fx.cfg.expected_letter_size
    strokes = [line((0, 0), (S, 0), n=100), line((S, 0), (S, S), n=100)]
    full = fx.extract(strokes, realtime=False).symbols
    for frac in get_config().prefix_fractions:
        pre = fx.extract_prefix(strokes, frac).symbols
        assert 0 < len(pre) <= len(full)
        np.testing.assert_array_equal(pre, full[: len(pre)])


def test_realtime_uses_bbox_when_larger_than_expected(fx):
    S = fx.cfg.expected_letter_size
    big = [line((0, 0), (2 * S, 0))]
    assert fx.extract(big, realtime=True).scale == pytest.approx(2 * S)
    small = [line((0, 0), (0.2 * S, 0))]
    assert fx.extract(small, realtime=True).scale == pytest.approx(S)


def test_config_conventions_and_prefixes():
    cfg = get_config()
    assert len(cfg.stroke_convention) == 3
    assert cfg.prefix_fractions == (0.3, 0.5, 0.7, 1.0)


# --- Extractor 3D sobre la esfera ---------------------------------------------
def sphere_arc(theta0, phi0, theta1, phi1, n=60, r=1.0):
    """Arco parametrizado en (θ, φ) sobre una esfera de radio r."""
    thetas = np.linspace(theta0, theta1, n)
    phis = np.linspace(phi0, phi1, n)
    st = np.sin(thetas)
    return r * np.column_stack([st * np.cos(phis), st * np.sin(phis), np.cos(thetas)])


@pytest.mark.parametrize("K", [8, 16])
def test_3d_cardinal_directions_on_equator(K):
    fx3 = FeatureExtractor3D(K=K)
    eq = np.pi / 2.0
    # Este (→): φ crece en el ecuador
    assert set(fx3.extract([sphere_arc(eq, -0.3, eq, 0.3)]).symbols) == {0}
    # Norte (↑): θ decrece hacia el polo norte (+Z)
    assert set(fx3.extract([sphere_arc(eq + 0.3, 0.0, eq - 0.3, 0.0)]).symbols) == {K // 4}
    # Oeste (←): φ decrece en el ecuador
    assert set(fx3.extract([sphere_arc(eq, 0.3, eq, -0.3)]).symbols) == {K // 2}
    # Sur (↓): θ crece hacia el polo sur (-Z)
    assert set(fx3.extract([sphere_arc(eq - 0.3, 0.0, eq + 0.3, 0.0)]).symbols) == {3 * K // 4}


def test_3d_camera_frame_rotation_invariance():
    fx3 = FeatureExtractor3D(K=16)
    # Trazo dibujado frente a la cámara en (azimut=0, elev=0)
    stroke_front = sphere_arc(np.pi / 2 + 0.25, -0.25, np.pi / 2 - 0.25, 0.25, r=2.5)
    sym_front = fx3.extract([stroke_front], camera_rotation=(0.0, 0.0)).symbols

    # Rotamos la cámara (azimut=135°, elev=35°) y rotamos el trazo solidariamente en el mundo
    cam_orient = (135.0, 35.0)
    R_cam = fx3.build_camera_matrix(cam_orient)
    stroke_rotated_world = stroke_front @ R_cam
    res_rot = fx3.extract([stroke_rotated_world], camera_rotation=cam_orient)

    np.testing.assert_array_equal(sym_front, res_rot.symbols)
    # Inversa al marco del mundo recupera el radio y posición inicial
    world_back = res_rot.to_world_coords(res_rot.strokes[0])
    np.testing.assert_allclose(world_back[0], stroke_rotated_world[0], atol=1e-6)


def test_3d_phi_unwrap_across_antimeridian():
    fx3 = FeatureExtractor3D(K=16)
    # Arco en el marco de cámara que cruza φ = π -> -π (de +170° a +190°)
    arc = sphere_arc(np.pi / 2, np.deg2rad(170.0), np.pi / 2, np.deg2rad(190.0), n=50)
    sph = fx3.cartesian_to_spherical(arc)
    # φ desenvuelto debe ser estrictamente creciente (sin salto de -2π)
    assert np.all(np.diff(sph[:, 1]) > 0)
    assert set(fx3.extract([arc]).symbols) == {0}


def test_3d_geodesic_resample_and_prefix():
    fx3 = FeatureExtractor3D(K=16)
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


