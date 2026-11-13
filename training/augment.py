"""Aumento de datos afín (2D y 3D) y carga del dataset consolidado (training/augment.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from core.config import PROJECT_ROOT, get_config

DATASET_PATH = PROJECT_ROOT / "data" / "glifo-dataset-2026-11-12.json"
CANONICAL_PATH = PROJECT_ROOT / "data" / "canonical_glyphs.json"

_DATASET_CACHE: dict[str, list[list[np.ndarray]]] | None = None
_CANONICAL_CACHE: dict[str, list[np.ndarray]] | None = None


def canonical_unit_strokes(letter: str) -> list[np.ndarray]:
    """Devuelve los sub-trazos canónicos en [0, 1]² de una letra desde data/canonical_glyphs.json."""
    global _CANONICAL_CACHE
    if _CANONICAL_CACHE is None:
        with open(CANONICAL_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        _CANONICAL_CACHE = {
            k: [
                np.asarray(st, dtype=float)[:, :2]
                for st in (v["strokes"] if isinstance(v, dict) else v)
                if len(st) > 0
            ]
            for k, v in raw.items()
        }
    if letter not in _CANONICAL_CACHE:
        raise ValueError(f"No existe glifo canónico para la letra {letter!r}")
    return [s.copy() for s in _CANONICAL_CACHE[letter]]


def load_dataset(
    letter: str,
    mode: str = "2d",
    max_samples: int | None = None,
    dataset_path: Path | None = None,
) -> list[list[np.ndarray]]:
    """Carga las muestras limpias de data/glifo-dataset-2026-11-12.json en 2D o proyectadas a 3D."""
    global _DATASET_CACHE
    path = dataset_path if dataset_path is not None else DATASET_PATH

    if dataset_path is None:
        if _DATASET_CACHE is None:
            with open(path, "r", encoding="utf-8") as fh:
                raw_data = json.load(fh)
            _DATASET_CACHE = {}
            for k, raw_list in raw_data.items():
                key = "ñ" if (k == "ñ" or (len(k) == 1 and ord(k) == 241)) else k
                parsed: list[list[np.ndarray]] = []
                for s in raw_list:
                    strokes = [np.asarray(st, dtype=float)[:, :2] for st in s if len(st) >= 1]
                    if strokes and sum(len(st) for st in strokes) >= 4:
                        parsed.append(strokes)
                _DATASET_CACHE[key] = parsed
        samples_2d = _DATASET_CACHE.get(letter, [])
    else:
        with open(path, "r", encoding="utf-8") as fh:
            raw_data = json.load(fh)
        raw_list = raw_data.get(letter, [])
        samples_2d = [
            [np.asarray(st, dtype=float)[:, :2] for st in s if len(st) >= 1]
            for s in raw_list
        ]
        samples_2d = [s for s in samples_2d if s and sum(len(st) for st in s) >= 4]

    if max_samples is not None:
        samples_2d = samples_2d[:max_samples]

    if mode == "3d":
        cfg = get_config().features
        return [
            map_2d_to_sphere(
                s,
                angular_span=cfg.expected_angle_3d,
                radius=cfg.sphere_radius,
            )
            for s in samples_2d
        ]
    return [[st.copy() for st in s] for s in samples_2d]


def augment_strokes(
    strokes: Sequence[np.ndarray],
    rng: np.random.Generator,
    rot_deg: float = 7.0,
    scale_jitter: float = 0.10,
    shear_jitter: float = 0.08,
    noise_std: float = 0.008,
) -> list[np.ndarray]:
    """Aplica perturbación afín suave + ruido gaussiano a una muestra de trazos 2D."""
    raw = [np.asarray(s, dtype=float) for s in strokes]
    all_pts = np.vstack(raw)
    center = all_pts.mean(axis=0)
    span = max(float(np.ptp(all_pts[:, 0])), float(np.ptp(all_pts[:, 1])), 1.0)

    theta = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    S = np.diag(1.0 + rng.uniform(-scale_jitter, scale_jitter, size=2))
    Sh = np.array([[1.0, rng.uniform(-shear_jitter, shear_jitter)], [0.0, 1.0]])
    M = R @ Sh @ S

    out: list[np.ndarray] = []
    for stroke in raw:
        transformed = (stroke - center) @ M.T + center
        if len(stroke) > 2 and noise_std > 0:
            transformed = transformed + rng.normal(0.0, noise_std * span, size=stroke.shape)
        out.append(transformed)
    return out


def map_2d_to_sphere(
    strokes_2d: Sequence[np.ndarray],
    angular_span: float | None = None,
    center_theta: float = np.pi / 2.0,
    center_phi: float = 0.0,
    radius: float | None = None,
    camera_rotation: np.ndarray | tuple[float, float] | Sequence[float] | None = None,
    y_axis_down: bool | None = None,
) -> list[np.ndarray]:
    """Mapea una muestra de sub-trazos 2D a la superficie de una esfera 3D."""
    from core.FeatureExtractor import FeatureExtractor3D

    cfg = get_config().features
    alpha = float(angular_span) if angular_span is not None else cfg.expected_angle_3d
    R = float(radius) if radius is not None else cfg.sphere_radius
    flip_y = cfg.y_axis_down if y_axis_down is None else y_axis_down

    raw = [np.asarray(s, dtype=float).reshape(-1, 2) for s in strokes_2d if len(s) > 0]
    if not raw:
        return []

    flip = np.array([1.0, -1.0]) if flip_y else np.ones(2)
    flipped = [s * flip for s in raw]
    all_pts = np.vstack(flipped)
    p_min, p_max = all_pts.min(axis=0), all_pts.max(axis=0)
    center = 0.5 * (p_min + p_max)
    scale = float(max(p_max[0] - p_min[0], p_max[1] - p_min[1], 1e-6))

    sin_t0 = float(np.sin(np.clip(center_theta, 0.15, np.pi - 0.15)))
    R_cam = FeatureExtractor3D.build_camera_matrix(camera_rotation)

    strokes_3d: list[np.ndarray] = []
    for s in flipped:
        uv = (s - center) / scale
        theta = np.clip(center_theta - alpha * uv[:, 1], 0.05, np.pi - 0.05)
        phi = center_phi + (alpha * uv[:, 0]) / sin_t0
        st = np.sin(theta)
        pts_cam = R * np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])
        strokes_3d.append(pts_cam @ R_cam)
    return strokes_3d


def augment_strokes_3d(
    strokes_3d: Sequence[np.ndarray],
    rng: np.random.Generator,
    rot_deg: float = 6.0,
    scale_jitter: float = 0.10,
    shear_jitter: float = 0.06,
    offset_rad: float = 0.06,
    noise_std: float = 0.005,
    camera_rotation: np.ndarray | tuple[float, float] | Sequence[float] | None = None,
) -> list[np.ndarray]:
    """Aplica perturbación afín en coordenadas esféricas tangentes (θ, φ) y reproyecta a S²."""
    from core.FeatureExtractor import FeatureExtractor3D

    raw = [np.asarray(s, dtype=float).reshape(-1, 3) for s in strokes_3d if len(s) > 0]
    if not raw:
        return []

    fx3d = FeatureExtractor3D()
    rot_unit, R_cam, radius = fx3d.rotate_to_camera_frame(raw, camera_rotation)
    sph_list = [fx3d.cartesian_to_spherical(u) for u in rot_unit]
    all_sph = np.vstack(sph_list)

    theta_c = float(np.mean(all_sph[:, 0]))
    phi_c = float(np.mean(all_sph[:, 1]))
    sin_tc = float(np.sin(np.clip(theta_c, 0.15, np.pi - 0.15)))

    ang = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    c, s = np.cos(ang), np.sin(ang)
    R_mat = np.array([[c, -s], [s, c]])
    S_mat = np.diag(1.0 + rng.uniform(-scale_jitter, scale_jitter, size=2))
    Sh_mat = np.array([[1.0, rng.uniform(-shear_jitter, shear_jitter)], [0.0, 1.0]])
    M = R_mat @ Sh_mat @ S_mat

    d_xi = float(rng.uniform(-offset_rad, offset_rad))
    d_eta = float(rng.uniform(-offset_rad, offset_rad))
    span = max(float(np.ptp(all_sph[:, 1]) * sin_tc), float(np.ptp(all_sph[:, 0])), 0.1)

    out: list[np.ndarray] = []
    for sph in sph_list:
        xi = (sph[:, 1] - phi_c) * sin_tc
        eta = -(sph[:, 0] - theta_c)
        local = np.column_stack([xi, eta]) @ M.T + np.array([d_xi, d_eta])
        if len(sph) > 2 and noise_std > 0:
            local = local + rng.normal(0.0, noise_std * span, size=local.shape)
        theta_new = np.clip(theta_c - local[:, 1], 0.08, np.pi - 0.08)
        phi_new = phi_c + local[:, 0] / sin_tc
        pts_unit = fx3d.spherical_to_cartesian(np.column_stack([theta_new, phi_new]))
        out.append((radius * pts_unit) @ R_cam)
    return out
