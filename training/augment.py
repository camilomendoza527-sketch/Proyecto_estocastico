"""Plantillas vectoriales 2D y aumento de datos para las 27 letras del español.

Proporciona trazos canónicos en coordenadas de pantalla (y hacia abajo) dentro
de un recuadro de lado expected_letter_size, respetando estrictamente la
convención de trazo aprobada:
1. Primero el cuerpo de la letra; después el punto, la virgulilla o la barra.
2. La barra de la 't' se traza siempre de izquierda a derecha.
3. Los dos trazos de la 'x' se trazan siempre de izquierda a derecha.

También incluye transformaciones afines (rotación suave, escala anisotrópica,
inclinación/shear y ruido correlacionado) para aumentar muestras reales o
generar un dataset base reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from core.config import PROJECT_ROOT, get_config


def _poly(points: Sequence[tuple[float, float]], n_per_seg: int = 16) -> np.ndarray:
    """Interpola linealmente una polilínea de puntos de control en [0, 1]²."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 1:
        return pts.copy()
    segments = [
        np.linspace(pts[i], pts[i + 1], n_per_seg, endpoint=(i == len(pts) - 2))
        for i in range(len(pts) - 1)
    ]
    return np.vstack(segments)


def _arc(
    cx: float, cy: float, rx: float, ry: float, deg_start: float, deg_end: float, n: int = 36
) -> np.ndarray:
    """Arco elíptico en coordenadas normalizadas (y hacia abajo)."""
    ang = np.deg2rad(np.linspace(deg_start, deg_end, n))
    return np.column_stack([cx + rx * np.cos(ang), cy - ry * np.sin(ang)])


def canonical_unit_strokes(letter: str) -> list[np.ndarray]:
    """Devuelve la lista de sub-trazos en [0, 1]² (y hacia abajo) de una letra."""
    if letter == "a":
        loop = _arc(0.45, 0.56, 0.25, 0.24, 35, 385, n=40)
        stem = _poly([(0.70, 0.34), (0.70, 0.80), (0.76, 0.82)])
        return [np.vstack([loop, stem])]
    if letter == "b":
        stem = _poly([(0.28, 0.12), (0.28, 0.82)])
        belly = _arc(0.48, 0.58, 0.22, 0.22, 130, -140, n=36)
        return [np.vstack([stem, belly])]
    if letter == "c":
        return [_arc(0.55, 0.55, 0.28, 0.26, 45, 315, n=42)]
    if letter == "d":
        loop = _arc(0.46, 0.60, 0.22, 0.20, 50, 390, n=34)
        up_down = _poly([(0.68, 0.48), (0.68, 0.12), (0.68, 0.84)])
        return [np.vstack([loop, up_down])]
    if letter == "e":
        bar = _poly([(0.26, 0.55), (0.76, 0.55)])
        curve = _arc(0.51, 0.55, 0.25, 0.25, 0, 315, n=40)
        return [np.vstack([bar, curve[1:]])]
    if letter == "f":
        # Cuerpo primero (gancho superior + tallo), después barra izq -> der
        hook = _arc(0.56, 0.26, 0.14, 0.14, 20, 180, n=20)
        stem = _poly([(0.42, 0.26), (0.42, 0.86)])
        bar = _poly([(0.26, 0.48), (0.62, 0.48)])
        return [np.vstack([hook, stem[1:]]), bar]
    if letter == "g":
        loop = _arc(0.48, 0.46, 0.22, 0.20, 45, 395, n=36)
        desc = _poly([(0.70, 0.26), (0.70, 0.78)])
        hook = _arc(0.50, 0.78, 0.20, 0.14, 0, -150, n=22)
        return [np.vstack([loop, desc, hook[1:]])]
    if letter == "h":
        stem = _poly([(0.28, 0.12), (0.28, 0.82), (0.28, 0.48)])
        arch = _arc(0.50, 0.50, 0.22, 0.16, 180, 0, n=22)
        leg = _poly([(0.72, 0.50), (0.72, 0.82)])
        return [np.vstack([stem, arch[1:], leg[1:]])]
    if letter == "i":
        # Cuerpo primero, punto arriba después
        stem = _poly([(0.50, 0.34), (0.50, 0.82)])
        dot = _poly([(0.50, 0.16), (0.51, 0.16)], n_per_seg=2)
        return [stem, dot]
    if letter == "j":
        # Cuerpo con gancho inferior primero, punto arriba después
        stem = _poly([(0.56, 0.32), (0.56, 0.76)])
        hook = _arc(0.40, 0.76, 0.16, 0.14, 0, -145, n=20)
        dot = _poly([(0.56, 0.15), (0.57, 0.15)], n_per_seg=2)
        return [np.vstack([stem, hook[1:]]), dot]
    if letter == "k":
        stem = _poly([(0.30, 0.12), (0.30, 0.84)])
        arms = _poly([(0.68, 0.34), (0.32, 0.58), (0.70, 0.84)])
        return [stem, arms]
    if letter == "l":
        return [_poly([(0.50, 0.12), (0.50, 0.84)], n_per_seg=36)]
    if letter == "m":
        s0 = _poly([(0.18, 0.36), (0.18, 0.82), (0.18, 0.48)])
        a1 = _arc(0.34, 0.48, 0.16, 0.14, 180, 0, n=18)
        s1 = _poly([(0.50, 0.48), (0.50, 0.82), (0.50, 0.48)])
        a2 = _arc(0.66, 0.48, 0.16, 0.14, 180, 0, n=18)
        s2 = _poly([(0.82, 0.48), (0.82, 0.82)])
        return [np.vstack([s0, a1[1:], s1[1:], a2[1:], s2[1:]])]
    if letter == "n":
        s0 = _poly([(0.28, 0.34), (0.28, 0.82), (0.28, 0.48)])
        arch = _arc(0.50, 0.48, 0.22, 0.16, 180, 0, n=22)
        leg = _poly([(0.72, 0.48), (0.72, 0.82)])
        return [np.vstack([s0, arch[1:], leg[1:]])]
    if letter == "ñ":
        # Cuerpo de 'n' primero, virgulilla arriba de izq a der después
        body = canonical_unit_strokes("n")[0]
        tilde = _poly([(0.30, 0.20), (0.42, 0.14), (0.58, 0.22), (0.70, 0.15)], n_per_seg=10)
        return [body, tilde]
    if letter == "o":
        return [_arc(0.50, 0.55, 0.26, 0.26, 90, 450, n=44)]
    if letter == "p":
        stem = _poly([(0.28, 0.32), (0.28, 0.90), (0.28, 0.38)])
        belly = _arc(0.48, 0.50, 0.22, 0.20, 140, -140, n=34)
        return [np.vstack([stem, belly])]
    if letter == "q":
        loop = _arc(0.46, 0.36, 0.19, 0.16, 45, 395, n=30)
        stem = _poly([(0.65, 0.22), (0.65, 0.92)])
        bar = _poly([(0.48, 0.68), (0.82, 0.68)])
        return [np.vstack([loop, stem]), bar]
    if letter == "r":
        stem = _poly([(0.34, 0.34), (0.34, 0.82), (0.34, 0.48)])
        arch = _arc(0.54, 0.48, 0.20, 0.15, 180, 30, n=22)
        return [np.vstack([stem, arch[1:]])]
    if letter == "s":
        top = _arc(0.50, 0.42, 0.20, 0.14, 30, 255, n=24)
        bot = _arc(0.50, 0.68, 0.20, 0.14, 75, -155, n=24)
        return [np.vstack([top, bot[1:]])]
    if letter == "t":
        # Cuerpo primero, barra siempre de izquierda a derecha después
        stem = _poly([(0.46, 0.14), (0.46, 0.76)])
        foot = _arc(0.56, 0.76, 0.10, 0.08, 180, 330, n=14)
        bar = _poly([(0.26, 0.36), (0.68, 0.36)])
        return [np.vstack([stem, foot[1:]]), bar]
    if letter == "u":
        left = _poly([(0.28, 0.34), (0.28, 0.66)])
        cup = _arc(0.50, 0.66, 0.22, 0.16, 180, 360, n=22)
        right = _poly([(0.72, 0.66), (0.72, 0.34), (0.72, 0.82)])
        return [np.vstack([left, cup[1:], right[1:]])]
    if letter == "v":
        return [_poly([(0.24, 0.34), (0.50, 0.82), (0.76, 0.34)], n_per_seg=22)]
    if letter == "w":
        return [
            _poly(
                [(0.16, 0.34), (0.33, 0.82), (0.50, 0.46), (0.67, 0.82), (0.84, 0.34)],
                n_per_seg=16,
            )
        ]
    if letter == "x":
        # Ambos trazos siempre de izquierda a derecha
        d1 = _poly([(0.24, 0.32), (0.76, 0.82)], n_per_seg=24)
        d2 = _poly([(0.24, 0.82), (0.76, 0.32)], n_per_seg=24)
        return [d1, d2]
    if letter == "y":
        cup = _poly([(0.26, 0.34), (0.48, 0.64)])
        desc = _poly([(0.72, 0.34), (0.32, 0.90)])
        return [cup, desc]
    if letter == "z":
        return [_poly([(0.24, 0.34), (0.76, 0.34), (0.24, 0.82), (0.76, 0.82)], n_per_seg=18)]

    raise ValueError(f"No existe plantilla para la letra {letter!r}")


def augment_strokes(
    strokes: Sequence[np.ndarray],
    rng: np.random.Generator,
    rot_deg: float = 7.0,
    scale_jitter: float = 0.10,
    shear_jitter: float = 0.08,
    noise_std: float = 0.008,
) -> list[np.ndarray]:
    """Aplica perturbación afín suave + ruido gaussiano a una muestra de trazos.

    Ecuación afín alrededor del centroide c:
        p' = c + R(θ) · [ 1   sh_x ] · [ s_x   0  ] · (p - c) + ε
                        [ sh_y  1  ]   [  0   s_y ]
    """
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
            jitter = rng.normal(0.0, noise_std * span, size=stroke.shape)
            transformed = transformed + jitter
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
    """Mapea una muestra de sub-trazos 2D a la superficie de una esfera 3D.

    Ecuaciones del mapeo carta tangente → parche esférico centrado en (θ_0, φ_0):
        1. Normalización centrada de los puntos 2D:
           (si y_axis_down)  y ← -y
           c = (p_min + p_max) / 2,   s = max(x_max - x_min, y_max - y_min, 1e-6)
           (u, v) = (p - c) / s ∈ [-0.5, 0.5]²
        2. Coordenadas esféricas con apertura angular α = angular_span:
           θ = θ_0 - α · v
           φ = φ_0 + (α · u) / sin(θ_0)
        3. Puntos 3D en el marco de la cámara y rotación al marco del mundo:
           p_cam(θ, φ) = R · (sin θ cos φ, sin θ sin φ, cos θ)
           p_world = p_cam · R_cam
    """
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
    p_min = all_pts.min(axis=0)
    p_max = all_pts.max(axis=0)
    center = 0.5 * (p_min + p_max)
    scale = float(max(p_max[0] - p_min[0], p_max[1] - p_min[1], 1e-6))

    sin_t0 = float(np.sin(np.clip(center_theta, 0.15, np.pi - 0.15)))
    R_cam = FeatureExtractor3D.build_camera_matrix(camera_rotation)

    strokes_3d: list[np.ndarray] = []
    for s in flipped:
        uv = (s - center) / scale
        u, v = uv[:, 0], uv[:, 1]
        theta = np.clip(center_theta - alpha * v, 0.05, np.pi - 0.05)
        phi = center_phi + (alpha * u) / sin_t0
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
    """Aplica perturbación afín en coordenadas esféricas tangentes (θ, φ) y reproyecta a la esfera.

    Ecuaciones en el plano tangente esférico local ξ = (φ - φ_c) sin θ_c, η = -(θ - θ_c):
        [ξ', η']^T = R(α) · Sh · S · [ξ, η]^T + [δφ · sin θ_c, -δθ]^T + ε
        θ' = θ_c - η',   φ' = φ_c + ξ' / sin θ_c
        p'_world = R · p̂(θ', φ') · R_cam
    """
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
    span = max(
        float(np.ptp(all_sph[:, 1]) * sin_tc),
        float(np.ptp(all_sph[:, 0])),
        0.1,
    )

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


def save_raw_sample(
    letter: str,
    strokes: Sequence[np.ndarray],
    mode: str = "2d",
    base_dir: Path | None = None,
    prefix: str = "sample",
) -> Path:
    """Guarda una muestra cruda en data/raw/<mode>/<alias>/<prefix>_NNN.json."""
    cfg = get_config()
    alias = cfg.file_name(letter)
    root = base_dir if base_dir is not None else PROJECT_ROOT / "data" / "raw"
    folder = root / mode / alias
    folder.mkdir(parents=True, exist_ok=True)

    existing = sorted(folder.glob(f"{prefix}_*.json"))
    idx = len(existing) + 1
    while (folder / f"{prefix}_{idx:03d}.json").exists():
        idx += 1

    decimals = 5 if mode == "3d" else 3
    target = folder / f"{prefix}_{idx:03d}.json"
    payload = {
        "letter": letter,
        "mode": mode,
        "strokes": [np.asarray(s, dtype=float).round(decimals).tolist() for s in strokes],
    }
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return target


def load_raw_samples(
    letter: str, mode: str = "2d", base_dir: Path | None = None
) -> list[list[np.ndarray]]:
    """Carga todas las muestras crudas de una letra desde data/raw/<mode>/<alias>/."""
    cfg = get_config()
    alias = cfg.file_name(letter)
    root = base_dir if base_dir is not None else PROJECT_ROOT / "data" / "raw"
    folder = root / mode / alias
    if not folder.exists():
        return []
    dim = 3 if mode == "3d" else 2
    samples: list[list[np.ndarray]] = []
    for fpath in sorted(folder.glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        strokes = [np.asarray(s, dtype=float).reshape(-1, dim) for s in data["strokes"]]
        strokes = [s for s in strokes if len(s) > 0]
        if strokes:
            samples.append(strokes)
    return samples


def canonical_stroke_variants(letter: str) -> list[list[np.ndarray]]:
    """Devuelve las variantes naturales de trazo de una letra para enriquecer el entrenamiento."""
    base = canonical_unit_strokes(letter)
    if letter == "a":
        # Variante 1: 1 trazo con colita manuscrita hacia la derecha (base)
        # Variante 2: 1 trazo con palo derecho totalmente recto vertical
        v2 = [
            np.vstack(
                [
                    _arc(0.45, 0.55, 0.25, 0.25, 40, 395, n=40),
                    _poly([(0.70, 0.30), (0.70, 0.82)]),
                ]
            )
        ]
        # Variante 3: 1 trazo cerrando a mitad del círculo y bajando con colita suave
        v3 = [
            np.vstack(
                [
                    _arc(0.45, 0.55, 0.25, 0.24, 35, 360, n=36),
                    _poly([(0.70, 0.55), (0.70, 0.82), (0.76, 0.79)]),
                ]
            )
        ]
        # Variante 4: 2 sub-trazos (círculo primero + palo vertical derecho después)
        v4 = [
            _arc(0.45, 0.55, 0.25, 0.24, 35, 390, n=38),
            _poly([(0.70, 0.32), (0.70, 0.82)]),
        ]
        return [base, v2, v3, v4]
    if letter == "d":
        # Variante 1: 1 trazo subiendo hasta arriba y bajando (base)
        # Variante 2: 2 sub-trazos (círculo + tallo alto desde arriba)
        v2 = [
            _arc(0.46, 0.60, 0.22, 0.20, 45, 390, n=34),
            _poly([(0.68, 0.12), (0.68, 0.84)]),
        ]
        return [base, v2]
    if letter == "q":
        # Variante 1: 2 sub-trazos (óvalo + tallo largo + barra horizontal en español)
        # Variante 2: 1 trazo con tallo largo descendente y quiebre final
        loop = _arc(0.46, 0.36, 0.19, 0.16, 45, 395, n=30)
        v2 = [np.vstack([loop, _poly([(0.65, 0.22), (0.65, 0.92), (0.80, 0.78)])])]
        return [base, v2]
    return [base]


def generate_seed_dataset_2d(
    samples_per_letter: int = 24,
    seed: int = 42,
    base_dir: Path | None = None,
    only_if_empty: bool = True,
) -> dict[str, int]:
    """Genera muestras sintéticas perturbadas en data/raw/2d/<letra>/ para las 27 letras."""
    cfg = get_config()
    box = cfg.features.expected_letter_size
    offset = np.array([50.0, 50.0])
    rng = np.random.default_rng(seed)

    counts: dict[str, int] = {}
    for letter in cfg.letters:
        existing = load_raw_samples(letter, mode="2d", base_dir=base_dir)
        if only_if_empty and len(existing) >= samples_per_letter:
            counts[letter] = len(existing)
            continue

        variants = [
            [s * box + offset for s in v] for v in canonical_stroke_variants(letter)
        ]
        to_create = max(0, samples_per_letter - len(existing))
        for i in range(to_create):
            base_strokes = variants[i % len(variants)]
            sample = base_strokes if i < len(variants) else augment_strokes(base_strokes, rng)
            save_raw_sample(
                letter, sample, mode="2d", base_dir=base_dir, prefix="synth"
            )
        counts[letter] = len(existing) + to_create
    return counts


def generate_seed_dataset_3d(
    samples_per_letter: int = 24,
    seed: int = 142,
    base_dir: Path | None = None,
    only_if_empty: bool = True,
) -> dict[str, int]:
    """Genera muestras 3D sobre la esfera en data/raw/3d/<letra>/ mediante el mapeo 2D → esfera."""
    cfg = get_config()
    box = cfg.features.expected_letter_size
    base_span = cfg.features.expected_angle_3d
    radius = cfg.features.sphere_radius
    rng = np.random.default_rng(seed)

    counts: dict[str, int] = {}
    for letter in cfg.letters:
        existing = load_raw_samples(letter, mode="3d", base_dir=base_dir)
        if only_if_empty and len(existing) >= samples_per_letter:
            counts[letter] = len(existing)
            continue

        samples_2d = load_raw_samples(letter, mode="2d", base_dir=base_dir)
        variants_2d = [
            [s * box for s in v] for v in canonical_stroke_variants(letter)
        ]

        to_create = max(0, samples_per_letter - len(existing))
        for i in range(to_create):
            if i < len(variants_2d):
                src_2d = variants_2d[i]
                span = base_span
                theta_0 = np.pi / 2.0
                phi_0 = 0.0
            else:
                base_2d = (
                    samples_2d[i % len(samples_2d)]
                    if samples_2d
                    else variants_2d[i % len(variants_2d)]
                )
                src_2d = augment_strokes(base_2d, rng)
                span = float(base_span * rng.uniform(0.85, 1.18))
                theta_0 = float(np.pi / 2.0 + rng.uniform(-0.18, 0.18))
                phi_0 = float(rng.uniform(-0.22, 0.22))

            sample_3d = map_2d_to_sphere(
                src_2d,
                angular_span=span,
                center_theta=theta_0,
                center_phi=phi_0,
                radius=radius,
            )
            save_raw_sample(
                letter, sample_3d, mode="3d", base_dir=base_dir, prefix="synth"
            )
        counts[letter] = len(existing) + to_create
    return counts


