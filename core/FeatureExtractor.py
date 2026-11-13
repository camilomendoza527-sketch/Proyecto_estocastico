"""FeatureExtractor: geometría y discretización de trazos en 2D y 3D (esfera).

Convierte un trazo a mano alzada (uno o varios sub-trazos) en una secuencia
de observaciones discretas O = (o_1, ..., o_T), con o_t ∈ V = {0, ..., K+3}:
    0 .. K-1   → dirección de movimiento con pluma abajo (K sectores angulares)
    K .. K+3   → salto con pluma levantada entre sub-trazos (→, ↑, ←, ↓)

Este módulo NO contiene lógica HMM ni código de renderizado OpenGL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from core.config import FeatureConfig, get_config

ArrayLike2D = Sequence[Sequence[float]] | np.ndarray
ArrayLike3D = Sequence[Sequence[float]] | np.ndarray
CameraOrientation = np.ndarray | tuple[float, float] | Sequence[float] | None

JUMP_LABELS = ("→", "↑", "←", "↓")


def compute_shape_features(strokes_2d: Sequence[np.ndarray]) -> np.ndarray:
    """Extrae el vector geométrico invariante g ∈ ℝ^7 de una lista de sub-trazos 2D (y hacia arriba).

    Componentes:
        g = [log(w/h), x'_ini, y'_ini, x'_fin, y'_fin, x̄', ȳ']
    donde (x', y') ∈ [0, 1]² son las coordenadas reescaladas a la caja envolvente propia.
    """
    valid = [np.asarray(s, dtype=float).reshape(-1, 2) for s in strokes_2d if len(s) > 0]
    if not valid:
        return np.zeros(7, dtype=np.float64)
    pts = np.vstack(valid)
    pmin = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - pmin, 1e-4)
    w, h = float(span[0]), float(span[1])
    norm = (pts - pmin) / span
    log_aspect = float(np.log(np.clip(w / h, 0.08, 12.0)))
    return np.array(
        [
            log_aspect,
            float(norm[0, 0]),
            float(norm[0, 1]),
            float(norm[-1, 0]),
            float(norm[-1, 1]),
            float(norm[:, 0].mean()),
            float(norm[:, 1].mean()),
        ],
        dtype=np.float64,
    )


@dataclass
class ExtractionResult:
    """Salida de FeatureExtractor2D.extract."""

    symbols: np.ndarray
    strokes: list[np.ndarray] = field(default_factory=list)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 1.0
    flip_y: bool = False
    shape: np.ndarray = field(default_factory=lambda: np.zeros(7))

    def to_input_coords(self, points: ArrayLike2D) -> np.ndarray:
        """Transformación inversa a coordenadas del lienzo: p = s · p' + p_min."""
        p = np.asarray(points, dtype=float).reshape(-1, 2) * self.scale + self.origin
        if self.flip_y:
            p = p * np.array([1.0, -1.0])
        return p


@dataclass
class ExtractionResult3D:
    """Salida de FeatureExtractor3D.extract."""

    symbols: np.ndarray
    strokes: list[np.ndarray] = field(default_factory=list)
    spherical_strokes: list[np.ndarray] = field(default_factory=list)
    camera_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    angular_scale: float = 1.0
    geodesic_step: float = 0.05
    radius: float = 1.0
    shape: np.ndarray = field(default_factory=lambda: np.zeros(7))

    def to_world_coords(self, points_cam: ArrayLike3D) -> np.ndarray:
        """Transformación inversa del marco de cámara al mundo: p_world = R · p_cam · R_cam."""
        pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        return self.radius * (pts @ self.camera_rotation)


class FeatureExtractor:
    """Base común 2D y 3D para cuantización angular y suavizado."""

    def __init__(self, features: FeatureConfig | None = None, K: int | None = None) -> None:
        self.cfg = features if features is not None else get_config().features
        self.K = int(K) if K is not None else self.cfg.K
        if self.K not in (8, 16):
            raise ValueError("K debe ser 8 o 16.")
        self.n_jump = self.cfg.n_jump_symbols

    @property
    def n_symbols(self) -> int:
        """|V| = K + 4."""
        return self.K + self.n_jump

    def is_jump(self, symbol: int) -> bool:
        return symbol >= self.K

    def symbol_label(self, symbol: int) -> str:
        if self.is_jump(symbol):
            return "J" + JUMP_LABELS[symbol - self.K]
        return f"d{symbol}"

    def quantize_direction(self, angles: np.ndarray | float) -> np.ndarray:
        """Ecuación: o = ⌊ ((θ + π/K) mod 2π) / (2π/K) ⌋ mod K."""
        width = 2.0 * np.pi / self.K
        shifted = np.mod(np.asarray(angles, dtype=float) + width / 2.0, 2.0 * np.pi)
        return (np.floor(shifted / width).astype(int)) % self.K

    def quantize_jump(self, angle: float) -> int:
        """Ecuación: o = K + (⌊ ((θ + π/4) mod 2π) / (π/2) ⌋ mod 4)."""
        shifted = np.mod(float(angle) + np.pi / 4.0, 2.0 * np.pi)
        return self.K + int(np.floor(shifted / (np.pi / 2.0))) % 4

    def symbol_direction(self, symbol: int) -> np.ndarray:
        """Vector unitario del centro del sector: u = (cos φ_o, sin φ_o)."""
        phi = (
            (np.pi / 2.0) * (symbol - self.K)
            if self.is_jump(symbol)
            else 2.0 * np.pi * symbol / self.K
        )
        return np.array([np.cos(phi), np.sin(phi)])

    def _moving_average(self, points: np.ndarray) -> np.ndarray:
        """Promedio móvil centrado O(n) con ventana simétrica adaptativa en los bordes."""
        w = self.cfg.smoothing_window
        n = len(points)
        if w <= 1 or n < 3:
            return points
        idx = np.arange(n)
        r_i = np.minimum(w // 2, np.minimum(idx, n - 1 - idx))
        csum = np.vstack([np.zeros((1, points.shape[1])), np.cumsum(points, axis=0)])
        return (csum[idx + r_i + 1] - csum[idx - r_i]) / (2 * r_i + 1)[:, None]


class FeatureExtractor2D(FeatureExtractor):
    """Pipeline 2D: normalizar → suavizar → remuestrear → cuantizar."""

    def normalize(
        self, strokes: list[np.ndarray], realtime: bool = False
    ) -> tuple[list[np.ndarray], np.ndarray, float]:
        """Normaliza por caja envolvente: p' = (p - p_min) / max(w, h, floor)."""
        flip = np.array([1.0, -1.0]) if self.cfg.y_axis_down else np.ones(2)
        flipped = [s * flip for s in strokes]
        all_pts = np.vstack(flipped)
        p_min = all_pts.min(axis=0)
        w, h = all_pts.max(axis=0) - p_min
        floor = self.cfg.expected_letter_size if realtime else 1e-6
        scale = float(max(w, h, floor))
        return [(s - p_min) / scale for s in flipped], p_min, scale

    def smooth(self, points: np.ndarray) -> np.ndarray:
        return self._moving_average(points)

    def resample(self, points: np.ndarray) -> np.ndarray:
        """Remuestreo por longitud de arco constante D: u_m = m · D."""
        if len(points) > 1:
            keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
            points = points[keep]
        if len(points) < 2:
            return points[:1].copy()

        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        u = np.arange(0.0, s[-1] + 1e-12, self.cfg.resample_step)
        return np.column_stack([np.interp(u, s, points[:, d]) for d in range(2)])

    def directions(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return np.empty(0, dtype=int)
        d = np.diff(points, axis=0)
        return self.quantize_direction(np.arctan2(d[:, 1], d[:, 0]))

    def jump(self, end_prev: np.ndarray, start_next: np.ndarray) -> int | None:
        delta = start_next - end_prev
        if np.linalg.norm(delta) < self.cfg.resample_step:
            return None
        return self.quantize_jump(np.arctan2(delta[1], delta[0]))

    def extract(
        self, strokes: Iterable[ArrayLike2D], realtime: bool = False
    ) -> ExtractionResult:
        """Convierte sub-trazos crudos 2D en observaciones O_{1:T} y vector de forma."""
        raw = [np.asarray(s, dtype=float).reshape(-1, 2) for s in strokes]
        raw = [s for s in raw if len(s) > 0]
        if not raw:
            return ExtractionResult(symbols=np.empty(0, dtype=int), flip_y=self.cfg.y_axis_down)

        normalized, origin, scale = self.normalize(raw, realtime=realtime)
        symbols: list[int] = []
        resampled: list[np.ndarray] = []
        prev_end: np.ndarray | None = None

        for stroke in normalized:
            stroke = self.smooth(stroke)
            if prev_end is not None:
                j = self.jump(prev_end, stroke[0])
                if j is not None:
                    symbols.append(j)
            q = self.resample(stroke)
            symbols.extend(self.directions(q).tolist())
            resampled.append(q)
            prev_end = stroke[-1]

        return ExtractionResult(
            symbols=np.asarray(symbols, dtype=int),
            strokes=resampled,
            origin=origin,
            scale=scale,
            flip_y=self.cfg.y_axis_down,
            shape=compute_shape_features(normalized),
        )

    @staticmethod
    def slice_prefix(strokes: Iterable[ArrayLike2D], fraction: float) -> list[np.ndarray]:
        """Recorta los sub-trazos 2D a una fracción f ∈ (0, 1] de la longitud total."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction debe estar en (0, 1].")
        raw = [np.asarray(s, dtype=float).reshape(-1, 2) for s in strokes]
        raw = [s for s in raw if len(s) > 0]
        if not raw or fraction >= 1.0:
            return raw

        segs_list = [np.linalg.norm(np.diff(s, axis=0), axis=1) if len(s) >= 2 else np.empty(0) for s in raw]
        lengths = [float(seg.sum()) for seg in segs_list]
        target = fraction * sum(lengths)
        if target <= 0.0:
            return [raw[0][:1]]

        out: list[np.ndarray] = []
        acc = 0.0
        for s, seg, L in zip(raw, segs_list, lengths):
            if acc + L < target - 1e-12:
                out.append(s)
                acc += L
                continue
            rem = target - acc
            if len(s) < 2 or rem <= 0.0:
                out.append(s[:1])
                break
            cum = np.r_[0.0, np.cumsum(seg)]
            k = min(max(int(np.searchsorted(cum, rem, side="right")) - 1, 0), len(s) - 2)
            t = np.clip((rem - cum[k]) / max(seg[k], 1e-12), 0.0, 1.0)
            out.append(np.vstack([s[: k + 1], s[k] + t * (s[k + 1] - s[k])]))
            break
        return out

    def extract_prefix(self, strokes: Iterable[ArrayLike2D], fraction: float) -> ExtractionResult:
        return self.extract(self.slice_prefix(strokes, fraction), realtime=True)

    def reconstruct_projected_strokes(
        self, result: ExtractionResult, future_symbols: Sequence[int]
    ) -> list[np.ndarray]:
        """Reconstruye polilíneas 2D proyectadas desde el cursor actual (SPEC §3.5)."""
        if not result.strokes or len(future_symbols) == 0 or len(result.strokes[-1]) == 0:
            return []
        D = self.cfg.resample_step
        curr = result.strokes[-1][-1].astype(float).copy()
        norm_strokes: list[list[np.ndarray]] = [[curr.copy()]]

        for sym in future_symbols:
            s_int = int(sym)
            u = self.symbol_direction(s_int)
            if self.is_jump(s_int):
                curr = curr + 2.5 * D * u
                norm_strokes.append([curr.copy()])
            else:
                curr = curr + D * u
                norm_strokes[-1].append(curr.copy())

        return [result.to_input_coords(np.vstack(seg)) for seg in norm_strokes if len(seg) >= 2]


class FeatureExtractor3D(FeatureExtractor):
    """Pipeline 3D sobre S²: rotar a cámara → (θ, φ) → suavizar → SLERP → rumbo tangente."""

    @staticmethod
    def build_camera_matrix(camera_rotation: CameraOrientation = None) -> np.ndarray:
        """Matriz ortonormal R_cam ∈ SO(3) centrada en la vista de la cámara."""
        if camera_rotation is None:
            return np.eye(3, dtype=float)
        arr = np.asarray(camera_rotation, dtype=float)
        if arr.shape == (3, 3):
            return arr
        if arr.shape == (2,):
            az, el = np.deg2rad(float(arr[0])), np.deg2rad(float(arr[1]))
        elif arr.shape == (3,):
            norm = float(np.linalg.norm(arr))
            if norm < 1e-12:
                return np.eye(3, dtype=float)
            v = arr / norm
            az, el = float(np.arctan2(v[1], v[0])), float(np.arcsin(np.clip(v[2], -1.0, 1.0)))
        else:
            raise ValueError("camera_rotation inválido.")

        ca, sa = np.cos(az), np.sin(az)
        ce, se = np.cos(el), np.sin(el)
        e_front = np.array([ce * ca, ce * sa, se], dtype=float)
        e_right = np.array([-sa, ca, 0.0], dtype=float)
        e_up = np.array([-se * ca, -se * sa, ce], dtype=float)
        return np.vstack([e_front, e_right, e_up])

    def rotate_to_camera_frame(
        self, strokes: list[np.ndarray], camera_rotation: CameraOrientation = None
    ) -> tuple[list[np.ndarray], np.ndarray, float]:
        R_cam = self.build_camera_matrix(camera_rotation)
        rotated: list[np.ndarray] = []
        radii: list[float] = []
        for s in strokes:
            norms = np.linalg.norm(s, axis=1, keepdims=True)
            radii.append(float(np.mean(norms)))
            rotated.append((s / np.maximum(norms, 1e-12)) @ R_cam.T)
        mean_radius = float(np.mean(radii)) if radii else self.cfg.sphere_radius
        return rotated, R_cam, mean_radius

    @staticmethod
    def cartesian_to_spherical(points_unit: np.ndarray) -> np.ndarray:
        """Convierte (x, y, z) ∈ S² a (θ, φ) con desenvolvimiento de φ en ±π."""
        theta = np.arccos(np.clip(points_unit[:, 2], -1.0, 1.0))
        phi = np.unwrap(np.arctan2(points_unit[:, 1], points_unit[:, 0]))
        return np.column_stack([theta, phi])

    @staticmethod
    def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
        """Convierte (θ, φ) a puntos unitarios en S²."""
        theta, phi = spherical[:, 0], spherical[:, 1]
        st = np.sin(theta)
        return np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

    def compute_angular_scale(
        self, spherical_strokes: list[np.ndarray], realtime: bool = False
    ) -> float:
        all_sph = np.vstack(spherical_strokes)
        theta_min, phi_min = all_sph.min(axis=0)
        theta_max, phi_max = all_sph.max(axis=0)
        sin_mean = float(np.sin(np.clip(np.mean(all_sph[:, 0]), 1e-3, np.pi - 1e-3)))
        w_ang = float(phi_max - phi_min) * sin_mean
        h_ang = float(theta_max - theta_min)
        floor = self.cfg.expected_angle_3d if realtime else 1e-6
        return float(max(w_ang, h_ang, floor))

    def smooth_spherical(self, points_unit: np.ndarray) -> np.ndarray:
        avg = self._moving_average(points_unit)
        return avg / np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def geodesic_distances(points_unit: np.ndarray) -> np.ndarray:
        """Δs_k = arccos(clip(p̂_{k-1} · p̂_k, -1, 1))."""
        if len(points_unit) < 2:
            return np.empty(0, dtype=float)
        dots = np.sum(points_unit[:-1] * points_unit[1:], axis=1)
        return np.arccos(np.clip(dots, -1.0, 1.0))

    def resample_geodesic(self, points_unit: np.ndarray, geodesic_step: float) -> np.ndarray:
        """Remuestreo sobre S² a paso geodésico constante mediante SLERP."""
        if len(points_unit) > 1:
            keep = np.r_[True, self.geodesic_distances(points_unit) > 1e-10]
            points_unit = points_unit[keep]
        if len(points_unit) < 2:
            return points_unit[:1].copy()

        seg = self.geodesic_distances(points_unit)
        s = np.r_[0.0, np.cumsum(seg)]
        if s[-1] < 1e-12 or geodesic_step <= 0:
            return points_unit[:1].copy()

        u = np.arange(0.0, s[-1] + 1e-12, geodesic_step)
        k_idx = np.clip(np.searchsorted(s, u, side="right") - 1, 0, len(points_unit) - 2)
        omega = seg[k_idx]
        t = np.clip(np.where(omega > 1e-12, (u - s[k_idx]) / np.maximum(omega, 1e-12), 0.0), 0.0, 1.0)

        p0, p1 = points_unit[k_idx], points_unit[k_idx + 1]
        sin_omega = np.sin(omega)
        use_slerp = sin_omega > 1e-8
        w0 = np.where(use_slerp, np.sin((1.0 - t) * omega) / np.maximum(sin_omega, 1e-12), 1.0 - t)
        w1 = np.where(use_slerp, np.sin(t * omega) / np.maximum(sin_omega, 1e-12), t)
        q = w0[:, None] * p0 + w1[:, None] * p1
        return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)

    def directions_spherical(self, spherical: np.ndarray) -> np.ndarray:
        """Rumbo tangente local: ψ_m = atan2(-dθ_m, dφ_m · sin θ̄_m)."""
        if len(spherical) < 2:
            return np.empty(0, dtype=int)
        dtheta = np.diff(spherical[:, 0])
        dphi = np.diff(spherical[:, 1])
        theta_mid = 0.5 * (spherical[:-1, 0] + spherical[1:, 0])
        return self.quantize_direction(np.arctan2(-dtheta, dphi * np.sin(theta_mid)))

    def jump_spherical(
        self, end_prev_unit: np.ndarray, start_next_unit: np.ndarray, geodesic_step: float
    ) -> int | None:
        dot = float(np.clip(np.dot(end_prev_unit, start_next_unit), -1.0, 1.0))
        if np.arccos(dot) < geodesic_step:
            return None
        sph = self.cartesian_to_spherical(np.vstack([end_prev_unit, start_next_unit]))
        dtheta = float(sph[1, 0] - sph[0, 0])
        dphi = float(sph[1, 1] - sph[0, 1])
        theta_mid = 0.5 * float(sph[0, 0] + sph[1, 0])
        return self.quantize_jump(float(np.arctan2(-dtheta, dphi * np.sin(theta_mid))))

    def extract(
        self,
        strokes: Iterable[ArrayLike3D],
        camera_rotation: CameraOrientation = None,
        realtime: bool = False,
    ) -> ExtractionResult3D:
        """Convierte sub-trazos 3D sobre la esfera en observaciones O_{1:T} y vector de forma."""
        raw = [np.asarray(s, dtype=float).reshape(-1, 3) for s in strokes]
        raw = [s for s in raw if len(s) > 0]
        R_cam = self.build_camera_matrix(camera_rotation)
        if not raw:
            return ExtractionResult3D(
                symbols=np.empty(0, dtype=int), camera_rotation=R_cam, radius=self.cfg.sphere_radius
            )

        rotated_unit, R_cam, radius = self.rotate_to_camera_frame(raw, R_cam)
        smoothed_unit = [self.smooth_spherical(s) for s in rotated_unit]
        raw_spherical = [self.cartesian_to_spherical(s) for s in smoothed_unit]

        angular_scale = self.compute_angular_scale(raw_spherical, realtime=realtime)
        geodesic_step = angular_scale * self.cfg.resample_step

        symbols: list[int] = []
        resampled_unit: list[np.ndarray] = []
        resampled_sph: list[np.ndarray] = []
        prev_end: np.ndarray | None = None

        for stroke_unit in smoothed_unit:
            if prev_end is not None:
                j = self.jump_spherical(prev_end, stroke_unit[0], geodesic_step)
                if j is not None:
                    symbols.append(j)
            q_unit = self.resample_geodesic(stroke_unit, geodesic_step)
            q_sph = self.cartesian_to_spherical(q_unit)
            symbols.extend(self.directions_spherical(q_sph).tolist())
            resampled_unit.append(q_unit)
            resampled_sph.append(q_sph)
            prev_end = stroke_unit[-1]

        all_sph = np.vstack(raw_spherical)
        sin_mean = float(np.sin(np.clip(np.mean(all_sph[:, 0]), 1e-3, np.pi - 1e-3)))
        tangent_2d = [np.column_stack([s[:, 1] * sin_mean, -s[:, 0]]) for s in raw_spherical]

        return ExtractionResult3D(
            symbols=np.asarray(symbols, dtype=int),
            strokes=resampled_unit,
            spherical_strokes=resampled_sph,
            camera_rotation=R_cam,
            angular_scale=angular_scale,
            geodesic_step=geodesic_step,
            radius=radius,
            shape=compute_shape_features(tangent_2d),
        )

    @classmethod
    def slice_prefix(cls, strokes: Iterable[ArrayLike3D], fraction: float) -> list[np.ndarray]:
        """Recorta los sub-trazos 3D a una fracción f ∈ (0, 1] de la longitud geodésica."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction debe estar en (0, 1].")
        raw = [np.asarray(s, dtype=float).reshape(-1, 3) for s in strokes]
        raw = [s for s in raw if len(s) > 0]
        if not raw or fraction >= 1.0:
            return raw

        unit_strokes = [s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12) for s in raw]
        segs = [cls.geodesic_distances(u) if len(u) >= 2 else np.empty(0) for u in unit_strokes]
        lengths = [float(seg.sum()) for seg in segs]
        target = fraction * sum(lengths)
        if target <= 0.0:
            return [raw[0][:1]]

        out: list[np.ndarray] = []
        acc = 0.0
        for s, seg, L in zip(raw, segs, lengths):
            if acc + L < target - 1e-12:
                out.append(s)
                acc += L
                continue
            rem = target - acc
            if len(s) < 2 or rem <= 0.0:
                out.append(s[:1])
                break
            cum = np.r_[0.0, np.cumsum(seg)]
            k = min(max(int(np.searchsorted(cum, rem, side="right")) - 1, 0), len(s) - 2)
            t = np.clip((rem - cum[k]) / max(seg[k], 1e-12), 0.0, 1.0)
            tip = s[k] + t * (s[k + 1] - s[k])
            norm_tip = float(np.linalg.norm(tip))
            r_mid = 0.5 * (float(np.linalg.norm(s[k])) + float(np.linalg.norm(s[k + 1])))
            if norm_tip > 1e-12:
                tip = tip * (r_mid / norm_tip)
            out.append(np.vstack([s[: k + 1], tip]))
            break
        return out

    def extract_prefix(
        self,
        strokes: Iterable[ArrayLike3D],
        fraction: float,
        camera_rotation: CameraOrientation = None,
    ) -> ExtractionResult3D:
        return self.extract(
            self.slice_prefix(strokes, fraction), camera_rotation=camera_rotation, realtime=True
        )

    def step_geodesic(self, point_unit: np.ndarray, symbol: int, step_rad: float) -> np.ndarray:
        """Mapa exponencial en S²: p̂_next = cos(δ)·p̂ + sin(δ)·t̂(ψ)."""
        u_dir = self.symbol_direction(symbol)
        sph = self.cartesian_to_spherical(point_unit.reshape(1, 3))[0]
        theta, phi = float(sph[0]), float(sph[1])
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)

        e_phi = np.array([-sp, cp, 0.0])
        neg_e_theta = np.array([-ct * cp, -ct * sp, st])
        tangent = u_dir[0] * e_phi + u_dir[1] * neg_e_theta
        t_norm = float(np.linalg.norm(tangent))
        if t_norm < 1e-12:
            return point_unit.copy()
        tangent /= t_norm

        delta = step_rad * (2.5 if self.is_jump(symbol) else 1.0)
        p_next = np.cos(delta) * point_unit + np.sin(delta) * tangent
        return p_next / max(float(np.linalg.norm(p_next)), 1e-12)

    def reconstruct_projected_strokes(
        self, result: ExtractionResult3D, future_symbols: Sequence[int]
    ) -> list[np.ndarray]:
        """Reconstruye curvas geodésicas 3D proyectadas en coordenadas del mundo (SPEC §3.5)."""
        if not result.strokes or len(future_symbols) == 0 or len(result.strokes[-1]) == 0:
            return []
        step_rad = max(float(result.geodesic_step), 0.02)
        curr = result.strokes[-1][-1].astype(float).copy()
        cam_strokes: list[list[np.ndarray]] = [[curr.copy()]]

        for sym in future_symbols:
            s_int = int(sym)
            curr = self.step_geodesic(curr, s_int, step_rad)
            if self.is_jump(s_int):
                cam_strokes.append([curr.copy()])
            else:
                cam_strokes[-1].append(curr.copy())

        return [result.to_world_coords(np.vstack(seg)) for seg in cam_strokes if len(seg) >= 2]

    @staticmethod
    def intersect_ray_sphere(
        ray_origin: np.ndarray, ray_dir: np.ndarray, radius: float = 1.0
    ) -> np.ndarray | None:
        """Intersección exacta rayo-esfera ‖o + t·d‖² = R²."""
        o = np.asarray(ray_origin, dtype=float).reshape(3)
        d = np.asarray(ray_dir, dtype=float).reshape(3)
        a = float(np.dot(d, d))
        if a < 1e-12:
            return None
        b = 2.0 * float(np.dot(o, d))
        c = float(np.dot(o, o) - radius * radius)
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sqrt_disc = float(np.sqrt(disc))
        t = (-b - sqrt_disc) / (2.0 * a)
        if t < 0.0:
            t = (-b + sqrt_disc) / (2.0 * a)
        if t < 0.0:
            return None
        hit = o + t * d
        norm = float(np.linalg.norm(hit))
        return hit * (radius / norm) if norm >= 1e-12 else None
