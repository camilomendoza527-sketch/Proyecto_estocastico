"""FeatureExtractor: geometría y discretización de trazos.

Convierte un trazo a mano alzada (uno o varios sub-trazos) en una secuencia
de observaciones discretas O = (o_1, ..., o_T), con o_t ∈ V = {0, ..., K+3}:

    0 .. K-1   → dirección de movimiento con la pluma abajo (K sectores
                 angulares; el sector 0 es el este y crecen en sentido antihorario)
    K .. K+3   → salto con la pluma levantada entre dos sub-trazos:
                 K+0 = →, K+1 = ↑, K+2 = ←, K+3 = ↓

Este módulo NO contiene lógica HMM: solo geometría (2D y 3D sobre la esfera).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from core.config import FeatureConfig, get_config

ArrayLike2D = Sequence[Sequence[float]] | np.ndarray

# Etiquetas legibles de los símbolos de salto (útiles para depuración e informes).
JUMP_LABELS = ("→", "↑", "←", "↓")


# -----------------------------------------------------------------------------
# Resultado de la extracción
# -----------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """Salida de :meth:`FeatureExtractor2D.extract`.

    Attributes:
        symbols:  secuencia de observaciones O_{1:T} (enteros en [0, K+4)).
        strokes:  sub-trazos remuestreados en coordenadas normalizadas.
        origin:   esquina mínima p_min de la caja envolvente (coordenadas con
                  y hacia arriba).
        scale:    factor de escala s usado en la normalización.
        flip_y:   True si la entrada tenía el eje y hacia abajo (pantalla).
    """

    symbols: np.ndarray
    strokes: list[np.ndarray] = field(default_factory=list)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 1.0
    flip_y: bool = False

    def to_input_coords(self, points: ArrayLike2D) -> np.ndarray:
        """Lleva puntos normalizados de vuelta a coordenadas de entrada (lienzo).

        Inversa de la normalización:  p = s · p' + p_min,  y además
        y ← -y si la entrada era de pantalla.
        """
        p = np.asarray(points, dtype=float).reshape(-1, 2) * self.scale + self.origin
        if self.flip_y:
            p = p * np.array([1.0, -1.0])
        return p


# -----------------------------------------------------------------------------
# Base común (2D y, más adelante, 3D): cuantización de ángulos
# -----------------------------------------------------------------------------
class FeatureExtractor:
    """Discretización angular común a los modos 2D y 3D.

    Args:
        features: parámetros de config/config.toml (sección [features]).
        K:        sobrescribe features.K. Lo usa training/evaluate.py para
                  comparar K=8 y K=16 sin editar el archivo de configuración.
    """

    def __init__(self, features: FeatureConfig | None = None, K: int | None = None):
        self.cfg = features if features is not None else get_config().features
        self.K = int(K) if K is not None else self.cfg.K
        if self.K not in (8, 16):
            raise ValueError("K debe ser 8 o 16.")
        self.n_jump = self.cfg.n_jump_symbols  # = 4

    # --- Alfabeto ------------------------------------------------------------
    @property
    def n_symbols(self) -> int:
        """|V| = K + 4."""
        return self.K + self.n_jump

    def is_jump(self, symbol: int) -> bool:
        """True si el símbolo es de pluma levantada (o ≥ K)."""
        return symbol >= self.K

    def symbol_label(self, symbol: int) -> str:
        """Texto legible: 'd3' (dirección 3) o 'J↑' (salto hacia arriba)."""
        if self.is_jump(symbol):
            return "J" + JUMP_LABELS[symbol - self.K]
        return f"d{symbol}"

    # --- Cuantización ---------------------------------------------------------
    def quantize_direction(self, angles: np.ndarray | float) -> np.ndarray:
        """Cuantiza ángulos de movimiento en K sectores centrados.

        Ecuación:  o = ⌊ ((θ + π/K) mod 2π) / (2π/K) ⌋ mod K
        con θ = atan2(Δy, Δx) ∈ (-π, π]. El sector k cubre
        [2πk/K - π/K, 2πk/K + π/K): el sector 0 está centrado en el este (→).
        """
        width = 2.0 * np.pi / self.K
        shifted = np.mod(np.asarray(angles, dtype=float) + width / 2.0, 2.0 * np.pi)
        return (np.floor(shifted / width).astype(int)) % self.K

    def quantize_jump(self, angle: float) -> int:
        """Cuantiza la dirección de un salto con la pluma levantada en 4 cuadrantes.

        Ecuación:  o = K + ( ⌊ ((θ + π/4) mod 2π) / (π/2) ⌋ mod 4 )
        Resultado: K+0 = → , K+1 = ↑ , K+2 = ← , K+3 = ↓.
        """
        shifted = np.mod(float(angle) + np.pi / 4.0, 2.0 * np.pi)
        return self.K + int(np.floor(shifted / (np.pi / 2.0))) % 4

    def symbol_direction(self, symbol: int) -> np.ndarray:
        """Vector unitario del centro del sector (inversa de la cuantización).

        Ecuaciones:
            dirección:  φ_o = 2π·o / K,          u = (cos φ_o, sin φ_o)
            salto:      φ_o = (π/2)·(o − K),     u = (cos φ_o, sin φ_o)
        """
        if self.is_jump(symbol):
            phi = (np.pi / 2.0) * (symbol - self.K)
        else:
            phi = 2.0 * np.pi * symbol / self.K
        return np.array([np.cos(phi), np.sin(phi)])


# -----------------------------------------------------------------------------
# Extractor 2D
# -----------------------------------------------------------------------------
class FeatureExtractor2D(FeatureExtractor):
    """Pipeline 2D: normalizar → suavizar → remuestrear → cuantizar.

    La entrada es una lista de sub-trazos (cada uno una secuencia de puntos
    (x, y)). Levantar el ratón separa sub-trazos, pero NO termina la letra:
    la captura completa se cierra solo con Reiniciar/Limpiar en la GUI.

    Nota para tiempo real: la caja envolvente crece mientras el usuario dibuja,
    así que la secuencia se recalcula completa en cada actualización.
    """

    # --- Paso 1: normalización por caja envolvente ----------------------------
    def normalize(
        self, strokes: list[np.ndarray], realtime: bool = False
    ) -> tuple[list[np.ndarray], np.ndarray, float]:
        """Normaliza el tamaño con la caja envolvente de TODOS los sub-trazos.

        Ecuaciones:
            (si la entrada es de pantalla)  y ← −y
            w = x_max − x_min,   h = y_max − y_min
            modo completo (realtime=False):  s = max(w, h, 1e-6)
            modo tiempo real (realtime=True): s = max(w, h, s_esp)
                con s_esp = expected_letter_size (lado del recuadro guía).
            p' = (p − p_min) / s
        En tiempo real un trazo parcial pequeño no se estira hasta 1, por lo
        que sus símbolos coinciden con el prefijo de la letra completa.
        """
        flip = np.array([1.0, -1.0]) if self.cfg.y_axis_down else np.ones(2)
        flipped = [s * flip for s in strokes]
        all_pts = np.vstack(flipped)
        p_min = all_pts.min(axis=0)
        w, h = all_pts.max(axis=0) - p_min
        floor = self.cfg.expected_letter_size if realtime else 1e-6
        scale = float(max(w, h, floor))
        return [(s - p_min) / scale for s in flipped], p_min, scale

    # --- Paso 2: suavizado ----------------------------------------------------
    def smooth(self, points: np.ndarray) -> np.ndarray:
        """Promedio móvil centrado, con una ventana simétrica que se achica en los bordes.

        Ecuación:  p̃_i = 1/(2r_i+1) · Σ_{j=−r_i}^{r_i} p_{i+j},
                   r_i = min(r, i, n−1−i),   r = (w−1)/2
        Los extremos (r_0 = r_{n−1} = 0) quedan fijos y un tramo recto no se
        deforma. Así el resultado no depende de la densidad de puntos.
        """
        w = self.cfg.smoothing_window
        n = len(points)
        if w <= 1 or n < 3:
            return points
        idx = np.arange(n)
        r_i = np.minimum(w // 2, np.minimum(idx, n - 1 - idx))
        # Sumas por ventana con sumas acumuladas: Σ_{a}^{b} p = C[b+1] − C[a].
        C = np.vstack([np.zeros((1, 2)), np.cumsum(points, axis=0)])
        sums = C[idx + r_i + 1] - C[idx - r_i]
        return sums / (2 * r_i + 1)[:, None]

    # --- Paso 3: remuestreo a distancia constante -----------------------------
    def resample(self, points: np.ndarray) -> np.ndarray:
        """Remuestrea un sub-trazo a pasos de longitud de arco constante D.

        Ecuaciones:
            s_0 = 0,   s_i = Σ_{k=1}^{i} ‖p_k − p_{k−1}‖        (longitud de arco)
            u_m = m · D,   m = 0, 1, ..., ⌊s_n / D⌋
            q_m = p(u_m)   por interpolación lineal de (s_i, p_i)
        Así la secuencia no depende de la velocidad del trazo: cada símbolo
        corresponde a la misma distancia recorrida.
        """
        # Quitar puntos consecutivos repetidos (s debe ser estrictamente creciente).
        if len(points) > 1:
            keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
            points = points[keep]
        if len(points) < 2:
            return points[:1].copy()

        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        D = self.cfg.resample_step
        u = np.arange(0.0, s[-1] + 1e-12, D)
        return np.column_stack([np.interp(u, s, points[:, d]) for d in range(2)])

    # --- Paso 4: cuantización -------------------------------------------------
    def directions(self, points: np.ndarray) -> np.ndarray:
        """Símbolos de dirección de un sub-trazo remuestreado.

        Ecuaciones:  Δ_m = q_{m+1} − q_m,   θ_m = atan2(Δy_m, Δx_m),
                     o_m = cuantizar(θ_m)  (ver quantize_direction)
        """
        if len(points) < 2:
            return np.empty(0, dtype=int)
        d = np.diff(points, axis=0)
        return self.quantize_direction(np.arctan2(d[:, 1], d[:, 0]))

    def jump(self, end_prev: np.ndarray, start_next: np.ndarray) -> int | None:
        """Símbolo del salto entre el final de un sub-trazo y el inicio del siguiente.

        Ecuaciones:  Δ = p_inicio(k+1) − p_fin(k),  θ = atan2(Δy, Δx)
                     o = cuantizar_salto(θ)  si ‖Δ‖ ≥ D;  si no, no se emite símbolo
                     (la pluma volvió prácticamente al mismo punto).
        """
        delta = start_next - end_prev
        if np.linalg.norm(delta) < self.cfg.resample_step:
            return None
        return self.quantize_jump(np.arctan2(delta[1], delta[0]))

    # --- Pipeline completo ----------------------------------------------------
    def extract(
        self, strokes: Iterable[ArrayLike2D], realtime: bool = False
    ) -> ExtractionResult:
        """Convierte sub-trazos crudos en la secuencia de observaciones O_{1:T}.

        O = dirs(trazo_1) ⊕ salto_{1→2} ⊕ dirs(trazo_2) ⊕ ... ⊕ dirs(trazo_m)
        (⊕ = concatenación).

        Args:
            strokes:  sub-trazos de la letra.
            realtime: True cuando se evalúa un trazo en curso en la GUI o un
                      prefijo en evaluate.py (usa expected_letter_size como
                      piso de la escala).
        """
        raw = [np.asarray(s, dtype=float).reshape(-1, 2) for s in strokes]
        raw = [s for s in raw if len(s) > 0]
        if not raw:
            return ExtractionResult(symbols=np.empty(0, dtype=int),
                                    flip_y=self.cfg.y_axis_down)

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
        )

    # --- Prefijos para tiempo real / evaluate.py ------------------------------
    @staticmethod
    def slice_prefix(
        strokes: Iterable[ArrayLike2D], fraction: float
    ) -> list[np.ndarray]:
        """Recorta los sub-trazos a una fracción f ∈ (0, 1] de la longitud total (pluma abajo).

        Ecuación:  L = Σ_k s_k,   L_obj = f · L;   se conservan los sub-trazos
        completos hasta acumular L_obj y el último se interpola en el punto exacto.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction debe estar en (0, 1].")
        raw = [np.asarray(s, dtype=float).reshape(-1, 2) for s in strokes]
        raw = [s for s in raw if len(s) > 0]
        if not raw or fraction >= 1.0:
            return raw

        lengths = [
            float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum())
            if len(s) >= 2 else 0.0
            for s in raw
        ]
        target = fraction * sum(lengths)
        if target <= 0.0:
            return [raw[0][:1]]

        out: list[np.ndarray] = []
        acc = 0.0
        for s, L in zip(raw, lengths):
            if acc + L < target - 1e-12:
                out.append(s)
                acc += L
                continue
            rem = target - acc
            if len(s) < 2 or rem <= 0.0:
                out.append(s[:1])
                break
            seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(seg)]
            k = int(np.searchsorted(cum, rem, side="right")) - 1
            k = min(max(k, 0), len(s) - 2)
            t = (rem - cum[k]) / max(seg[k], 1e-12)
            tip = s[k] + np.clip(t, 0.0, 1.0) * (s[k + 1] - s[k])
            out.append(np.vstack([s[: k + 1], tip]))
            break
        return out

    def extract_prefix(
        self, strokes: Iterable[ArrayLike2D], fraction: float
    ) -> ExtractionResult:
        """Extrae símbolos del prefijo de fracción f usando la escala de tiempo real."""
        return self.extract(self.slice_prefix(strokes, fraction), realtime=True)

    def reconstruct_projected_strokes(
        self, result: ExtractionResult, future_symbols: Sequence[int]
    ) -> list[np.ndarray]:
        """Reconstruye las polilíneas 2D proyectadas en coordenadas del lienzo desde el cursor actual.

        Ecuaciones (SPEC §3.5):
            Partiendo del último punto normalizado p̂_0 = q_{último}:
            - Si o_k < K (pluma abajo):
                p̂_{k} = p̂_{k-1} + D · u(o_k),   u(o_k) = (cos φ_{o_k}, sin φ_{o_k})
            - Si o_k ≥ K (salto con pluma levantada):
                inicia un nuevo sub-trazo en p̂_{k} = p̂_{k-1} + 2.5 · D · u(o_k)
            Conversión final a coordenadas del lienzo:
                p_lienzo = to_input_coords(p̂)
        """
        if not result.strokes or len(future_symbols) == 0:
            return []
        last_stroke = result.strokes[-1]
        if len(last_stroke) == 0:
            return []

        D = self.cfg.resample_step
        curr = last_stroke[-1].astype(float).copy()
        norm_strokes: list[list[np.ndarray]] = [[curr.copy()]]

        for sym in future_symbols:
            u = self.symbol_direction(int(sym))
            if self.is_jump(int(sym)):
                curr = curr + 2.5 * D * u
                norm_strokes.append([curr.copy()])
            else:
                curr = curr + D * u
                norm_strokes[-1].append(curr.copy())

        return [
            result.to_input_coords(np.vstack(seg))
            for seg in norm_strokes
            if len(seg) >= 2
        ]



# -----------------------------------------------------------------------------
# Resultado y Extractor 3D (sobre la esfera)
# -----------------------------------------------------------------------------
ArrayLike3D = Sequence[Sequence[float]] | np.ndarray
CameraOrientation = np.ndarray | tuple[float, float] | Sequence[float] | None


@dataclass
class ExtractionResult3D:
    """Salida de :meth:`FeatureExtractor3D.extract`.

    Attributes:
        symbols:           secuencia de observaciones O_{1:T} (enteros en [0, K+4)).
        strokes:           sub-trazos remuestreados sobre la esfera unitaria en el
                           marco centrado en la cámara (x, y, z).
        spherical_strokes: sub-trazos remuestreados en coordenadas esféricas (θ, φ)
                           con φ desenvuelto.
        camera_rotation:   matriz ortonormal R_cam ∈ SO(3) (3×3) usada para rotar
                           del mundo al marco de la cámara: p_cam = p_world · R_cam^T.
        angular_scale:     escala angular s_ang (rad) de la letra sobre la esfera.
        geodesic_step:     paso geodésico efectivo D_geo = s_ang · D (rad).
        radius:            radio medio R de los puntos de entrada.
    """

    symbols: np.ndarray
    strokes: list[np.ndarray] = field(default_factory=list)
    spherical_strokes: list[np.ndarray] = field(default_factory=list)
    camera_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    angular_scale: float = 1.0
    geodesic_step: float = 0.05
    radius: float = 1.0

    def to_world_coords(self, points_cam: ArrayLike3D) -> np.ndarray:
        """Lleva puntos del marco de la cámara (esfera unitaria) a coordenadas 3D del mundo.

        Ecuación inversa de la rotación y proyección radial:
            p_world = R · p_cam · R_cam
        """
        pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        return self.radius * (pts @ self.camera_rotation)


class FeatureExtractor3D(FeatureExtractor):
    """Pipeline 3D sobre la esfera:

    rotar al marco de la cámara → esféricas (θ, φ) con unwrap(Δφ) →
    suavizar sobre la esfera → remuestrear a distancia geodésica constante →
    cuantizar el rumbo en el plano tangente local con atan2(-dθ, dφ · sin θ).

    En el marco centrado en la cámara:
        - El centro de la vista apunta en +X = (1, 0, 0), que corresponde al
          ecuador frontal (θ = π/2, φ = 0), lejos de las singularidades de los
          polos (θ = 0, π) y del antimeridiano (φ = ±π).
        - El eje +Y apunta a la derecha (este: φ creciente → sector 0).
        - El eje +Z apunta hacia arriba (norte: θ decreciente → sector K/4).
    """

    # --- Paso 1: rotación al marco centrado en la vista de la cámara ----------
    @staticmethod
    def build_camera_matrix(camera_rotation: CameraOrientation = None) -> np.ndarray:
        """Construye la matriz ortonormal R_cam ∈ SO(3) del marco de la cámara.

        Ecuaciones para cámara definida por (azimut α, elevación β):
            e_front = (cos β cos α,  cos β sin α,  sin β)     (hacia la cámara → +X_cam)
            e_right = (-sin α,       cos α,        0)         (derecha pantalla → +Y_cam)
            e_up    = e_front × e_right                       (arriba pantalla  → +Z_cam)
                     = (-sin β cos α, -sin β sin α, cos β)
            R_cam   = [ e_front ; e_right ; e_up ] ∈ ℝ^{3×3},   p_cam = p_world · R_cam^T
        """
        if camera_rotation is None:
            return np.eye(3, dtype=float)
        arr = np.asarray(camera_rotation, dtype=float)
        if arr.shape == (3, 3):
            return arr
        if arr.shape == (2,):
            az = np.deg2rad(float(arr[0]))
            el = np.deg2rad(float(arr[1]))
        elif arr.shape == (3,):
            norm = float(np.linalg.norm(arr))
            if norm < 1e-12:
                return np.eye(3, dtype=float)
            v = arr / norm
            az = float(np.arctan2(v[1], v[0]))
            el = float(np.arcsin(np.clip(v[2], -1.0, 1.0)))
        else:
            raise ValueError(
                "camera_rotation debe ser None, una tupla (azimut_deg, elev_deg), "
                "un vector 3D de vista o una matriz 3x3."
            )

        ca, sa = np.cos(az), np.sin(az)
        ce, se = np.cos(el), np.sin(el)
        e_front = np.array([ce * ca, ce * sa, se], dtype=float)
        e_right = np.array([-sa, ca, 0.0], dtype=float)
        e_up = np.array([-se * ca, -se * sa, ce], dtype=float)
        return np.vstack([e_front, e_right, e_up])

    def rotate_to_camera_frame(
        self, strokes: list[np.ndarray], camera_rotation: CameraOrientation = None
    ) -> tuple[list[np.ndarray], np.ndarray, float]:
        """Rota los sub-trazos 3D al marco centrado en la cámara y normaliza a S².

        Ecuaciones:
            r_i = ‖p_i‖,   p̂_i = p_i / r_i ∈ S²
            p̂_cam,i = R_cam · p̂_i   (vector fila: p̂_i · R_cam^T)
        """
        R_cam = self.build_camera_matrix(camera_rotation)
        rotated: list[np.ndarray] = []
        radii: list[float] = []
        for s in strokes:
            norms = np.linalg.norm(s, axis=1, keepdims=True)
            radii.append(float(np.mean(norms)))
            unit_pts = s / np.maximum(norms, 1e-12)
            rotated.append(unit_pts @ R_cam.T)
        mean_radius = float(np.mean(radii)) if radii else self.cfg.sphere_radius
        return rotated, R_cam, mean_radius

    # --- Paso 2: coordenadas esféricas (θ, φ) y desenvolvimiento de Δφ --------
    @staticmethod
    def cartesian_to_spherical(points_unit: np.ndarray) -> np.ndarray:
        """Convierte puntos unitarios (x, y, z) ∈ S² a esféricas (θ, φ) con φ desenvuelto.

        Ecuaciones:
            θ_i = arccos( clip(z_i, -1, 1) ) ∈ [0, π]        (colatitud desde +Z)
            φ_i = atan2(y_i, x_i) ∈ (-π, π]                  (longitud desde +X hacia +Y)
            Δφ_i = ((φ_i - φ_{i-1} + π) mod 2π) - π          (desenvolvimiento en ±π)
            φ̃_i = φ_0 + Σ_{k=1}^{i} Δφ_k
        """
        z = np.clip(points_unit[:, 2], -1.0, 1.0)
        theta = np.arccos(z)
        phi = np.arctan2(points_unit[:, 1], points_unit[:, 0])
        phi_unwrapped = np.unwrap(phi)
        return np.column_stack([theta, phi_unwrapped])

    @staticmethod
    def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
        """Convierte coordenadas esféricas (θ, φ) a puntos sobre la esfera unitaria S².

        Ecuación:
            p̂(θ, φ) = (sin θ · cos φ,  sin θ · sin φ,  cos θ)
        """
        theta = spherical[:, 0]
        phi = spherical[:, 1]
        st = np.sin(theta)
        return np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

    def compute_angular_scale(
        self, spherical_strokes: list[np.ndarray], realtime: bool = False
    ) -> float:
        """Calcula la escala angular s_ang de la caja envolvente esférica de los sub-trazos.

        Ecuaciones:
            θ̄ = media(θ),   w_ang = (φ_max - φ_min) · sin θ̄,   h_ang = θ_max - θ_min
            modo completo (realtime=False):   s_ang = max(w_ang, h_ang, 1e-6)
            modo tiempo real (realtime=True): s_ang = max(w_ang, h_ang, expected_angle_3d)
        """
        all_sph = np.vstack(spherical_strokes)
        theta_min, phi_min = all_sph.min(axis=0)
        theta_max, phi_max = all_sph.max(axis=0)
        sin_mean_theta = float(np.sin(np.clip(np.mean(all_sph[:, 0]), 1e-3, np.pi - 1e-3)))
        w_ang = float(phi_max - phi_min) * sin_mean_theta
        h_ang = float(theta_max - theta_min)
        floor = self.cfg.expected_angle_3d if realtime else 1e-6
        return float(max(w_ang, h_ang, floor))

    # --- Paso 3: suavizado sobre la esfera ------------------------------------
    def smooth_spherical(self, points_unit: np.ndarray) -> np.ndarray:
        """Promedio móvil centrado en R³ seguido de reproyección radial a S².

        Ecuaciones:
            p̄_i = 1/(2r_i+1) · Σ_{j=-r_i}^{r_i} p̂_{i+j},   r_i = min(r, i, n-1-i)
            p̃_i = p̄_i / ‖p̄_i‖ ∈ S²
        Los extremos (r_0 = r_{n-1} = 0) permanecen exactos sobre S².
        """
        w = self.cfg.smoothing_window
        n = len(points_unit)
        if w <= 1 or n < 3:
            return points_unit
        idx = np.arange(n)
        r_i = np.minimum(w // 2, np.minimum(idx, n - 1 - idx))
        C = np.vstack([np.zeros((1, 3)), np.cumsum(points_unit, axis=0)])
        sums = C[idx + r_i + 1] - C[idx - r_i]
        avg = sums / (2 * r_i + 1)[:, None]
        norms = np.linalg.norm(avg, axis=1, keepdims=True)
        return avg / np.maximum(norms, 1e-12)

    # --- Paso 4: remuestreo a distancia geodésica constante (SLERP) -----------
    @staticmethod
    def geodesic_distances(points_unit: np.ndarray) -> np.ndarray:
        """Distancias geodésicas Δs_k (ángulos de círculo máximo) entre puntos consecutivos.

        Ecuación:
            Δs_k = arccos( clip(p̂_{k-1} · p̂_k, -1, 1) )
        """
        if len(points_unit) < 2:
            return np.empty(0, dtype=float)
        dots = np.sum(points_unit[:-1] * points_unit[1:], axis=1)
        return np.arccos(np.clip(dots, -1.0, 1.0))

    def resample_geodesic(
        self, points_unit: np.ndarray, geodesic_step: float
    ) -> np.ndarray:
        """Remuestrea un sub-trazo sobre S² a pasos de longitud de arco geodésico constante D_geo.

        Ecuaciones:
            s_0 = 0,   s_i = Σ_{k=1}^{i} arccos(p̂_{k-1} · p̂_k)
            u_m = m · D_geo,   m = 0, 1, ..., ⌊s_n / D_geo⌋
            Interpolación geodésica (SLERP) en el tramo [s_k, s_{k+1}] con Ω = Δs_{k+1},
            t = (u_m - s_k) / Ω:
                q̂_m = [ sin((1-t)Ω) / sin Ω ] · p̂_k + [ sin(tΩ) / sin Ω ] · p̂_{k+1}
        """
        if len(points_unit) > 1:
            seg_init = self.geodesic_distances(points_unit)
            keep = np.r_[True, seg_init > 1e-10]
            points_unit = points_unit[keep]
        if len(points_unit) < 2:
            return points_unit[:1].copy()

        seg = self.geodesic_distances(points_unit)
        s = np.r_[0.0, np.cumsum(seg)]
        if s[-1] < 1e-12 or geodesic_step <= 0:
            return points_unit[:1].copy()

        u = np.arange(0.0, s[-1] + 1e-12, geodesic_step)
        k_idx = np.searchsorted(s, u, side="right") - 1
        k_idx = np.clip(k_idx, 0, len(points_unit) - 2)

        omega = seg[k_idx]
        t = np.where(omega > 1e-12, (u - s[k_idx]) / np.maximum(omega, 1e-12), 0.0)
        t = np.clip(t, 0.0, 1.0)

        p0 = points_unit[k_idx]
        p1 = points_unit[k_idx + 1]

        sin_omega = np.sin(omega)
        use_slerp = sin_omega > 1e-8
        w0 = np.where(use_slerp, np.sin((1.0 - t) * omega) / np.maximum(sin_omega, 1e-12), 1.0 - t)
        w1 = np.where(use_slerp, np.sin(t * omega) / np.maximum(sin_omega, 1e-12), t)

        q = w0[:, None] * p0 + w1[:, None] * p1
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        return q / np.maximum(norms, 1e-12)

    # --- Paso 5: cuantización en el plano tangente local ----------------------
    def directions_spherical(self, spherical: np.ndarray) -> np.ndarray:
        """Símbolos de dirección a partir de puntos remuestreados en esféricas (θ, φ).

        Ecuación del rumbo en el plano tangente local ortonormal (e_φ, -e_θ):
            dθ_m = θ_{m+1} - θ_m,   dφ_m = φ_{m+1} - φ_m   (con φ ya desenvuelto)
            θ̄_m  = (θ_m + θ_{m+1}) / 2
            ψ_m  = atan2(-dθ_m, dφ_m · sin θ̄_m)
            o_m  = cuantizar(ψ_m) ∈ {0, ..., K-1}
        """
        if len(spherical) < 2:
            return np.empty(0, dtype=int)
        dtheta = np.diff(spherical[:, 0])
        dphi = np.diff(spherical[:, 1])
        theta_mid = 0.5 * (spherical[:-1, 0] + spherical[1:, 0])
        angles = np.arctan2(-dtheta, dphi * np.sin(theta_mid))
        return self.quantize_direction(angles)

    def jump_spherical(
        self,
        end_prev_unit: np.ndarray,
        start_next_unit: np.ndarray,
        geodesic_step: float,
    ) -> int | None:
        """Símbolo de salto con pluma levantada entre dos sub-trazos sobre S².

        Ecuaciones:
            Δs_jump = arccos( clip(p̂_fin · p̂_inicio, -1, 1) )
            Si Δs_jump < D_geo: no se emite símbolo (None).
            Si Δs_jump ≥ D_geo:
                Δθ = θ_inicio - θ_fin,   Δφ = ((φ_inicio - φ_fin + π) mod 2π) - π
                ψ_jump = atan2(-Δθ, Δφ · sin((θ_fin + θ_inicio)/2))
                o = cuantizar_salto(ψ_jump) ∈ {K, K+1, K+2, K+3}
        """
        dot = float(np.clip(np.dot(end_prev_unit, start_next_unit), -1.0, 1.0))
        if np.arccos(dot) < geodesic_step:
            return None
        sph = self.cartesian_to_spherical(np.vstack([end_prev_unit, start_next_unit]))
        dtheta = float(sph[1, 0] - sph[0, 0])
        dphi = float(sph[1, 1] - sph[0, 1])
        theta_mid = 0.5 * float(sph[0, 0] + sph[1, 0])
        psi = float(np.arctan2(-dtheta, dphi * np.sin(theta_mid)))
        return self.quantize_jump(psi)

    # --- Pipeline completo 3D -------------------------------------------------
    def extract(
        self,
        strokes: Iterable[ArrayLike3D],
        camera_rotation: CameraOrientation = None,
        realtime: bool = False,
    ) -> ExtractionResult3D:
        """Convierte sub-trazos 3D (x, y, z) sobre la esfera en observaciones O_{1:T}.

        O = dirs(trazo_1) ⊕ salto_{1→2} ⊕ dirs(trazo_2) ⊕ ... ⊕ dirs(trazo_m).
        """
        raw = [np.asarray(s, dtype=float).reshape(-1, 3) for s in strokes]
        raw = [s for s in raw if len(s) > 0]
        R_cam = self.build_camera_matrix(camera_rotation)
        if not raw:
            return ExtractionResult3D(
                symbols=np.empty(0, dtype=int),
                camera_rotation=R_cam,
                radius=self.cfg.sphere_radius,
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

        return ExtractionResult3D(
            symbols=np.asarray(symbols, dtype=int),
            strokes=resampled_unit,
            spherical_strokes=resampled_sph,
            camera_rotation=R_cam,
            angular_scale=angular_scale,
            geodesic_step=geodesic_step,
            radius=radius,
        )

    # --- Prefijos para tiempo real / evaluación 3D ----------------------------
    @classmethod
    def slice_prefix(
        cls, strokes: Iterable[ArrayLike3D], fraction: float
    ) -> list[np.ndarray]:
        """Recorta los sub-trazos 3D a una fracción f ∈ (0, 1] de la longitud geodésica total.

        Ecuación:
            L = Σ_k s_k  (con Δs = arccos(p̂_{i-1} · p̂_i)),   L_obj = f · L.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction debe estar en (0, 1].")
        raw = [np.asarray(s, dtype=float).reshape(-1, 3) for s in strokes]
        raw = [s for s in raw if len(s) > 0]
        if not raw or fraction >= 1.0:
            return raw

        unit_strokes = [
            s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12) for s in raw
        ]
        lengths = [
            float(cls.geodesic_distances(u).sum()) if len(u) >= 2 else 0.0
            for u in unit_strokes
        ]
        target = fraction * sum(lengths)
        if target <= 0.0:
            return [raw[0][:1]]

        out: list[np.ndarray] = []
        acc = 0.0
        for s, u, L in zip(raw, unit_strokes, lengths):
            if acc + L < target - 1e-12:
                out.append(s)
                acc += L
                continue
            rem = target - acc
            if len(s) < 2 or rem <= 0.0:
                out.append(s[:1])
                break
            seg = cls.geodesic_distances(u)
            cum = np.r_[0.0, np.cumsum(seg)]
            k = int(np.searchsorted(cum, rem, side="right")) - 1
            k = min(max(k, 0), len(s) - 2)
            t = (rem - cum[k]) / max(seg[k], 1e-12)
            tip = s[k] + np.clip(t, 0.0, 1.0) * (s[k + 1] - s[k])
            norm_tip = np.linalg.norm(tip)
            r_mid = 0.5 * (np.linalg.norm(s[k]) + np.linalg.norm(s[k + 1]))
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
        """Extrae símbolos del prefijo 3D de fracción f usando la escala angular de tiempo real."""
        return self.extract(
            self.slice_prefix(strokes, fraction),
            camera_rotation=camera_rotation,
            realtime=True,
        )

    # --- Proyección geodésica de símbolos futuros sobre la esfera ------------
    def step_geodesic(
        self, point_unit: np.ndarray, symbol: int, step_rad: float
    ) -> np.ndarray:
        """Avanza un paso geodésico δ = step_rad sobre S² desde p̂ en el rumbo del símbolo.

        Ecuaciones del mapa exponencial en S² con base tangente local (e_φ, -e_θ):
            e_φ  = (-sin φ,  cos φ,  0)
            -e_θ = (-cos θ cos φ,  -cos θ sin φ,  sin θ)
            t̂(ψ) = cos ψ · e_φ + sin ψ · (-e_θ)
            p̂_next = cos(δ) · p̂ + sin(δ) · t̂(ψ)
        """
        u_dir = self.symbol_direction(symbol)  # (cos ψ, sin ψ)
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
        """Reconstruye las curvas geodésicas 3D proyectadas en coordenadas del mundo.

        Ecuaciones (SPEC §3.5 sobre S²):
            Partiendo de p̂_0 = q̂_{último} ∈ S² en el marco de la cámara:
                p̂_k = cos(δ_k) · p̂_{k-1} + sin(δ_k) · t̂(ψ_{o_k})
            donde δ_k = D_geo para trazos (o_k < K) y 2.5 · D_geo para saltos (o_k ≥ K).
            Finalmente se transforma cada sub-trazo al marco del mundo:
                p_world = R · p̂ · R_cam
        """
        if not result.strokes or len(future_symbols) == 0:
            return []
        last_stroke = result.strokes[-1]
        if len(last_stroke) == 0:
            return []

        step_rad = max(float(result.geodesic_step), 0.02)
        curr = last_stroke[-1].astype(float).copy()
        cam_strokes: list[list[np.ndarray]] = [[curr.copy()]]

        for sym in future_symbols:
            s_int = int(sym)
            curr = self.step_geodesic(curr, s_int, step_rad)
            if self.is_jump(s_int):
                cam_strokes.append([curr.copy()])
            else:
                cam_strokes[-1].append(curr.copy())

        return [
            result.to_world_coords(np.vstack(seg))
            for seg in cam_strokes
            if len(seg) >= 2
        ]

    def build_guide_frame_curves(
        self,
        camera_rotation: CameraOrientation = None,
        radius: float | None = None,
        angular_span: float | None = None,
        n_pts: int = 60,
    ) -> list[np.ndarray]:
        """Construye el recuadro esférico guía y sus ejes centrales en el marco de la cámara.

        Ecuación del parche guía centrado en (θ = π/2, φ = 0) con semiancho α/2:
            θ ∈ [π/2 - α/2, π/2 + α/2],   φ ∈ [-α/2, +α/2]
            p_world(θ, φ) = (1.003 · R) · p̂(θ, φ) · R_cam
        """
        R = (float(radius) if radius is not None else self.cfg.sphere_radius) * 1.004
        alpha = float(angular_span) if angular_span is not None else self.cfg.expected_angle_3d
        half = 0.5 * alpha
        R_cam = self.build_camera_matrix(camera_rotation)

        t_min, t_max = np.pi / 2.0 - half, np.pi / 2.0 + half
        p_min, p_max = -half, half
        ts = np.linspace(t_min, t_max, n_pts)
        ps = np.linspace(p_min, p_max, n_pts)

        # 4 bordes del parche esférico + 2 ejes centrales (ecuador y meridiano frontal)
        sph_curves = [
            np.column_stack([np.full(n_pts, t_min), ps]),              # borde superior
            np.column_stack([ts, np.full(n_pts, p_max)]),              # borde derecho
            np.column_stack([np.full(n_pts, t_max), ps[::-1]]),        # borde inferior
            np.column_stack([ts[::-1], np.full(n_pts, p_min)]),        # borde izquierdo
            np.column_stack([np.full(n_pts, np.pi / 2.0), ps]),        # ecuador guía
            np.column_stack([ts, np.zeros(n_pts)]),                    # meridiano guía
        ]
        return [(R * self.spherical_to_cartesian(c)) @ R_cam for c in sph_curves]

    @staticmethod
    def intersect_ray_sphere(
        ray_origin: np.ndarray, ray_dir: np.ndarray, radius: float = 1.0
    ) -> np.ndarray | None:
        """Intersección exacta rayo-esfera centrada en el origen para trazar con el ratón.

        Ecuaciones:
            ‖o + t·d‖² = R²  ⇒  a·t² + b·t + c = 0
            a = d · d,   b = 2·(o · d),   c = o · o - R²,   Δ = b² - 4ac
            Si Δ ≥ 0: t = (-b - √Δ) / (2a),   p = R · (o + t·d) / ‖o + t·d‖
        """
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
        if norm < 1e-12:
            return None
        return hit * (radius / norm)



