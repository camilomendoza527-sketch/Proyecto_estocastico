"""Ventana C: Reconocedor y Visualizador 3D — Lienzo Esférico (SPEC §2.3).

Sección Izquierda:
- Lienzo 3D Esférico con pyqtgraph.opengl (GLViewWidget).
- Renderizado de la esfera de soporte + grilla de meridianos/paralelos +
  parche guía ecuatorial centrado en la vista de la cámara.
- Conversión angular a (θ, φ) y proyección geodésica punteada en tiempo real
  delegadas íntegramente a FeatureExtractor3D y HMMClassifier.
- Soporte para trazado sobre la esfera (clic izquierdo + arrastrar),
  rotación (clic derecho + arrastrar o casilla Rotar), zoom (rueda) y reorientación.
- Levantar el ratón solo separa sub-trazos; únicamente "Limpiar Lienzo"
  y "Reiniciar Captura" cierran la captura.

Sección Derecha:
- RankingPanel ordenado descendentemente según P(Carácter | O_{1:t}) con iconos.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QVector4D
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.config import get_config
from core.FeatureExtractor import FeatureExtractor3D
from core.HMMEngine import HMMClassifier
from gui.RankingPanel import RankingPanel


class SphereCanvas3D(gl.GLViewWidget):
    """Lienzo interactivo 3D sobre una esfera con trazado por ray-casting y proyección geodésica."""

    def __init__(
        self,
        extractor: FeatureExtractor3D,
        on_strokes_changed: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.extractor = extractor
        self.radius = float(extractor.cfg.sphere_radius)
        self.on_strokes_changed = on_strokes_changed

        self.strokes: list[list[np.ndarray]] = []
        self._current_stroke: list[np.ndarray] | None = None
        self.projected_strokes: list[np.ndarray] = []
        self.show_projection: bool = True
        self.rotate_mode: bool = False
        self._last_mouse_pos: QPointF | None = None

        # Orientación de cámara (azimut, elevación) fijada al iniciar el primer trazo.
        # La vista inicial tiene una inclinación sutil (18° de elevación) para que los paralelos y
        # meridianos muestren de inmediato el volumen 3D y la curvatura real del balón.
        self.active_camera_orientation: tuple[float, float] = (0.0, 18.0)

        self.setMinimumSize(420, 420)
        self.setBackgroundColor("#ffffff")
        self.opts["distance"] = 2.05 * self.radius
        self.opts["fov"] = 60
        self.opts["azimuth"] = 0.0
        self.opts["elevation"] = 18.0
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._stroke_items: list[gl.GLLinePlotItem | gl.GLScatterPlotItem | gl.GLMeshItem] = []
        self._proj_items: list[gl.GLLinePlotItem | gl.GLScatterPlotItem] = []
        self._guide_items: list[gl.GLLinePlotItem] = []

        self._init_sphere_scene()

    def current_camera_orientation(self) -> tuple[float, float]:
        """Devuelve (azimuth_deg, elevation_deg) actuales de la cámara."""
        return float(self.opts["azimuth"]), float(self.opts["elevation"])

    def _init_sphere_scene(self) -> None:
        """Crea la malla de la esfera, meridianos/paralelos y el recuadro guía ecuatorial."""
        md = gl.MeshData.sphere(rows=56, cols=56, radius=self.radius * 0.994)
        self.sphere_mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(0.93, 0.96, 0.99, 1.0),
            shader="shaded",
            glOptions="opaque",
        )
        self.addItem(self.sphere_mesh)

        # Grilla esférica de paralelos y meridianos (da percepción real de volumen y curvatura 3D)
        r_grid = self.radius * 1.001
        phis = np.linspace(0.0, 2.0 * np.pi, 96)
        for theta_deg in (20, 40, 60, 80, 90, 100, 120, 140, 160):
            th = np.deg2rad(theta_deg)
            pts = r_grid * np.column_stack(
                [np.sin(th) * np.cos(phis), np.sin(th) * np.sin(phis), np.full_like(phis, np.cos(th))]
            )
            is_eq = theta_deg == 90
            color = (0.42, 0.52, 0.66, 0.65) if is_eq else (0.58, 0.65, 0.75, 0.42)
            width = 1.5 if is_eq else 1.0
            item = gl.GLLinePlotItem(pos=pts, color=color, width=width, antialias=True)
            self.addItem(item)

        thetas = np.linspace(0.0, np.pi, 96)
        for phi_deg in range(0, 180, 20):
            ph = np.deg2rad(phi_deg)
            pts = r_grid * np.column_stack(
                [np.sin(thetas) * np.cos(ph), np.sin(thetas) * np.sin(ph), np.cos(thetas)]
            )
            is_prime = phi_deg in (0, 90)
            color = (0.42, 0.52, 0.66, 0.60) if is_prime else (0.58, 0.65, 0.75, 0.42)
            width = 1.4 if is_prime else 1.0
            item = gl.GLLinePlotItem(pos=pts, color=color, width=width, antialias=True)
            self.addItem(item)

        self._update_guide_frame(self.current_camera_orientation())

    def _update_guide_frame(self, cam_orient: tuple[float, float]) -> None:
        """Actualiza el parche guía ecuatorial centrado en la orientación indicada."""
        for item in self._guide_items:
            self.removeItem(item)
        self._guide_items.clear()

        curves = self.extractor.build_guide_frame_curves(
            camera_rotation=cam_orient, radius=self.radius
        )
        for idx, curve in enumerate(curves):
            is_border = idx < 4
            color = (0.25, 0.42, 0.68, 0.85) if is_border else (0.45, 0.55, 0.68, 0.45)
            width = 2.0 if is_border else 1.0
            line = gl.GLLinePlotItem(pos=curve, color=color, width=width, antialias=True)
            self.addItem(line)
            self._guide_items.append(line)

    # --- Intersección y mapeo esférico con deformación de curvatura -----------
    def _screen_to_sphere_point(self, x_px: float, y_px: float) -> np.ndarray | None:
        """Mapea el píxel lógico (x, y) a la esfera S² aplicando deformación esférica real.

        Combina la intersección rayo-esfera de OpenGL con la parametrización de coordenadas
        esféricas intrínsecas (θ, φ) del marco de cámara. De esta forma, al trazar con el
        ratón sobre el balón, las líneas rectas del ratón siguen los paralelos y meridianos
        esféricos y se curvan/deforman visualmente en pantalla según la zona de la esfera
        donde se dibuje (centro vs bordes/polos), exactamente como al pintar sobre un balón.
        """
        w = max(1, int(self.width()))
        h = max(1, int(self.height()))
        ndc_x = 2.0 * (float(x_px) / w) - 1.0
        ndc_y = 1.0 - 2.0 * (float(y_px) / h)

        # GLViewWidget.projectionMatrix requiere (region, viewport) en coordenadas lógicas (w, h)
        logical_vp = (0, 0, w, h)
        mvp = self.projectionMatrix(logical_vp, logical_vp) * self.viewMatrix()
        inv_mvp, invertible = mvp.inverted()
        if not invertible:
            return None

        near_4 = inv_mvp.map(QVector4D(float(ndc_x), float(ndc_y), -1.0, 1.0))
        far_4 = inv_mvp.map(QVector4D(float(ndc_x), float(ndc_y), 1.0, 1.0))
        if abs(near_4.w()) < 1e-9 or abs(far_4.w()) < 1e-9:
            return None

        p_near = np.array([near_4.x(), near_4.y(), near_4.z()], dtype=float) / near_4.w()
        p_far = np.array([far_4.x(), far_4.y(), far_4.z()], dtype=float) / far_4.w()
        ray_dir = p_far - p_near
        hit = self.extractor.intersect_ray_sphere(p_near, ray_dir, radius=self.radius)
        if hit is None:
            return None

        # Proyectar el punto de impacto en la base ortonormal de la cámara actual
        # para obtener las coordenadas tangentes normalizadas (u, v) sobre el disco del balón
        cam_orient = self.current_camera_orientation()
        R_cam = self.extractor.build_camera_matrix(cam_orient)
        hit_cam = (hit / max(float(np.linalg.norm(hit)), 1e-12)) @ R_cam.T
        u = float(hit_cam[1])   # derecha en pantalla ∈ [-1, 1]
        v = float(-hit_cam[2])  # abajo en pantalla ∈ [-1, 1]

        # Mapear (u, v) mediante la carta esférica intrínseca (Δφ, Δθ) para que el trazo
        # se deforme con la curvatura gaussiana real de la esfera al acercarse al borde/polos
        deformed = self.extractor.map_screen_delta_to_sphere(
            u=u,
            v=v,
            camera_rotation=cam_orient,
            radius=self.radius,
            angular_gain=1.15,
        )
        return deformed if deformed is not None else hit

    # --- Gestión de trazos y renderizado 3D -----------------------------------
    def get_numpy_strokes(self) -> list[np.ndarray]:
        return [np.vstack(s) for s in self.strokes if len(s) > 0]

    def clear_strokes(self) -> None:
        self.strokes.clear()
        self._current_stroke = None
        self.projected_strokes.clear()
        self.active_camera_orientation = self.current_camera_orientation()
        self._update_guide_frame(self.active_camera_orientation)
        self._rebuild_stroke_items()
        self._rebuild_projection_items()
        self.on_strokes_changed()

    def undo_substroke(self) -> None:
        if self.strokes:
            self.strokes.pop()
            self._current_stroke = None
            self._rebuild_stroke_items()
            self.on_strokes_changed()

    def reset_camera_view(self) -> None:
        """Reorienta la cámara a la vista estándar del balón."""
        self.opts["azimuth"] = 0.0
        self.opts["elevation"] = 18.0
        self.opts["distance"] = 2.05 * self.radius
        if not self.strokes:
            self.active_camera_orientation = (0.0, 18.0)
            self._update_guide_frame(self.active_camera_orientation)
        self.update()

    def set_projected_strokes(self, projected: list[np.ndarray]) -> None:
        self.projected_strokes = projected
        self._rebuild_projection_items()

    def _rebuild_stroke_items(self) -> None:
        for item in self._stroke_items:
            self.removeItem(item)
        self._stroke_items.clear()

        r_lift = self.radius * 1.006
        for idx, stroke_pts in enumerate(self.strokes):
            if not stroke_pts:
                continue
            raw_pts = np.vstack(stroke_pts)
            if len(raw_pts) == 1:
                p_single = self.extractor.densify_spherical_polyline(raw_pts, radius=r_lift)
                dot = gl.GLScatterPlotItem(
                    pos=p_single, color=(0.09, 0.27, 0.78, 1.0), size=9.0, pxMode=True
                )
                self.addItem(dot)
                self._stroke_items.append(dot)
            else:
                # 1. Cinta tangente 3D (GLMeshItem) que sufre escorzo y deformación de anchura sobre la esfera
                ribbon = self.extractor.build_spherical_ribbon_mesh(
                    raw_pts, radius=r_lift, half_width_rad=0.017, max_angle_step=0.012
                )
                if ribbon is not None:
                    verts, faces = ribbon
                    md_ribbon = gl.MeshData(vertexes=verts, faces=faces)
                    ribbon_item = gl.GLMeshItem(
                        meshdata=md_ribbon,
                        smooth=False,
                        color=(0.09, 0.27, 0.78, 0.96),
                        glOptions="translucent",
                    )
                    self.addItem(ribbon_item)
                    self._stroke_items.append(ribbon_item)

                # 2. Contorno central antialias sobre el arco geodésico (SLERP)
                curved_pts = self.extractor.densify_spherical_polyline(
                    raw_pts, radius=r_lift * 1.001, max_angle_step=0.012
                )
                line = gl.GLLinePlotItem(
                    pos=curved_pts, color=(0.07, 0.22, 0.68, 1.0), width=2.8, antialias=True
                )
                self.addItem(line)
                self._stroke_items.append(line)

            # Puntos naranjas sobre el arco geodésico si hubo salto de pluma levantada entre sub-trazos
            if idx > 0 and self.strokes[idx - 1]:
                p0 = self.strokes[idx - 1][-1]
                p1 = stroke_pts[0]
                arc_pts = self.extractor.slerp_arc(
                    p0, p1, radius=r_lift, max_angle_step=0.04
                )
                jump_dots = gl.GLScatterPlotItem(
                    pos=arc_pts, color=(0.85, 0.46, 0.02, 0.90), size=4.5, pxMode=True
                )
                self.addItem(jump_dots)
                self._stroke_items.append(jump_dots)
        self.update()

    def _rebuild_projection_items(self) -> None:
        for item in self._proj_items:
            self.removeItem(item)
        self._proj_items.clear()

        if not self.show_projection or not self.projected_strokes:
            self.update()
            return

        r_lift = self.radius * 1.009
        for p_stroke in self.projected_strokes:
            if len(p_stroke) < 2:
                continue
            lifted_nodes = self.extractor.densify_spherical_polyline(
                p_stroke, radius=r_lift, max_angle_step=10.0  # conservar nodos originales a radio r_lift
            )
            curved_proj = self.extractor.densify_spherical_polyline(
                p_stroke, radius=r_lift, max_angle_step=0.018
            )
            proj_ribbon = self.extractor.build_spherical_ribbon_mesh(
                p_stroke, radius=r_lift, half_width_rad=0.011, max_angle_step=0.018
            )
            if proj_ribbon is not None:
                verts, faces = proj_ribbon
                md_proj = gl.MeshData(vertexes=verts, faces=faces)
                mesh_item = gl.GLMeshItem(
                    meshdata=md_proj,
                    smooth=False,
                    color=(0.09, 0.64, 0.29, 0.42),
                    glOptions="translucent",
                )
                self.addItem(mesh_item)
                self._proj_items.append(mesh_item)

            dots = gl.GLScatterPlotItem(
                pos=lifted_nodes, color=(0.09, 0.64, 0.29, 0.95), size=6.0, pxMode=True
            )
            line = gl.GLLinePlotItem(
                pos=curved_proj, color=(0.09, 0.64, 0.29, 0.70), width=2.0, antialias=True
            )
            self.addItem(line)
            self.addItem(dots)
            self._proj_items.extend([line, dots])
        self.update()

    # --- Eventos de ratón (trazado vs rotación de la esfera) ------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        self._last_mouse_pos = pos

        if event.button() == Qt.MouseButton.LeftButton and not self.rotate_mode:
            if not self.strokes:
                self.active_camera_orientation = self.current_camera_orientation()
                self._update_guide_frame(self.active_camera_orientation)
            hit = self._screen_to_sphere_point(pos.x(), pos.y())
            if hit is not None:
                self._current_stroke = [hit]
                self.strokes.append(self._current_stroke)
                self._rebuild_stroke_items()
                self.on_strokes_changed()
                event.accept()
                return
        self._current_stroke = None
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if (
            self._current_stroke is not None
            and (event.buttons() & Qt.MouseButton.LeftButton)
            and not self.rotate_mode
        ):
            hit = self._screen_to_sphere_point(pos.x(), pos.y())
            if hit is not None:
                if np.linalg.norm(hit - self._current_stroke[-1]) > 0.003 * self.radius:
                    self._current_stroke.append(hit)
                    self._rebuild_stroke_items()
                    self.on_strokes_changed()
            self._last_mouse_pos = pos
            event.accept()
            return

        # Rotación orbital de la esfera con clic derecho, botón central o casilla "Rotar"
        if self._last_mouse_pos is not None and (
            (event.buttons() & Qt.MouseButton.RightButton)
            or (event.buttons() & Qt.MouseButton.MiddleButton)
            or (self.rotate_mode and (event.buttons() & Qt.MouseButton.LeftButton))
        ):
            dx = pos.x() - self._last_mouse_pos.x()
            dy = pos.y() - self._last_mouse_pos.y()
            self.orbit(-dx * 0.45, dy * 0.45)
            if not self.strokes:
                self.active_camera_orientation = self.current_camera_orientation()
                self._update_guide_frame(self.active_camera_orientation)
            self.update()
        self._last_mouse_pos = pos
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._current_stroke is not None:
            self._current_stroke = None
            self.on_strokes_changed()
        self._last_mouse_pos = None
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom suave sobre la esfera acotado a un rango seguro."""
        delta = event.angleDelta().y()
        factor = 0.92 if delta > 0 else 1.08
        new_dist = float(np.clip(self.opts["distance"] * factor, 1.45 * self.radius, 5.0 * self.radius))
        self.opts["distance"] = new_dist
        self.update()
        event.accept()


class View3D(QWidget):
    """Ventana C: Reconocedor y Visualizador 3D (Lienzo Esférico)."""

    def __init__(
        self,
        classifier: HMMClassifier,
        extractor: FeatureExtractor3D,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cfg = get_config()
        self.classifier = classifier
        self.extractor = extractor

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Columna principal izquierda: cabecera + lienzo 3D + pie
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(28, 24, 24, 24)
        left_layout.setSpacing(14)

        # Barra superior: Título a la izquierda, botones a la derecha
        top_bar = QHBoxLayout()
        header_v = QVBoxLayout()
        header_v.setSpacing(2)

        lbl_title = QLabel("Modo 3D")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #1e293b;")
        lbl_sub = QLabel("Superficie esférica")
        lbl_sub.setStyleSheet("font-size: 13px; color: #64748b;")
        header_v.addWidget(lbl_title)
        header_v.addWidget(lbl_sub)
        top_bar.addLayout(header_v, stretch=1)

        # Botones de acción en la esquina superior derecha
        self.btn_rotate_mode = QPushButton("🔄 Girar esfera: OFF")
        self.btn_rotate_mode.setCheckable(True)
        self.btn_rotate_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rotate_mode.setStyleSheet(self._toggle_button_style(False))
        self.btn_rotate_mode.toggled.connect(self._on_toggle_rotate_mode)

        self.btn_reorient = QPushButton("Centrar esfera")
        self.btn_reorient.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reorient.setStyleSheet(self._button_style())
        self.btn_reorient.clicked.connect(self._reset_camera_and_canvas)

        self.btn_restart = QPushButton("Reiniciar captura")
        self.btn_restart.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_restart.setStyleSheet(self._button_style())
        self.btn_restart.clicked.connect(self.reset_capture)

        self.btn_clear = QPushButton("Limpiar lienzo")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setStyleSheet(self._button_style())
        self.btn_clear.clicked.connect(self.reset_capture)

        top_bar.addWidget(self.btn_rotate_mode)
        top_bar.addWidget(self.btn_reorient)
        top_bar.addWidget(self.btn_restart)
        top_bar.addWidget(self.btn_clear)
        left_layout.addLayout(top_bar)

        # Contenedor de la esfera 3D
        sphere_frame = QFrame()
        sphere_frame.setObjectName("SphereViewportFrame")
        sphere_frame.setStyleSheet(
            "QFrame#SphereViewportFrame { background-color: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 12px; }"
        )
        sf_layout = QVBoxLayout(sphere_frame)
        sf_layout.setContentsMargins(4, 4, 4, 4)
        self.canvas3d = SphereCanvas3D(
            extractor=self.extractor,
            on_strokes_changed=self._evaluate_realtime,
        )
        sf_layout.addWidget(self.canvas3d)
        left_layout.addWidget(sphere_frame, stretch=1)

        # Pie inferior
        self.lbl_obs = QLabel(
            "Mantén presionado el clic izquierdo para trazar sobre la esfera. "
            "Clic derecho o rueda para rotar y zoom."
        )
        self.lbl_obs.setStyleSheet("font-size: 12px; color: #64748b;")
        left_layout.addWidget(self.lbl_obs)

        root.addWidget(left_widget, stretch=1)

        # Panel lateral derecho de ranking
        self.ranking_panel = RankingPanel(self)
        root.addWidget(self.ranking_panel)

    def _button_style(self) -> str:
        return (
            "QPushButton { "
            "background-color: #ffffff; "
            "border: 1px solid #cbd5e1; "
            "border-radius: 6px; "
            "padding: 7px 14px; "
            "font-size: 12px; "
            "font-weight: 500; "
            "color: #334155; "
            "} "
            "QPushButton:hover { "
            "background-color: #f1f5f9; "
            "border-color: #94a3b8; "
            "}"
        )

    def _toggle_button_style(self, active: bool) -> str:
        if active:
            return (
                "QPushButton { "
                "background-color: #0f766e; "
                "border: 1px solid #0f766e; "
                "border-radius: 6px; "
                "padding: 7px 14px; "
                "font-size: 12px; "
                "font-weight: 700; "
                "color: #ffffff; "
                "} "
                "QPushButton:hover { "
                "background-color: #115e59; "
                "}"
            )
        return self._button_style()

    def _reset_camera_and_canvas(self) -> None:
        self.canvas3d.reset_camera_view()

    def reset_capture(self) -> None:
        """Cierra la captura actual en 3D y limpia la esfera."""
        self.canvas3d.clear_strokes()
        self.ranking_panel.reset()

    def _on_toggle_rotate_mode(self, checked: bool) -> None:
        self.canvas3d.rotate_mode = checked
        self.btn_rotate_mode.setText("🔄 Girar esfera: ON" if checked else "🔄 Girar esfera: OFF")
        self.btn_rotate_mode.setStyleSheet(self._toggle_button_style(checked))
        self.canvas3d.setCursor(
            Qt.CursorShape.OpenHandCursor if checked else Qt.CursorShape.CrossCursor
        )

    def _evaluate_realtime(self) -> None:
        strokes = self.canvas3d.get_numpy_strokes()
        if not strokes:
            self.canvas3d.set_projected_strokes([])
            self.ranking_panel.reset_ranking()
            az, el = self.canvas3d.current_camera_orientation()
            self.lbl_obs.setText(
                f"Esfera lista (azimut={az:.0f}°, elevación={el:.0f}°). "
                "Clic izquierdo dibuja sobre la esfera; clic derecho rota la esfera."
            )
            return

        res = self.extractor.extract(
            strokes,
            camera_rotation=self.canvas3d.active_camera_orientation,
            realtime=True,
        )
        symbols = res.symbols.tolist()
        if not symbols:
            self.canvas3d.set_projected_strokes([])
            self.ranking_panel.reset_ranking()
            self.lbl_obs.setText(f"Sub-trazos 3D: {len(strokes)} | Continúa trazando sobre la esfera...")
            return

        ranking, top_letter, current_state, future_symbols = self.classifier.predict_and_project(
            symbols, max_symbols=45
        )
        n_states = self.classifier.models[top_letter].n_states if top_letter else None
        self.ranking_panel.update_ranking(
            ranking=ranking,
            top_letter=top_letter,
            current_state=current_state,
            n_states=n_states,
            n_obs=len(symbols),
        )

        projected_3d = self.extractor.reconstruct_projected_strokes(res, future_symbols)
        self.canvas3d.set_projected_strokes(projected_3d)

        last_sph = res.spherical_strokes[-1][-1]
        theta_deg = float(np.rad2deg(last_sph[0]))
        phi_deg = float(np.rad2deg(last_sph[1]))
        sym_txt = " ".join(self.extractor.symbol_label(s) for s in symbols[-14:])
        self.lbl_obs.setText(
            f"(θ={theta_deg:5.1f}°, φ={phi_deg:5.1f}°)  |  Sub-trazos: {len(strokes)}  |  "
            f"T = {len(symbols)}  |  Proyección '{top_letter}': {len(future_symbols)} pasos  |  {sym_txt}"
        )

