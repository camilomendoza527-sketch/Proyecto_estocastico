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
from PySide6.QtGui import QFont, QVector4D
from PySide6.QtWidgets import (
    QCheckBox,
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

        # Orientación de cámara (azimut, elevación) fijada al iniciar el primer trazo
        self.active_camera_orientation: tuple[float, float] = (0.0, 0.0)

        self.setMinimumSize(420, 420)
        self.setBackgroundColor("#e2e8f0")
        self.opts["distance"] = 2.85 * self.radius
        self.opts["fov"] = 45
        self.opts["azimuth"] = 0.0
        self.opts["elevation"] = 0.0
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._stroke_items: list[gl.GLLinePlotItem | gl.GLScatterPlotItem] = []
        self._proj_items: list[gl.GLLinePlotItem | gl.GLScatterPlotItem] = []
        self._guide_items: list[gl.GLLinePlotItem] = []

        self._init_sphere_scene()

    def current_camera_orientation(self) -> tuple[float, float]:
        """Devuelve (azimuth_deg, elevation_deg) actuales de la cámara."""
        return float(self.opts["azimuth"]), float(self.opts["elevation"])

    def _init_sphere_scene(self) -> None:
        """Crea la malla de la esfera, meridianos/paralelos y el recuadro guía ecuatorial."""
        md = gl.MeshData.sphere(rows=44, cols=44, radius=self.radius * 0.994)
        self.sphere_mesh = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(0.94, 0.96, 0.98, 1.0),
            shader="shaded",
            glOptions="opaque",
        )
        self.addItem(self.sphere_mesh)

        # Grilla sutil de paralelos y meridianos
        r_grid = self.radius * 1.001
        phis = np.linspace(0.0, 2.0 * np.pi, 72)
        for theta_deg in (30, 60, 90, 120, 150):
            th = np.deg2rad(theta_deg)
            pts = r_grid * np.column_stack(
                [np.sin(th) * np.cos(phis), np.sin(th) * np.sin(phis), np.full_like(phis, np.cos(th))]
            )
            item = gl.GLLinePlotItem(
                pos=pts, color=(0.58, 0.64, 0.72, 0.45), width=1.0, antialias=True
            )
            self.addItem(item)

        thetas = np.linspace(0.0, np.pi, 72)
        for phi_deg in range(0, 180, 30):
            ph = np.deg2rad(phi_deg)
            pts = r_grid * np.column_stack(
                [np.sin(thetas) * np.cos(ph), np.sin(thetas) * np.sin(ph), np.cos(thetas)]
            )
            item = gl.GLLinePlotItem(
                pos=pts, color=(0.58, 0.64, 0.72, 0.45), width=1.0, antialias=True
            )
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

    # --- Intersección rayo-esfera desde coordenadas de pantalla ---------------
    def _screen_to_sphere_point(self, x_px: float, y_px: float) -> np.ndarray | None:
        """Desproyecta el píxel lógico (x, y) a un rayo 3D y halla su intersección con la esfera."""
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
        return self.extractor.intersect_ray_sphere(p_near, ray_dir, radius=self.radius)

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
        """Reorienta la cámara a la vista frontal estándar."""
        self.opts["azimuth"] = 0.0
        self.opts["elevation"] = 0.0
        self.opts["distance"] = 2.85 * self.radius
        if not self.strokes:
            self.active_camera_orientation = (0.0, 0.0)
            self._update_guide_frame(self.active_camera_orientation)
        self.update()

    def set_projected_strokes(self, projected: list[np.ndarray]) -> None:
        self.projected_strokes = projected
        self._rebuild_projection_items()

    def _rebuild_stroke_items(self) -> None:
        for item in self._stroke_items:
            self.removeItem(item)
        self._stroke_items.clear()

        r_lift = 1.007
        for idx, stroke_pts in enumerate(self.strokes):
            if not stroke_pts:
                continue
            pts = np.vstack(stroke_pts) * r_lift
            if len(pts) == 1:
                dot = gl.GLScatterPlotItem(
                    pos=pts, color=(0.11, 0.31, 0.85, 1.0), size=8.0, pxMode=True
                )
                self.addItem(dot)
                self._stroke_items.append(dot)
            else:
                line = gl.GLLinePlotItem(
                    pos=pts, color=(0.11, 0.31, 0.85, 1.0), width=4.0, antialias=True
                )
                self.addItem(line)
                self._stroke_items.append(line)

            # Puntos naranjas si hubo salto de pluma levantada entre sub-trazos
            if idx > 0 and self.strokes[idx - 1]:
                p0 = self.strokes[idx - 1][-1] * r_lift
                p1 = stroke_pts[0] * r_lift
                t_vals = np.linspace(0.0, 1.0, 10)[:, None]
                chord = (1.0 - t_vals) * p0 + t_vals * p1
                norms = np.linalg.norm(chord, axis=1, keepdims=True)
                arc_pts = chord * (self.radius * r_lift / np.maximum(norms, 1e-12))
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

        r_lift = 1.009
        for p_stroke in self.projected_strokes:
            if len(p_stroke) < 2:
                continue
            lifted = p_stroke * r_lift
            dots = gl.GLScatterPlotItem(
                pos=lifted, color=(0.09, 0.64, 0.29, 0.95), size=6.5, pxMode=True
            )
            line = gl.GLLinePlotItem(
                pos=lifted, color=(0.09, 0.64, 0.29, 0.50), width=2.2, antialias=True
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


class View3D(QWidget):
    """Ventana C: Reconocedor y Visualizador 3D (Lienzo Esférico)."""

    def __init__(
        self,
        classifier: HMMClassifier,
        extractor: FeatureExtractor3D,
        on_back_menu: Callable[[], None],
        on_switch_2d: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cfg = get_config()
        self.classifier = classifier
        self.extractor = extractor

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 14)
        root.setSpacing(10)

        # --- Barra superior de navegación -------------------------------------
        nav = QHBoxLayout()
        btn_menu = QPushButton("← Menú Principal")
        btn_menu.setStyleSheet(self._secondary_btn_style())
        btn_menu.clicked.connect(on_back_menu)

        lbl_header = QLabel("Modo 3D — Reconocedor y Visualizador sobre Esfera")
        lbl_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        lbl_header.setStyleSheet("color: #1e293b; background: transparent;")

        btn_to_2d = QPushButton("← Ir a Modo 2D (Plano)")
        btn_to_2d.setStyleSheet(self._primary_btn_style())
        btn_to_2d.clicked.connect(on_switch_2d)

        nav.addWidget(btn_menu)
        nav.addSpacing(10)
        nav.addWidget(lbl_header, stretch=1)
        nav.addWidget(btn_to_2d)
        root.addLayout(nav)

        # --- Cuerpo principal: Izquierda (Esfera 3D) | Derecha (Ranking) ------
        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, stretch=1)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        body.addLayout(left_col, stretch=1)

        sphere_frame = QFrame()
        sphere_frame.setObjectName("SphereViewportFrame")
        sphere_frame.setStyleSheet(
            "QFrame#SphereViewportFrame { background-color: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 8px; }"
        )
        sf_layout = QVBoxLayout(sphere_frame)
        sf_layout.setContentsMargins(2, 2, 2, 2)
        self.canvas3d = SphereCanvas3D(
            extractor=self.extractor,
            on_strokes_changed=self._evaluate_realtime,
        )
        sf_layout.addWidget(self.canvas3d)
        left_col.addWidget(sphere_frame, stretch=1)

        self.lbl_obs = QLabel(
            "Clic izquierdo + arrastrar sobre la esfera: dibujar trazo  |  "
            "Clic derecho + arrastrar (o rueda): rotar y hacer zoom."
        )
        self.lbl_obs.setWordWrap(True)
        self.lbl_obs.setFont(QFont("Segoe UI", 9))
        self.lbl_obs.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; "
            "padding: 6px 10px; color: #475569;"
        )
        left_col.addWidget(self.lbl_obs)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.btn_clear = QPushButton("Limpiar Lienzo")
        self.btn_clear.setStyleSheet(self._danger_btn_style())
        self.btn_clear.clicked.connect(self.reset_capture)

        self.btn_restart = QPushButton("Reiniciar Captura")
        self.btn_restart.setStyleSheet(self._primary_btn_style())
        self.btn_restart.clicked.connect(self.reset_capture)

        self.btn_undo = QPushButton("Deshacer Sub-trazo")
        self.btn_undo.setStyleSheet(self._secondary_btn_style())
        self.btn_undo.clicked.connect(self.canvas3d.undo_substroke)

        self.btn_reorient = QPushButton("Centrar Esfera")
        self.btn_reorient.setStyleSheet(self._secondary_btn_style())
        self.btn_reorient.clicked.connect(self.canvas3d.reset_camera_view)

        self.chk_rotate = QCheckBox("Rotar con clic izq.")
        self.chk_rotate.setStyleSheet("color: #1e293b;")
        self.chk_rotate.toggled.connect(self._on_toggle_rotate_mode)

        self.chk_proj = QCheckBox("Proyección geodésica")
        self.chk_proj.setChecked(True)
        self.chk_proj.setStyleSheet("color: #1e293b; font-weight: bold;")
        self.chk_proj.toggled.connect(self._on_toggle_projection)

        ctrl_row.addWidget(self.btn_clear)
        ctrl_row.addWidget(self.btn_restart)
        ctrl_row.addWidget(self.btn_undo)
        ctrl_row.addWidget(self.btn_reorient)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(self.chk_rotate)
        ctrl_row.addWidget(self.chk_proj)
        left_col.addLayout(ctrl_row)

        # Panel derecho con ancho fijo para que al maximizar no se estire ni oculte letras
        self.ranking_panel = RankingPanel(title="Clasificación 3D — P(Carácter | O₁:ₜ)")
        body.addWidget(self.ranking_panel, stretch=0)
        self.ranking_panel.reset_ranking()

    def reset_capture(self) -> None:
        """Cierra la captura actual en 3D y limpia la esfera."""
        self.canvas3d.clear_strokes()
        self.ranking_panel.reset_ranking()

    def _on_toggle_rotate_mode(self, checked: bool) -> None:
        self.canvas3d.rotate_mode = checked
        self.canvas3d.setCursor(
            Qt.CursorShape.OpenHandCursor if checked else Qt.CursorShape.CrossCursor
        )

    def _on_toggle_projection(self, checked: bool) -> None:
        self.canvas3d.show_projection = checked
        self.canvas3d._rebuild_projection_items()

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

    @staticmethod
    def _primary_btn_style() -> str:
        return (
            "QPushButton { background-color: #2563eb; color: #ffffff; border: none; "
            "border-radius: 6px; padding: 7px 14px; font-weight: bold; } "
            "QPushButton:hover { background-color: #1d4ed8; }"
        )

    @staticmethod
    def _secondary_btn_style() -> str:
        return (
            "QPushButton { background-color: #ffffff; color: #1e293b; border: 1px solid #cbd5e1; "
            "border-radius: 6px; padding: 7px 14px; font-weight: bold; } "
            "QPushButton:hover { background-color: #f1f5f9; }"
        )

    @staticmethod
    def _danger_btn_style() -> str:
        return (
            "QPushButton { background-color: #dc2626; color: #ffffff; border: none; "
            "border-radius: 6px; padding: 7px 14px; font-weight: bold; } "
            "QPushButton:hover { background-color: #b91c1c; }"
        )
