"""Ventana B: Reconocedor y Visualizador 2D (SPEC §2.2).

Sección Izquierda:
- Lienzo interactivo 2D con recuadro guía de lado expected_letter_size.
- Levantar el ratón solo separa sub-trazos; únicamente los botones
  "Limpiar Lienzo" y "Reiniciar Captura" cierran la captura.
- Proyección dinámica de trayectoria futura (autocompletado espacial punteado)
  actualizada en tiempo real paso a paso mientras el usuario dibuja.

Sección Derecha:
- RankingPanel ordenado descendentemente según P(Carácter | O_{1:t}) con iconos.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
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
from core.FeatureExtractor import FeatureExtractor2D
from core.HMMEngine import HMMClassifier
from gui.RankingPanel import RankingPanel


class RecognizerCanvas2D(QFrame):
    """Lienzo interactivo 2D para trazado a mano alzada y proyección futura punteada."""

    def __init__(
        self,
        guide_size: float,
        on_strokes_changed: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Canvas2DFrame")
        self.guide_size = float(guide_size)
        self.on_strokes_changed = on_strokes_changed

        self.strokes: list[list[tuple[float, float]]] = []
        self._current_stroke: list[tuple[float, float]] | None = None
        self.projected_strokes: list[np.ndarray] = []
        self.show_projection: bool = True
        self._last_center: tuple[float, float] | None = None

        self.setMinimumSize(420, 420)
        self.setStyleSheet(
            "QFrame#Canvas2DFrame { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; }"
        )
        self.setCursor(Qt.CursorShape.CrossCursor)

    def guide_rect(self) -> QRectF:
        w, h = self.width(), self.height()
        s = self.guide_size
        return QRectF((w - s) / 2.0, (h - s) / 2.0, s, s)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Mantiene los trazos centrados en el recuadro guía al maximizar o redimensionar."""
        super().resizeEvent(event)
        new_cx = self.width() / 2.0
        new_cy = self.height() / 2.0
        if self._last_center is not None and self.strokes:
            dx = new_cx - self._last_center[0]
            dy = new_cy - self._last_center[1]
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                for s in self.strokes:
                    for i, (x, y) in enumerate(s):
                        s[i] = (x + dx, y + dy)
                for p in self.projected_strokes:
                    if len(p) > 0:
                        p[:, 0] += dx
                        p[:, 1] += dy
        self._last_center = (new_cx, new_cy)

    def clear_strokes(self) -> None:
        """Limpia los sub-trazos y la proyección actual."""
        self.strokes.clear()
        self._current_stroke = None
        self.projected_strokes.clear()
        self.update()
        self.on_strokes_changed()

    def undo_substroke(self) -> None:
        if self.strokes:
            self.strokes.pop()
            self._current_stroke = None
            self.update()
            self.on_strokes_changed()

    def get_numpy_strokes(self) -> list[np.ndarray]:
        return [np.asarray(s, dtype=float) for s in self.strokes if len(s) > 0]

    def set_projected_strokes(self, projected: list[np.ndarray]) -> None:
        self.projected_strokes = projected
        self.update()

    # --- Eventos de ratón (levantar el ratón solo separa sub-trazos) ----------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._current_stroke = [(pos.x(), pos.y())]
            self.strokes.append(self._current_stroke)
            self.on_strokes_changed()
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._current_stroke is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self._current_stroke.append((pos.x(), pos.y()))
            self.on_strokes_changed()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._current_stroke = None
            self.on_strokes_changed()
            self.update()

    # --- Renderizado ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.guide_rect()

        # Recuadro guía de tamaño expected_letter_size
        guide_pen = QPen(QColor("#94a3b8"), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(guide_pen)
        painter.drawRect(rect)

        # Ejes centrales sutiles
        axis_pen = QPen(QColor("#e2e8f0"), 1.0, Qt.PenStyle.DotLine)
        painter.setPen(axis_pen)
        painter.drawLine(
            QPointF(rect.left(), rect.center().y()),
            QPointF(rect.right(), rect.center().y()),
        )
        painter.drawLine(
            QPointF(rect.center().x(), rect.top()),
            QPointF(rect.center().x(), rect.bottom()),
        )

        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            int(rect.left() + 4),
            int(rect.top() - 6),
            f"Área guía ({int(self.guide_size)}×{int(self.guide_size)} px)",
        )

        # 1. Proyección dinámica futura (verde punteado)
        if self.show_projection and self.projected_strokes:
            proj_pen = QPen(QColor(22, 163, 74, 210), 3.0, Qt.PenStyle.DashLine)
            proj_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            proj_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(proj_pen)

            for p_idx, p_stroke in enumerate(self.projected_strokes):
                if len(p_stroke) < 2:
                    continue
                path = QPainterPath(QPointF(float(p_stroke[0, 0]), float(p_stroke[0, 1])))
                for pt in p_stroke[1:]:
                    path.lineTo(QPointF(float(pt[0]), float(pt[1])))
                painter.drawPath(path)

                if p_idx > 0 and len(self.projected_strokes[p_idx - 1]) > 0:
                    prev_end = self.projected_strokes[p_idx - 1][-1]
                    curr_start = p_stroke[0]
                    jump_proj_pen = QPen(QColor(217, 119, 6, 140), 1.3, Qt.PenStyle.DotLine)
                    painter.setPen(jump_proj_pen)
                    painter.drawLine(
                        QPointF(float(prev_end[0]), float(prev_end[1])),
                        QPointF(float(curr_start[0]), float(curr_start[1])),
                    )
                    painter.setPen(proj_pen)

            last_proj = self.projected_strokes[-1][-1]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(22, 163, 74, 210))
            painter.drawEllipse(QPointF(float(last_proj[0]), float(last_proj[1])), 4.0, 4.0)

        # 2. Trazos realizados por el usuario (azul oscuro nítido)
        user_pen = QPen(QColor("#1d4ed8"), 3.8, Qt.PenStyle.SolidLine)
        user_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        user_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(user_pen)

        for idx, stroke in enumerate(self.strokes):
            if not stroke:
                continue
            if len(stroke) == 1:
                painter.drawEllipse(QPointF(*stroke[0]), 3.2, 3.2)
            else:
                path = QPainterPath(QPointF(*stroke[0]))
                for x, y in stroke[1:]:
                    path.lineTo(QPointF(x, y))
                painter.drawPath(path)

            if idx > 0 and self.strokes[idx - 1]:
                prev_pt = QPointF(*self.strokes[idx - 1][-1])
                curr_pt = QPointF(*stroke[0])
                jump_pen = QPen(QColor("#d97706"), 1.4, Qt.PenStyle.DotLine)
                painter.setPen(jump_pen)
                painter.drawLine(prev_pt, curr_pt)
                painter.setPen(user_pen)


class View2D(QWidget):
    """Ventana B: Reconocedor y Visualizador 2D."""

    def __init__(
        self,
        classifier: HMMClassifier,
        extractor: FeatureExtractor2D,
        on_back_menu: Callable[[], None],
        on_switch_3d: Callable[[], None],
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

        lbl_header = QLabel("Modo 2D — Reconocedor y Visualizador en Plano Cartesiano")
        lbl_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        lbl_header.setStyleSheet("color: #1e293b; background: transparent;")

        btn_to_3d = QPushButton("Ir a Modo 3D (Esfera) →")
        btn_to_3d.setStyleSheet(self._primary_btn_style())
        btn_to_3d.clicked.connect(on_switch_3d)

        nav.addWidget(btn_menu)
        nav.addSpacing(10)
        nav.addWidget(lbl_header, stretch=1)
        nav.addWidget(btn_to_3d)
        root.addLayout(nav)

        # --- Cuerpo principal: Izquierda (Lienzo) | Derecha (Ranking) ---------
        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, stretch=1)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        body.addLayout(left_col, stretch=1)

        self.canvas = RecognizerCanvas2D(
            guide_size=self.cfg.features.expected_letter_size,
            on_strokes_changed=self._evaluate_realtime,
        )
        left_col.addWidget(self.canvas, stretch=1)

        self.lbl_obs = QLabel(
            "Dibuja con clic izquierdo dentro del recuadro guía. Levantar el ratón separa sub-trazos."
        )
        self.lbl_obs.setWordWrap(True)
        self.lbl_obs.setFont(QFont("Segoe UI", 9))
        self.lbl_obs.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; "
            "padding: 6px 10px; color: #475569;"
        )
        left_col.addWidget(self.lbl_obs)

        # Controles: "Limpiar Lienzo" y "Reiniciar Captura"
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
        self.btn_undo.clicked.connect(self.canvas.undo_substroke)

        self.chk_proj = QCheckBox("Mostrar proyección punteada")
        self.chk_proj.setChecked(True)
        self.chk_proj.setStyleSheet("color: #1e293b; font-weight: bold;")
        self.chk_proj.toggled.connect(self._on_toggle_projection)

        ctrl_row.addWidget(self.btn_clear)
        ctrl_row.addWidget(self.btn_restart)
        ctrl_row.addWidget(self.btn_undo)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(self.chk_proj)
        left_col.addLayout(ctrl_row)

        # Panel derecho de clasificación con ancho fijo para no deformarse al maximizar
        self.ranking_panel = RankingPanel(title="Clasificación 2D — P(Carácter | O₁:ₜ)")
        body.addWidget(self.ranking_panel, stretch=0)
        self.ranking_panel.reset_ranking()

    def reset_capture(self) -> None:
        """Cierra la captura actual y reinicia el lienzo y las probabilidades."""
        self.canvas.clear_strokes()
        self.ranking_panel.reset_ranking()

    def _on_toggle_projection(self, checked: bool) -> None:
        self.canvas.show_projection = checked
        self.canvas.update()

    def _evaluate_realtime(self) -> None:
        strokes = self.canvas.get_numpy_strokes()
        if not strokes:
            self.canvas.set_projected_strokes([])
            self.ranking_panel.reset_ranking()
            self.lbl_obs.setText(
                "Lienzo listo. Levantar el ratón solo separa sub-trazos; usa 'Limpiar Lienzo' "
                "o 'Reiniciar Captura' para empezar otra letra."
            )
            return

        res = self.extractor.extract(strokes, realtime=True)
        symbols = res.symbols.tolist()
        if not symbols:
            self.canvas.set_projected_strokes([])
            self.ranking_panel.reset_ranking()
            self.lbl_obs.setText(f"Sub-trazos: {len(strokes)} | Continúa trazando...")
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

        projected = self.extractor.reconstruct_projected_strokes(res, future_symbols)
        self.canvas.set_projected_strokes(projected)

        sym_txt = " ".join(self.extractor.symbol_label(s) for s in symbols[-16:])
        self.lbl_obs.setText(
            f"Sub-trazos: {len(strokes)}  |  Observaciones T = {len(symbols)}  |  "
            f"Proyección '{top_letter}': {len(future_symbols)} pasos  |  Últimos: {sym_txt}"
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
