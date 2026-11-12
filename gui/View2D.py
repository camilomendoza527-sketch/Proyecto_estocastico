"""Ventana Modo 2D: Reconocedor y Visualizador en Plano Cartesiano (gui/View2D.py).

Diseño limpio acorde a la Imagen 2:
- Barra superior: 'Modo 2D' (título), 'Plano cartesiano' (subtítulo) y botones [Reiniciar captura], [Limpiar lienzo].
- Lienzo central: tarjeta blanca redondeada con cruz central tenue, marca de agua 'Dibuja una letra aquí' y pie informativo.
- Panel derecho: clasificación en tiempo real compacta con glifos de referencia.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
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
        on_strokes_changed: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Canvas2DFrame")
        self.on_strokes_changed = on_strokes_changed

        self.strokes: list[list[tuple[float, float]]] = []
        self._current_stroke: list[tuple[float, float]] | None = None
        self.projected_strokes: list[np.ndarray] = []
        self.show_projection: bool = True
        self._last_center: tuple[float, float] | None = None

        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QFrame#Canvas2DFrame { background-color: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 12px; }"
        )
        self.setCursor(Qt.CursorShape.CrossCursor)

    def resizeEvent(self, event) -> None:  # noqa: N802
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

    def get_numpy_strokes(self) -> list[np.ndarray]:
        return [np.asarray(s, dtype=float) for s in self.strokes if len(s) > 0]

    def set_projected_strokes(self, projected: list[np.ndarray]) -> None:
        self.projected_strokes = projected
        self.update()

    # --- Eventos de ratón -----------------------------------------------------
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
            if self._current_stroke:
                lx, ly = self._current_stroke[-1]
                if (pos.x() - lx) ** 2 + (pos.y() - ly) ** 2 < 2.25:
                    return
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

        w = float(self.width())
        h = float(self.height())
        cx = w / 2.0
        cy = h / 2.0

        # Ejes centrales tenues (cruz en el centro del lienzo)
        axis_pen = QPen(QColor("#f1f5f9"), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(axis_pen)
        painter.drawLine(QPointF(20, cy), QPointF(w - 20, cy))
        painter.drawLine(QPointF(cx, 20), QPointF(cx, h - 20))

        # Texto marca de agua cuando el lienzo está vacío
        has_strokes = any(len(s) > 0 for s in self.strokes)
        if not has_strokes:
            painter.setPen(QColor("#cbd5e1"))
            font = QFont("Segoe UI", 16, QFont.Weight.Normal)
            painter.setFont(font)
            rect = QRectF(0, 0, w, h)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Dibuja una letra aquí")
            return

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
        user_pen = QPen(QColor("#1e293b"), 3.8, Qt.PenStyle.SolidLine)
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
    """Pantalla de Modo 2D con lienzo y clasificación lateral."""

    def __init__(
        self,
        classifier: HMMClassifier,
        extractor: FeatureExtractor2D,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cfg = get_config()
        self.classifier = classifier
        self.extractor = extractor

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Columna principal izquierda: cabecera + lienzo + pie
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(28, 24, 24, 24)
        left_layout.setSpacing(14)

        # Barra superior: Título a la izquierda, botones a la derecha
        top_bar = QHBoxLayout()
        header_v = QVBoxLayout()
        header_v.setSpacing(2)

        lbl_title = QLabel("Modo 2D")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #1e293b;")
        lbl_sub = QLabel("Plano cartesiano")
        lbl_sub.setStyleSheet("font-size: 13px; color: #64748b;")
        header_v.addWidget(lbl_title)
        header_v.addWidget(lbl_sub)
        top_bar.addLayout(header_v, stretch=1)

        # Botones de acción en la esquina superior derecha
        self.btn_restart = QPushButton("Reiniciar captura")
        self.btn_restart.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_restart.setStyleSheet(self._button_style())
        self.btn_restart.clicked.connect(self.reset_capture)

        self.btn_clear = QPushButton("Limpiar lienzo")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setStyleSheet(self._button_style())
        self.btn_clear.clicked.connect(self.reset_capture)

        top_bar.addWidget(self.btn_restart)
        top_bar.addWidget(self.btn_clear)
        left_layout.addLayout(top_bar)

        # Lienzo de dibujo central
        self.canvas = RecognizerCanvas2D(on_strokes_changed=self._evaluate_realtime)
        left_layout.addWidget(self.canvas, stretch=1)

        # Pie inferior
        lbl_footer = QLabel("Mantén presionado el clic izquierdo para trazar.")
        lbl_footer.setStyleSheet("font-size: 12px; color: #64748b;")
        left_layout.addWidget(lbl_footer)

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

    def reset_capture(self) -> None:
        """Reinicia el lienzo y el panel de ranking."""
        self.canvas.clear_strokes()
        self.ranking_panel.reset()

    def _evaluate_realtime(self) -> None:
        """Ejecuta la extracción de características y clasificador HMM en tiempo real."""
        raw_strokes = self.canvas.get_numpy_strokes()
        if not raw_strokes or all(len(s) < 2 for s in raw_strokes):
            self.ranking_panel.reset()
            self.canvas.set_projected_strokes([])
            return

        res = self.extractor.extract(raw_strokes, realtime=True)
        symbols = res.symbols.tolist()
        if not symbols:
            self.ranking_panel.reset()
            self.canvas.set_projected_strokes([])
            return

        ranking, top_letter, current_state, future_symbols = self.classifier.predict_and_project(
            symbols, max_symbols=45
        )
        if not ranking:
            self.ranking_panel.reset()
            self.canvas.set_projected_strokes([])
            return

        # Autocompletado proyectado futuro
        if self.canvas.show_projection and future_symbols:
            projected = self.extractor.reconstruct_projected_strokes(res, future_symbols)
            self.canvas.set_projected_strokes(projected)
        else:
            self.canvas.set_projected_strokes([])

        self.ranking_panel.update_probabilities(
            ranking=ranking,
            viterbi_state=current_state,
            n_obs=len(symbols),
        )
