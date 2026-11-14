"""Vista interactiva de Guía de Letras (gui/LetterGuide.py).

Muestra el orden, punto de inicio (1, 2) y dirección exacta de cada sub-trazo
acorde al dataset entrenado (data/canonical_glyphs.json), permitiendo animar el
recorrido y practicar el trazo directamente sobre la plantilla.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.config import get_config
from gui.RankingPanel import load_canonical_glyphs

STROKE_COLORS = ("#0f766e", "#d97706", "#7c3aed")


class GuideCanvas(QFrame):
    """Lienzo animado de demostración de trazos y práctica libre."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("GuideCanvasFrame")
        self.setMinimumSize(420, 380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QFrame#GuideCanvasFrame { background-color: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 12px; }"
        )
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._letter: str = "a"
        self._strokes: list[list[tuple[float, float]]] = []
        self._start_points: list[tuple[float, float]] = []
        self._flat_points: list[tuple[int, float, float]] = []

        self._anim_idx: float = 0.0
        self._is_paused: bool = False
        self._show_anim: bool = True
        self._speed_multiplier: float = 1.0

        self.user_strokes: list[list[tuple[float, float]]] = []
        self._current_user_stroke: list[tuple[float, float]] | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._tick_animation)
        self._timer.start()

    def set_letter(self, letter: str) -> None:
        self._letter = letter
        glyph = load_canonical_glyphs().get(letter, {})
        raw_strokes = glyph.get("strokes", [])
        self._strokes = [
            [(float(p[0]), float(p[1])) for p in st if len(p) >= 2]
            for st in raw_strokes
            if len(st) > 0
        ]
        self._start_points = [st[0] for st in self._strokes if st]
        self._flat_points = [
            (s_idx, x, y)
            for s_idx, st in enumerate(self._strokes)
            for (x, y) in st
        ]
        self._anim_idx = 0.0
        self._show_anim = True
        self.user_strokes.clear()
        self._current_user_stroke = None
        self.update()

    def restart_animation(self) -> None:
        self._anim_idx = 0.0
        self._show_anim = True
        self._is_paused = False
        self.update()

    def clear_user_strokes(self) -> None:
        self.user_strokes.clear()
        self._current_user_stroke = None
        self.update()

    def _tick_animation(self) -> None:
        if self._is_paused or not self._show_anim or not self._flat_points:
            return
        self._anim_idx += 1.4 * self._speed_multiplier
        if self._anim_idx >= len(self._flat_points) + 18:
            self._anim_idx = 0.0
        self.update()

    def _to_screen(self, nx: float, ny: float, box: QRectF) -> QPointF:
        return QPointF(box.left() + nx * box.width(), box.top() + ny * box.height())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._current_user_stroke = [(pos.x(), pos.y())]
            self.user_strokes.append(self._current_user_stroke)
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._current_user_stroke is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self._current_user_stroke.append((pos.x(), pos.y()))
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._current_user_stroke = None
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w, h = float(self.width()), float(self.height())
        side = min(w, h) * 0.68
        box = QRectF((w - side) / 2.0, (h - side) / 2.0, side, side)

        # Recuadro guía y ejes centrales
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#e2e8f0"), 1.4, Qt.PenStyle.DashLine))
        painter.drawRoundedRect(box, 12.0, 12.0)
        painter.setPen(QPen(QColor("#f1f5f9"), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(box.left(), box.center().y()), QPointF(box.right(), box.center().y()))
        painter.drawLine(QPointF(box.center().x(), box.top()), QPointF(box.center().x(), box.bottom()))

        # 1. Trazos base completos con flechas de dirección
        for s_idx, st in enumerate(self._strokes):
            color_hex = STROKE_COLORS[s_idx % len(STROKE_COLORS)]
            base_color = QColor(color_hex)
            base_color.setAlpha(65)
            pts = [self._to_screen(x, y, box) for x, y in st]
            if len(pts) == 1:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(base_color)
                painter.drawEllipse(pts[0], 6.0, 6.0)
            elif len(pts) >= 2:
                pen = QPen(base_color, 6.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(pen)
                path = QPainterPath(pts[0])
                for p in pts[1:]:
                    path.lineTo(p)
                painter.drawPath(path)
                self._draw_direction_arrows(painter, pts, QColor(color_hex))

            if s_idx > 0 and self._strokes[s_idx - 1]:
                p_prev = self._to_screen(*self._strokes[s_idx - 1][-1], box)
                p_curr = pts[0]
                painter.setPen(QPen(QColor("#d97706"), 1.5, Qt.PenStyle.DotLine))
                painter.drawLine(p_prev, p_curr)

        # 2. Progreso animado del trazo
        if self._show_anim and self._flat_points:
            limit = min(int(self._anim_idx), len(self._flat_points))
            drawn_by_stroke: dict[int, list[QPointF]] = {}
            for i in range(limit):
                s_idx, nx, ny = self._flat_points[i]
                drawn_by_stroke.setdefault(s_idx, []).append(self._to_screen(nx, ny, box))

            for s_idx, pts in drawn_by_stroke.items():
                c = QColor(STROKE_COLORS[s_idx % len(STROKE_COLORS)])
                if len(pts) == 1:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(c)
                    painter.drawEllipse(pts[0], 5.5, 5.5)
                elif len(pts) >= 2:
                    pen = QPen(c, 4.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(pen)
                    path = QPainterPath(pts[0])
                    for p in pts[1:]:
                        path.lineTo(p)
                    painter.drawPath(path)

            if 0 < limit <= len(self._flat_points):
                s_idx, nx, ny = self._flat_points[limit - 1]
                head = self._to_screen(nx, ny, box)
                painter.setPen(QPen(QColor("#ffffff"), 2.0))
                painter.setBrush(QColor(STROKE_COLORS[s_idx % len(STROKE_COLORS)]))
                painter.drawEllipse(head, 7.0, 7.0)

        # 3. Marcadores numerados de inicio (1, 2...)
        for s_idx, (sx, sy) in enumerate(self._start_points):
            sp = self._to_screen(sx, sy, box)
            c = QColor(STROKE_COLORS[s_idx % len(STROKE_COLORS)])
            painter.setPen(QPen(QColor("#ffffff"), 2.0))
            painter.setBrush(c)
            painter.drawEllipse(sp, 11.0, 11.0)
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(QRectF(sp.x() - 11, sp.y() - 11, 22, 22), Qt.AlignmentFlag.AlignCenter, str(s_idx + 1))

        # 4. Trazos de práctica del usuario
        painter.setBrush(Qt.BrushStyle.NoBrush)
        user_pen = QPen(QColor("#1e293b"), 3.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(user_pen)
        for st in self.user_strokes:
            if len(st) >= 2:
                path = QPainterPath(QPointF(*st[0]))
                for x, y in st[1:]:
                    path.lineTo(QPointF(x, y))
                painter.drawPath(path)

    @staticmethod
    def _draw_direction_arrows(painter: QPainter, pts: list[QPointF], color: QColor) -> None:
        if len(pts) < 6:
            return
        indices = [len(pts) // 2, len(pts) - 1]
        arrow_color = QColor(color)
        arrow_color.setAlpha(190)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(arrow_color)
        for idx in indices:
            p_prev = pts[max(0, idx - 3)]
            p_curr = pts[idx]
            dx, dy = p_curr.x() - p_prev.x(), p_curr.y() - p_prev.y()
            if math.hypot(dx, dy) < 2.0:
                continue
            angle = math.atan2(dy, dx)
            size = 7.5
            p1 = QPointF(p_curr.x() - size * math.cos(angle - 0.5), p_curr.y() - size * math.sin(angle - 0.5))
            p2 = QPointF(p_curr.x() - size * math.cos(angle + 0.5), p_curr.y() - size * math.sin(angle + 0.5))
            painter.drawPolygon(QPolygonF([p_curr, p1, p2]))


class LetterGuideView(QWidget):
    """Vista de Guía de Letras con selector de las 27 letras y animación paso a paso."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cfg = get_config()
        self.current_letter: str = "a"
        self._buttons: dict[str, QPushButton] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 22)
        root.setSpacing(12)

        top_bar = QHBoxLayout()
        header_v = QVBoxLayout()
        lbl_title = QLabel("Guía de letras")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #1e293b;")
        lbl_sub = QLabel("Orden, punto de inicio y dirección exacta de cada letra en el dataset")
        lbl_sub.setStyleSheet("font-size: 13px; color: #64748b;")
        header_v.addWidget(lbl_title)
        header_v.addWidget(lbl_sub)
        top_bar.addLayout(header_v, stretch=1)

        self.btn_play_pause = QPushButton("Pausar")
        self.btn_speed = QPushButton("Velocidad: 1.0x")
        self.btn_restart = QPushButton("Reiniciar animación")
        self.btn_clear = QPushButton("Limpiar trazo")

        for b in (self.btn_play_pause, self.btn_speed, self.btn_restart, self.btn_clear):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                "QPushButton { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; "
                "padding: 6px 12px; font-size: 12px; font-weight: 500; color: #334155; } "
                "QPushButton:hover { background: #f1f5f9; }"
            )
            top_bar.addWidget(b)

        self.btn_play_pause.clicked.connect(self._toggle_pause)
        self.btn_speed.clicked.connect(self._cycle_speed)
        self.btn_restart.clicked.connect(self._on_restart)
        self.btn_clear.clicked.connect(lambda: self.demo_canvas.clear_user_strokes())
        root.addLayout(top_bar)

        # Selector de las 27 letras
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, ch in enumerate(self.cfg.letters):
            btn = QPushButton(ch)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(34)
            btn.clicked.connect(lambda _=False, c=ch: self.select_letter(c))
            self._buttons[ch] = btn
            grid.addWidget(btn, i // 14, i % 14)
        root.addLayout(grid)

        self.lbl_info = QLabel()
        self.lbl_info.setStyleSheet(
            "font-size: 13px; font-weight: 500; color: #0f766e; background: #f0fdf4; "
            "border: 1px solid #bbf7d0; border-radius: 8px; padding: 8px 12px;"
        )
        root.addWidget(self.lbl_info)

        self.demo_canvas = GuideCanvas(self)
        root.addWidget(self.demo_canvas, stretch=1)

        self.select_letter("a")

    def select_letter(self, letter: str) -> None:
        self.current_letter = letter
        self.demo_canvas.set_letter(letter)
        n_st = len(self.demo_canvas._strokes)
        if n_st == 1:
            desc = f"Letra {letter} — 1 trazo continuo: Inicia en el círculo 1 (verde) y sigue las flechas sin levantar el ratón."
        elif n_st == 2:
            desc = (
                f"Letra {letter} — 2 sub-trazos: 1) Inicia en 1 (verde). "
                "2) Levanta y completa desde 2 (naranja)."
            )
        else:
            desc = (
                f"Letra {letter} — {n_st} sub-trazos: 1) Inicia en 1 (verde). "
                "2) Continúa en 2 (naranja). 3) Completa desde 3 (morado)."
            )
        self.lbl_info.setText(desc + " Puedes dibujar encima para practicar.")

        for ch, btn in self._buttons.items():
            if ch == letter:
                btn.setStyleSheet(
                    "QPushButton { background: #0f766e; color: #ffffff; font-weight: 700; "
                    "border: 1px solid #0f766e; border-radius: 6px; font-size: 13px; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background: #ffffff; color: #334155; font-weight: 600; "
                    "border: 1px solid #e2e8f0; border-radius: 6px; font-size: 13px; } "
                    "QPushButton:hover { background: #f1f5f9; }"
                )

    def _toggle_pause(self) -> None:
        self.demo_canvas._is_paused = not self.demo_canvas._is_paused
        self.btn_play_pause.setText("Reanudar" if self.demo_canvas._is_paused else "Pausar")

    def _on_restart(self) -> None:
        self.demo_canvas.restart_animation()
        self.btn_play_pause.setText("Pausar")

    def _cycle_speed(self) -> None:
        cur = self.demo_canvas._speed_multiplier
        nxt = 0.5 if cur == 1.0 else (1.5 if cur == 0.5 else 1.0)
        self.demo_canvas._speed_multiplier = nxt
        self.btn_speed.setText(f"Velocidad: {nxt}x")
