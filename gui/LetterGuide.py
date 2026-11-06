"""Guía interactiva y visual de trazado de letras (gui/LetterGuide.py).

Permite al usuario:
- Ver la animación automática guiada del lápiz (en bucle continuo a 60 FPS con estela y puntero visual).
- Aprender el orden, dirección y sentido de los trazos gracias a flechas direccionales y puntos numerados (1, 2...).
- Controlar la reproducción de la animación: Pausar, Reanudar, cambiar Velocidad (0.5x Lento, 1x Normal, 1.5x Rápido) o Reiniciar.
- Practicar opcionalmente con el ratón a mano alzada sobre el lienzo escolar.
- Seleccionar cualquiera de las 27 letras del alfabeto español en la cuadrícula lateral.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
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

from core.config import PROJECT_ROOT, get_config


def _load_canonical_glyphs() -> dict[str, dict]:
    """Carga los glifos canónicos de data/canonical_glyphs.json o un fallback."""
    path = PROJECT_ROOT / "data" / "canonical_glyphs.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


class StrokeDemoCanvas(QWidget):
    """Lienzo que muestra la animación automática continua de la letra con lápiz visual,

    renglón caligráfico, y permite opcionalmente practicar a mano alzada con el ratón.
    """

    stroke_drawn = Signal()
    animation_state_changed = Signal(str)  # Emite mensajes de estado para la barra inferior

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._letter: str = "a"
        self._n_strokes: int = 1
        self._strokes: list[np.ndarray] = []
        self._start_points: list[tuple[float, float]] = []

        # Trazos realizados por el usuario con el ratón
        self.user_strokes: list[list[tuple[float, float]]] = []
        self._current_user_stroke: list[tuple[float, float]] | None = None

        # Motor de animación automática
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_progress: float = 0.0
        self._anim_speed_base: float = 0.012  # Velocidad suave de escritura continua
        self._speed_multiplier: float = 1.0   # 0.5x, 1.0x, 1.5x
        self._pause_ticks: int = 0
        self._pause_duration_ticks: int = 45  # ~1.0 s de pausa con la letra completa en pantalla
        self._show_anim: bool = True
        self._is_paused: bool = False
        self._current_stroke_idx: int = 0

        self.setStyleSheet("background-color: #ffffff; border-radius: 12px;")

    # --- Configuración de datos y control de animación -------------------------
    def set_glyph_data(
        self,
        letter: str,
        n_strokes: int,
        strokes: Sequence[Sequence[Sequence[float]]],
        start_points: Sequence[Sequence[float]],
    ) -> None:
        """Actualiza la letra a demostrar, limpia los trazos del usuario y reinicia la animación."""
        self._letter = letter
        self._n_strokes = max(1, n_strokes)
        self._strokes = [np.asarray(s, dtype=float) for s in strokes]
        self._start_points = [(float(p[0]), float(p[1])) for p in start_points]
        self.clear_user_drawing(restart_anim=True)

    def clear_user_drawing(self, restart_anim: bool = True) -> None:
        """Borra los trazos realizados por el usuario para volver a practicar."""
        self.user_strokes.clear()
        self._current_user_stroke = None
        if restart_anim:
            self.restart_animation()
        else:
            self.update()

    def restart_animation(self) -> None:
        """Reinicia el recorrido animado del lápiz demostrador desde el inicio."""
        self._show_anim = True
        self._is_paused = False
        self._anim_progress = 0.0
        self._pause_ticks = 0
        self._current_stroke_idx = 0
        if not self._anim_timer.isActive():
            self._anim_timer.start(20)  # ~50-60 FPS
        self.update()
        self.animation_state_changed.emit("Animación activa")

    def toggle_play_pause(self) -> bool:
        """Alterna entre pausar y reanudar la animación. Retorna True si quedó pausada."""
        self._is_paused = not self._is_paused
        if not self._is_paused and not self._anim_timer.isActive():
            self._anim_timer.start(20)
        self.update()
        state_text = "Pausada" if self._is_paused else "Animación activa"
        self.animation_state_changed.emit(state_text)
        return self._is_paused

    def set_speed_multiplier(self, mult: float) -> None:
        """Ajusta el multiplicador de velocidad (ej. 0.5, 1.0, 1.5)."""
        self._speed_multiplier = max(0.2, float(mult))

    def _on_anim_tick(self) -> None:
        if not self._show_anim or self._is_paused:
            return

        # Pausa cuando se completa la letra para que el usuario aprecie el resultado
        if self._pause_ticks > 0:
            self._pause_ticks -= 1
            if self._pause_ticks == 0:
                # Se reinicia el ciclo de animación suavemente
                self._anim_progress = 0.0
                self.animation_state_changed.emit("Animación activa")
                self.update()
            return

        self._anim_progress += self._anim_speed_base * self._speed_multiplier
        if self._anim_progress >= 1.0:
            self._anim_progress = 1.0
            self._pause_ticks = self._pause_duration_ticks
            self.animation_state_changed.emit("Letra completada. Repitiendo...")
            self.update()
        else:
            self.update()

    # --- Eventos de ratón para trazado libre opcional --------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            # Pausa la animación automática para permitir al usuario practicar a mano
            self._show_anim = False
            pos = event.position()
            self._current_user_stroke = [(pos.x(), pos.y())]
            self.user_strokes.append(self._current_user_stroke)
            self.update()
            self.stroke_drawn.emit()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._current_user_stroke is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self._current_user_stroke.append((pos.x(), pos.y()))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._current_user_stroke = None
            self.update()
            self.stroke_drawn.emit()

    # --- Renderizado con QPainter ----------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = float(self.width())
        h = float(self.height())

        # Fondo con tarjeta redondeada y sutil borde
        rect = QRectF(0, 0, w, h)
        painter.setPen(QPen(QColor("#e2e8f0"), 1.5))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 12, 12)

        # Renglón caligráfico escolar de 4 líneas horizontales
        pad_x = 36.0
        pad_y = 48.0
        box_w = w - 2 * pad_x
        box_h = h - 2 * pad_y

        pen_guide = QPen(QColor("#cbd5e1"), 1.0, Qt.PenStyle.DashLine)
        pen_base = QPen(QColor("#94a3b8"), 1.3, Qt.PenStyle.SolidLine)

        y_top = pad_y + 0.15 * box_h
        y_mid = pad_y + 0.40 * box_h
        y_base = pad_y + 0.72 * box_h
        y_desc = pad_y + 0.92 * box_h

        painter.setPen(pen_guide)
        painter.drawLine(QPointF(pad_x, y_top), QPointF(w - pad_x, y_top))
        painter.drawLine(QPointF(pad_x, y_mid), QPointF(w - pad_x, y_mid))
        painter.setPen(pen_base)
        painter.drawLine(QPointF(pad_x, y_base), QPointF(w - pad_x, y_base))
        painter.setPen(pen_guide)
        painter.drawLine(QPointF(pad_x, y_desc), QPointF(w - pad_x, y_desc))

        if not self._strokes:
            return

        box_side = min(box_w, box_h)
        cx = pad_x + 0.5 * (box_w - box_side)
        cy = pad_y + 0.5 * (box_h - box_side)

        def to_px(norm_xy: np.ndarray) -> np.ndarray:
            return np.column_stack([cx + norm_xy[:, 0] * box_side, cy + norm_xy[:, 1] * box_side])

        # 1. Dibujar traza fantasma completa en gris tenue (guía base de fondo)
        pen_ghost = QPen(
            QColor("#e2e8f0"), 7.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
        )
        painter.setPen(pen_ghost)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for s in self._strokes:
            if len(s) == 0:
                continue
            pts = to_px(s)
            if len(pts) == 1:
                painter.setBrush(QBrush(QColor("#e2e8f0")))
                painter.drawEllipse(QPointF(pts[0, 0], pts[0, 1]), 4.0, 4.0)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                path = QPainterPath()
                path.moveTo(pts[0, 0], pts[0, 1])
                for i in range(1, len(pts)):
                    path.lineTo(pts[i, 0], pts[i, 1])
                painter.drawPath(path)

        # 2. Líneas punteadas de desplazamiento entre trazos (levantamiento de lápiz en el aire)
        if len(self._strokes) > 1:
            pen_air = QPen(QColor("#94a3b8"), 1.2, Qt.PenStyle.DotLine)
            painter.setPen(pen_air)
            for i in range(len(self._strokes) - 1):
                s1 = self._strokes[i]
                s2 = self._strokes[i + 1]
                if len(s1) > 0 and len(s2) > 0:
                    p_end = to_px(s1[-1:])
                    p_next = to_px(s2[:1])
                    painter.drawLine(
                        QPointF(p_end[0, 0], p_end[0, 1]),
                        QPointF(p_next[0, 0], p_next[0, 1]),
                    )

        # 3. Animación automática: Dibujo progresivo en vivo + Lápiz animado
        active_stroke_idx = 0
        current_tip: QPointF | None = None
        current_dir: tuple[float, float] | None = None

        if self._show_anim and not self.user_strokes:
            # Trazo animado en verde esmeralda vibrante (#16a34a)
            pen_anim = QPen(
                QColor("#16a34a"), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
            )
            painter.setPen(pen_anim)

            lens = [
                float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))) if len(s) >= 2 else 0.05
                for s in self._strokes
            ]
            total_len = max(sum(lens), 1e-6)
            target_len = self._anim_progress * total_len

            acc_len = 0.0

            for s_idx, (s, slen) in enumerate(zip(self._strokes, lens)):
                if len(s) == 0:
                    continue
                pts = to_px(s)
                if len(pts) == 1:
                    if acc_len <= target_len:
                        painter.setBrush(QBrush(QColor("#16a34a")))
                        painter.drawEllipse(QPointF(pts[0, 0], pts[0, 1]), 3.5, 3.5)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        current_tip = QPointF(pts[0, 0], pts[0, 1])
                        active_stroke_idx = s_idx
                    acc_len += slen
                    continue

                path = QPainterPath()
                path.moveTo(pts[0, 0], pts[0, 1])

                if acc_len + slen <= target_len:
                    # Trazo completamente dibujado
                    for i in range(1, len(pts)):
                        path.lineTo(pts[i, 0], pts[i, 1])
                    painter.drawPath(path)
                    acc_len += slen
                    current_tip = QPointF(pts[-1, 0], pts[-1, 1])
                    active_stroke_idx = s_idx
                elif acc_len < target_len:
                    # Trazo actualmente en progreso de dibujo
                    rem = target_len - acc_len
                    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                    cum = np.r_[0.0, np.cumsum(segs)]
                    k = int(np.searchsorted(cum, rem, side="right")) - 1
                    k = min(max(k, 0), len(pts) - 2)
                    t = (rem - cum[k]) / max(segs[k], 1e-6)
                    tip = pts[k] + np.clip(t, 0.0, 1.0) * (pts[k + 1] - pts[k])
                    for i in range(1, k + 1):
                        path.lineTo(pts[i, 0], pts[i, 1])
                    path.lineTo(tip[0], tip[1])
                    painter.drawPath(path)
                    current_tip = QPointF(tip[0], tip[1])

                    # Dirección instantánea del lápiz
                    dx = float(pts[k + 1, 0] - pts[k, 0])
                    dy = float(pts[k + 1, 1] - pts[k, 1])
                    norm = math.hypot(dx, dy)
                    if norm > 1e-5:
                        current_dir = (dx / norm, dy / norm)

                    active_stroke_idx = s_idx
                    break
                else:
                    break

            # Dibujar el cabezal del lápiz demostrador con resplandor visual
            if current_tip is not None and self._anim_progress < 1.0:
                # Halo exterior luminoso suave
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(22, 163, 74, 55)))
                painter.drawEllipse(current_tip, 13.0, 13.0)

                # Anillo intermedio blanco protector
                painter.setPen(QPen(QColor("#ffffff"), 2.0))
                painter.setBrush(QBrush(QColor("#15803d")))
                painter.drawEllipse(current_tip, 6.5, 6.5)

                # Núcleo blanco brillante
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.drawEllipse(current_tip, 2.2, 2.2)

                # Pequeña flecha direccional apuntando en el sentido del trazo
                if current_dir is not None:
                    ux, uy = current_dir
                    arrow_dist = 11.0
                    arrow_tip = QPointF(current_tip.x() + ux * arrow_dist, current_tip.y() + uy * arrow_dist)
                    painter.setPen(QPen(QColor("#15803d"), 1.8, Qt.PenStyle.SolidLine))
                    painter.drawLine(current_tip, arrow_tip)

        # 4. Trazos realizados manualmente por el usuario con el ratón
        if self.user_strokes:
            pen_user = QPen(
                QColor("#2563eb"), 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
            )
            painter.setPen(pen_user)
            for stroke in self.user_strokes:
                if not stroke:
                    continue
                if len(stroke) == 1:
                    painter.setBrush(QBrush(QColor("#2563eb")))
                    painter.drawEllipse(QPointF(*stroke[0]), 3.5, 3.5)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                else:
                    path = QPainterPath()
                    path.moveTo(QPointF(*stroke[0]))
                    for pt in stroke[1:]:
                        path.lineTo(QPointF(*pt))
                    painter.drawPath(path)

        # 5. Círculos verdes numerados (1, 2, 3...) en el inicio de cada trazo
        for idx, sp in enumerate(self._start_points):
            sp_px = QPointF(cx + sp[0] * box_side, cy + sp[1] * box_side)

            # Si este trazo es el que está dibujándose en este momento, resaltar el badge
            is_active_badge = (idx == active_stroke_idx) and (self._anim_progress < 1.0) and self._show_anim

            if is_active_badge:
                # Anillo de pulso exterior
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(22, 163, 74, 60)))
                painter.drawEllipse(sp_px, 17.0, 17.0)

            radius = 11.5
            painter.setPen(QPen(QColor("#ffffff"), 2.0))
            painter.setBrush(QBrush(QColor("#15803d" if is_active_badge else "#16a34a")))
            painter.drawEllipse(sp_px, radius, radius)

            painter.setPen(QColor("#ffffff"))
            font = QFont("Segoe UI", 9, QFont.Weight.Bold)
            painter.setFont(font)
            text = str(idx + 1)
            rect_num = QRectF(sp_px.x() - radius, sp_px.y() - radius, 2 * radius, 2 * radius)
            painter.drawText(rect_num, Qt.AlignmentFlag.AlignCenter, text)


class LetterGuideView(QWidget):
    """Vista completa de 'Guía de letras' con animación automática continua,

    controles de reproducción (pausar, velocidad, reiniciar) y selector de 27 letras.
    """

    letter_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cfg = get_config()
        self.glyphs = _load_canonical_glyphs()
        self.current_letter: str = "a"
        self._buttons: dict[str, QPushButton] = {}
        self._speed_modes: list[tuple[str, float]] = [
            ("Velocidad: 1x", 1.0),
            ("Velocidad: 0.5x (Lento)", 0.5),
            ("Velocidad: 1.5x (Rápido)", 1.5),
        ]
        self._current_speed_idx: int = 0

        self._build_ui()
        self.select_letter("a")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        # Encabezado con título e instrucciones claras
        header_box = QVBoxLayout()
        header_box.setSpacing(4)
        title = QLabel("Guía de letras")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #1e293b;")
        subtitle = QLabel(
            "Animación automática que muestra cómo trazar cada letra. "
            "Sigue el punto numerado (1, 2) y la dirección del lápiz."
        )
        subtitle.setStyleSheet("font-size: 13px; color: #64748b;")
        header_box.addWidget(title)
        header_box.addWidget(subtitle)
        layout.addLayout(header_box)

        # Área principal dividida: izquierda animación y controles, derecha cuadrícula
        main_h = QHBoxLayout()
        main_h.setSpacing(24)

        # Columna Izquierda: Lienzo interactivo + barra de controles
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        self.demo_canvas = StrokeDemoCanvas(self)
        self.demo_canvas.stroke_drawn.connect(self._on_user_stroke_updated)
        self.demo_canvas.animation_state_changed.connect(self._on_anim_state_changed)
        left_col.addWidget(self.demo_canvas, stretch=1)

        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(8)

        self.lbl_info = QLabel("Letra a  —  1 trazo")
        self.lbl_info.setStyleSheet("font-size: 13px; font-weight: 600; color: #334155;")
        bottom_bar.addWidget(self.lbl_info)

        self.lbl_status = QLabel("Animación activa")
        self.lbl_status.setStyleSheet("font-size: 12px; color: #16a34a; font-weight: 500; margin-left: 8px;")
        bottom_bar.addWidget(self.lbl_status)

        bottom_bar.addStretch()

        # Botón Pausar / Reanudar animación
        self.btn_play_pause = QPushButton("⏸ Pausar")
        self.btn_play_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_play_pause.setStyleSheet(
            "QPushButton { background-color: #ffffff; color: #334155; border: 1px solid #cbd5e1; "
            "border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 500; } "
            "QPushButton:hover { background-color: #f1f5f9; }"
        )
        self.btn_play_pause.clicked.connect(self._on_play_pause_clicked)
        bottom_bar.addWidget(self.btn_play_pause)

        # Botón de Velocidad
        self.btn_speed = QPushButton("Velocidad: 1x")
        self.btn_speed.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_speed.setStyleSheet(
            "QPushButton { background-color: #ffffff; color: #334155; border: 1px solid #cbd5e1; "
            "border-radius: 6px; padding: 6px 10px; font-size: 12px; font-weight: 500; } "
            "QPushButton:hover { background-color: #f1f5f9; }"
        )
        self.btn_speed.clicked.connect(self._on_speed_clicked)
        bottom_bar.addWidget(self.btn_speed)

        # Botón Reiniciar animación
        self.btn_replay = QPushButton("Reiniciar")
        self.btn_replay.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_replay.setStyleSheet(
            "QPushButton { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; "
            "border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 600; } "
            "QPushButton:hover { background-color: #dcfce7; }"
        )
        self.btn_replay.clicked.connect(self.demo_canvas.restart_animation)
        self.btn_restart = self.btn_replay
        bottom_bar.addWidget(self.btn_replay)

        # Botón Limpiar trazo manual
        self.btn_clear = QPushButton("Limpiar trazo")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setStyleSheet(
            "QPushButton { background-color: #ffffff; color: #64748b; border: 1px solid #e2e8f0; "
            "border-radius: 6px; padding: 6px 10px; font-size: 12px; font-weight: 500; } "
            "QPushButton:hover { background-color: #f8fafc; color: #334155; }"
        )
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        bottom_bar.addWidget(self.btn_clear)

        left_col.addLayout(bottom_bar)
        main_h.addLayout(left_col, stretch=5)

        # Columna Derecha: Cuadrícula de 27 tarjetas de letras (5 columnas)
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        lbl_grid_title = QLabel("Selecciona una letra:")
        lbl_grid_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #475569;")
        right_col.addWidget(lbl_grid_title)

        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(8)
        grid.setContentsMargins(0, 0, 0, 0)

        # 27 letras en 5 columnas
        letters = list(self.cfg.letters)
        n_cols = 5
        for idx, letter in enumerate(letters):
            row = idx // n_cols
            col = idx % n_cols
            btn = QPushButton(letter)
            btn.setFixedSize(50, 50)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._btn_style(selected=False))
            btn.clicked.connect(lambda checked, l=letter: self.select_letter(l))
            grid.addWidget(btn, row, col)
            self._buttons[letter] = btn

        right_col.addWidget(grid_widget)
        right_col.addStretch()
        main_h.addLayout(right_col, stretch=4)

        layout.addLayout(main_h, stretch=1)

    # --- Acciones de control --------------------------------------------------
    def _on_play_pause_clicked(self) -> None:
        is_paused = self.demo_canvas.toggle_play_pause()
        self.btn_play_pause.setText("▶ Reanudar" if is_paused else "⏸ Pausar")

    def _on_speed_clicked(self) -> None:
        self._current_speed_idx = (self._current_speed_idx + 1) % len(self._speed_modes)
        label, mult = self._speed_modes[self._current_speed_idx]
        self.btn_speed.setText(label)
        self.demo_canvas.set_speed_multiplier(mult)

    def _on_anim_state_changed(self, state: str) -> None:
        self.lbl_status.setText(state)
        if "completada" in state.lower():
            self.lbl_status.setStyleSheet("font-size: 12px; color: #0284c7; font-weight: 500; margin-left: 8px;")
        elif "pausada" in state.lower():
            self.lbl_status.setStyleSheet("font-size: 12px; color: #d97706; font-weight: 500; margin-left: 8px;")
        else:
            self.lbl_status.setStyleSheet("font-size: 12px; color: #16a34a; font-weight: 500; margin-left: 8px;")

    def _on_user_stroke_updated(self) -> None:
        n = len(self.demo_canvas.user_strokes)
        if n > 0:
            plural = "s" if n > 1 else ""
            self.lbl_status.setText(f"Trazando a mano ({n} trazo{plural})...")
            self.lbl_status.setStyleSheet("font-size: 12px; font-weight: 600; color: #2563eb; margin-left: 8px;")

    def _on_clear_clicked(self) -> None:
        self.demo_canvas.clear_user_drawing(restart_anim=True)
        self.btn_play_pause.setText("⏸ Pausar")

    def _btn_style(self, selected: bool) -> str:
        if selected:
            return (
                "QPushButton { "
                "background-color: #f0fdf4; "
                "border: 2px solid #16a34a; "
                "border-radius: 8px; "
                "font-size: 18px; "
                "font-weight: 700; "
                "color: #15803d; "
                "}"
            )
        return (
            "QPushButton { "
            "background-color: #ffffff; "
            "border: 1px solid #e2e8f0; "
            "border-radius: 8px; "
            "font-size: 17px; "
            "font-weight: 600; "
            "color: #334155; "
            "} "
            "QPushButton:hover { "
            "background-color: #f8fafc; "
            "border-color: #cbd5e1; "
            "}"
        )

    def select_letter(self, letter: str) -> None:
        """Selecciona una letra, actualiza el resaltado visual e inicia la animación automática."""
        if self.current_letter in self._buttons:
            self._buttons[self.current_letter].setStyleSheet(self._btn_style(selected=False))

        self.current_letter = letter
        if letter in self._buttons:
            self._buttons[letter].setStyleSheet(self._btn_style(selected=True))

        g = self.glyphs.get(letter)
        if g is not None:
            n_st = int(g.get("n_strokes", 1))
            strokes = g.get("strokes", [])
            start_pts = g.get("start_points", [])
        else:
            n_st = 1
            strokes = []
            start_pts = []

        plural = "s" if n_st > 1 else ""
        self.lbl_info.setText(f"Letra {letter}  —  {n_st} trazo{plural}")
        self.btn_play_pause.setText("⏸ Pausar")
        self.demo_canvas.set_glyph_data(letter, n_st, strokes, start_pts)
        self.letter_selected.emit(letter)
