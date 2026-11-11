"""Guía interactiva y visual de trazado caligráfico de letras (gui/LetterGuide.py).

Permite al usuario:
- Ver la animación automática guiada a velocidad natural de escritura humana (~2.5 a 3.5 segundos por letra).
- Observar el trazado progresivo continuo a 60 FPS con renglón escolar, punta de lápiz iluminada y flecha direccional.
- Observar el desplazamiento en el aire (pen-up) entre trazos para letras compuestas (i, j, ñ, t, x).
- Disfrutar de una pausa de apreciación de 1.5s con la letra terminada antes de reiniciar el bucle.
- Controlar la reproducción: Pausar / Reanudar, cambiar Velocidad (1x Normal, 0.5x Lento, 1.5x Rápido) o Reiniciar.
- Practicar libremente con el ratón a mano alzada sobre el lienzo cuando lo desee.
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
    """Lienzo que reproduce la animación caligráfica a velocidad promedio de escritura humana,

    con punta de lápiz visual, líneas de vuelo aéreo y soporte de práctica manual con el ratón.
    """

    stroke_drawn = Signal()
    animation_state_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._letter: str = "a"
        self._n_strokes: int = 1
        self._strokes: list[np.ndarray] = []
        self._start_points: list[tuple[float, float]] = []

        # Trazos realizados manualmente por el usuario
        self.user_strokes: list[list[tuple[float, float]]] = []
        self._current_user_stroke: list[tuple[float, float]] | None = None

        # Motor de tiempo paramétrico continuo
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._elapsed_time: float = 0.0      # Tiempo acumulado en segundos dentro del ciclo
        self._speed_multiplier: float = 1.0  # 1.0x (Normal), 0.5x (Lento), 1.5x (Rápido)
        self._show_anim: bool = True
        self._is_paused: bool = False

        # Parámetros del cronograma (timeline)
        self._timeline_phases: list[dict] = []
        self._total_cycle_time: float = 4.0  # Duración total del ciclo (escritura + vuelos + pausa)
        self._pause_hold_time: float = 1.5   # Pausa de apreciación al terminar la letra

        self.setStyleSheet("background-color: #ffffff; border-radius: 12px;")

    # --- Construcción del cronograma a velocidad humana -----------------------
    def _build_timeline(self) -> None:
        """Construye un cronograma realista basado en la longitud y complejidad de la letra.

        Tiempo promedio de escritura:
        - 1 trazo: ~2.4 a 2.8 segundos de trazado continuo.
        - 2 trazos (i, j, t, x): ~2.2s trazo principal + ~0.45s vuelo + ~0.7s segundo trazo (~3.35s).
        - 3 trazos (ñ): ~2.2s cuerpo n + ~0.45s vuelo + ~0.8s virgulilla (~3.45s).
        - Pausa final con letra completa: 1.5 segundos antes de reiniciar el bucle.
        """
        self._timeline_phases.clear()
        if not self._strokes:
            self._total_cycle_time = 1.0
            return

        # Calcular longitud espacial de cada trazo
        lengths = []
        for s in self._strokes:
            if len(s) >= 2:
                slen = float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)))
                lengths.append(max(slen, 0.08))
            else:
                lengths.append(0.18)  # Punto aislado (ej. punto de la 'i' o 'j')

        total_len = max(sum(lengths), 1e-4)

        # Duración total de tinta en segundos (promedio caligráfico humano)
        base_ink_time = 2.4 + 0.35 * max(0, len(self._strokes) - 1)
        flight_duration = 0.45  # Tiempo que tarda la mano en levantarse y desplazarse

        current_t = 0.0
        for i, slen in enumerate(lengths):
            # Duración proporcional a la longitud geométrica del trazo
            stroke_dur = max(0.55, base_ink_time * (slen / total_len))
            self._timeline_phases.append({
                "type": "stroke",
                "stroke_idx": i,
                "t_start": current_t,
                "t_end": current_t + stroke_dur,
                "duration": stroke_dur,
                "length": slen,
            })
            current_t += stroke_dur

            # Si hay un siguiente trazo, agregar fase de vuelo aéreo (pen-up)
            if i < len(self._strokes) - 1:
                self._timeline_phases.append({
                    "type": "flight",
                    "from_idx": i,
                    "to_idx": i + 1,
                    "t_start": current_t,
                    "t_end": current_t + flight_duration,
                    "duration": flight_duration,
                })
                current_t += flight_duration

        # Fase final de apreciación: letra completada en reposo
        self._timeline_phases.append({
            "type": "pause",
            "t_start": current_t,
            "t_end": current_t + self._pause_hold_time,
            "duration": self._pause_hold_time,
        })
        current_t += self._pause_hold_time

        self._total_cycle_time = current_t

    def set_glyph_data(
        self,
        letter: str,
        n_strokes: int,
        strokes: Sequence[Sequence[Sequence[float]]],
        start_points: Sequence[Sequence[float]],
    ) -> None:
        """Actualiza la letra a demostrar, recalcula el cronograma y reinicia."""
        self._letter = letter
        self._n_strokes = max(1, n_strokes)
        self._strokes = [np.asarray(s, dtype=float) for s in strokes]
        self._start_points = [(float(p[0]), float(p[1])) for p in start_points]
        self._build_timeline()
        self.clear_user_drawing(restart_anim=True)

    def clear_user_drawing(self, restart_anim: bool = True) -> None:
        """Borra los trazos del usuario para volver a practicar."""
        self.user_strokes.clear()
        self._current_user_stroke = None
        if restart_anim:
            self.restart_animation()
        else:
            self.update()

    def restart_animation(self) -> None:
        """Reinicia la animación desde el segundo cero a velocidad normal."""
        self._show_anim = True
        self._is_paused = False
        self._elapsed_time = 0.0
        if not self._anim_timer.isActive():
            self._anim_timer.start(20)  # ~50 FPS
        self.update()
        self.animation_state_changed.emit("Iniciando trazo guiado...")

    def toggle_play_pause(self) -> bool:
        """Alterna entre pausar y reanudar la animación."""
        self._is_paused = not self._is_paused
        if not self._is_paused and not self._anim_timer.isActive():
            self._anim_timer.start(20)
        self.update()
        state_text = "Pausada" if self._is_paused else "Trazando en vivo..."
        self.animation_state_changed.emit(state_text)
        return self._is_paused

    def set_speed_multiplier(self, mult: float) -> None:
        """Ajusta el multiplicador de velocidad (1.0 = normal ~2.8s, 0.5 = lento ~5.5s, 1.5 = rápido)."""
        self._speed_multiplier = max(0.2, float(mult))

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._show_anim and not self._is_paused and not self._anim_timer.isActive():
            self._anim_timer.start(20)

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        if self._anim_timer.isActive():
            self._anim_timer.stop()

    def _on_anim_tick(self) -> None:
        if not self._show_anim or self._is_paused:
            return

        dt = 0.020 * self._speed_multiplier
        self._elapsed_time += dt

        if self._elapsed_time >= self._total_cycle_time:
            self._elapsed_time = 0.0  # Reiniciar bucle suavemente

        self.update()

    # --- Eventos de ratón para dibujo a mano alzada opcional --------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
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

    # --- Renderizado de alta definición con QPainter ---------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = float(self.width())
        h = float(self.height())

        # Fondo con tarjeta redondeada
        rect = QRectF(0, 0, w, h)
        painter.setPen(QPen(QColor("#e2e8f0"), 1.5))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 12, 12)

        # Renglón caligráfico escolar
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

        # 1. Traza fantasma en gris tenue (guía base de fondo)
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

        # 2. Líneas punteadas de desplazamiento entre trazos (vuelo en el aire)
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

        # 3. Animación guiada en tiempo real según el cronograma
        active_badge_idx = 0
        current_tip: QPointF | None = None
        current_dir: tuple[float, float] | None = None
        is_pen_down = True

        if self._show_anim and not self.user_strokes and self._timeline_phases:
            # Identificar en qué fase nos encontramos
            curr_phase = self._timeline_phases[-1]
            for ph in self._timeline_phases:
                if ph["t_start"] <= self._elapsed_time < ph["t_end"]:
                    curr_phase = ph
                    break

            # Pluma para el trazo en verde esmeralda cálido
            pen_ink = QPen(
                QColor("#16a34a"), 5.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
            )
            painter.setPen(pen_ink)

            if curr_phase["type"] == "stroke":
                target_idx = curr_phase["stroke_idx"]
                active_badge_idx = target_idx
                is_pen_down = True

                # Dibujar los trazos anteriores completos
                for prev_i in range(target_idx):
                    pts = to_px(self._strokes[prev_i])
                    if len(pts) == 1:
                        painter.setBrush(QBrush(QColor("#16a34a")))
                        painter.drawEllipse(QPointF(pts[0, 0], pts[0, 1]), 3.5, 3.5)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                    else:
                        p = QPainterPath()
                        p.moveTo(pts[0, 0], pts[0, 1])
                        for k in range(1, len(pts)):
                            p.lineTo(pts[k, 0], pts[k, 1])
                        painter.drawPath(p)

                # Dibujar el trazo actual en progreso continuo
                curr_pts = to_px(self._strokes[target_idx])
                tau = np.clip((self._elapsed_time - curr_phase["t_start"]) / curr_phase["duration"], 0.0, 1.0)

                if len(curr_pts) == 1:
                    painter.setBrush(QBrush(QColor("#16a34a")))
                    painter.drawEllipse(QPointF(curr_pts[0, 0], curr_pts[0, 1]), 3.5, 3.5)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    current_tip = QPointF(curr_pts[0, 0], curr_pts[0, 1])
                else:
                    segs = np.linalg.norm(np.diff(curr_pts, axis=0), axis=1)
                    cum = np.r_[0.0, np.cumsum(segs)]
                    target_dist = tau * cum[-1]

                    idx = int(np.searchsorted(cum, target_dist, side="right")) - 1
                    idx = min(max(idx, 0), len(curr_pts) - 2)
                    frac = (target_dist - cum[idx]) / max(segs[idx], 1e-6)
                    interp_pt = curr_pts[idx] + np.clip(frac, 0.0, 1.0) * (curr_pts[idx + 1] - curr_pts[idx])

                    p = QPainterPath()
                    p.moveTo(curr_pts[0, 0], curr_pts[0, 1])
                    for k in range(1, idx + 1):
                        p.lineTo(curr_pts[k, 0], curr_pts[k, 1])
                    p.lineTo(interp_pt[0], interp_pt[1])
                    painter.drawPath(p)

                    current_tip = QPointF(interp_pt[0], interp_pt[1])
                    dx = float(curr_pts[idx + 1, 0] - curr_pts[idx, 0])
                    dy = float(curr_pts[idx + 1, 1] - curr_pts[idx, 1])
                    norm = math.hypot(dx, dy)
                    if norm > 1e-5:
                        current_dir = (dx / norm, dy / norm)

            elif curr_phase["type"] == "flight":
                # Lápiz levantado desplazándose hacia el siguiente punto de inicio
                from_idx = curr_phase["from_idx"]
                to_idx = curr_phase["to_idx"]
                active_badge_idx = to_idx
                is_pen_down = False

                # Todos los trazos hasta from_idx están completos
                for prev_i in range(to_idx):
                    pts = to_px(self._strokes[prev_i])
                    if len(pts) == 1:
                        painter.setBrush(QBrush(QColor("#16a34a")))
                        painter.drawEllipse(QPointF(pts[0, 0], pts[0, 1]), 3.5, 3.5)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                    else:
                        p = QPainterPath()
                        p.moveTo(pts[0, 0], pts[0, 1])
                        for k in range(1, len(pts)):
                            p.lineTo(pts[k, 0], pts[k, 1])
                        painter.drawPath(p)

                # Interpolar posición aérea del lápiz
                p_from = to_px(self._strokes[from_idx][-1:])
                p_to = to_px(self._strokes[to_idx][:1])
                tau = np.clip((self._elapsed_time - curr_phase["t_start"]) / curr_phase["duration"], 0.0, 1.0)
                # Curva suave ease-in-out
                tau_smooth = 0.5 - 0.5 * math.cos(math.pi * tau)
                pos_x = p_from[0, 0] + tau_smooth * (p_to[0, 0] - p_from[0, 0])
                pos_y = p_from[0, 1] + tau_smooth * (p_to[0, 1] - p_from[0, 1])
                current_tip = QPointF(pos_x, pos_y)

                dx = float(p_to[0, 0] - p_from[0, 0])
                dy = float(p_to[0, 1] - p_from[0, 1])
                norm = math.hypot(dx, dy)
                if norm > 1e-5:
                    current_dir = (dx / norm, dy / norm)

            elif curr_phase["type"] == "pause":
                # Letra 100% completada; contemplación estática
                active_badge_idx = -1
                current_tip = None
                for s in self._strokes:
                    pts = to_px(s)
                    if len(pts) == 1:
                        painter.setBrush(QBrush(QColor("#16a34a")))
                        painter.drawEllipse(QPointF(pts[0, 0], pts[0, 1]), 3.5, 3.5)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                    else:
                        p = QPainterPath()
                        p.moveTo(pts[0, 0], pts[0, 1])
                        for k in range(1, len(pts)):
                            p.lineTo(pts[k, 0], pts[k, 1])
                        painter.drawPath(p)

            # Dibujar el puntero del lápiz (estilo visual destacado)
            if current_tip is not None:
                if is_pen_down:
                    # Trazo activo en papel: halo verde vivo y núcleo protector
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(22, 163, 74, 55)))
                    painter.drawEllipse(current_tip, 13.0, 13.0)

                    painter.setPen(QPen(QColor("#ffffff"), 2.0))
                    painter.setBrush(QBrush(QColor("#15803d")))
                    painter.drawEllipse(current_tip, 6.5, 6.5)

                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor("#ffffff")))
                    painter.drawEllipse(current_tip, 2.2, 2.2)

                    # Flecha direccional
                    if current_dir is not None:
                        ux, uy = current_dir
                        arrow_tip = QPointF(current_tip.x() + ux * 12.0, current_tip.y() + uy * 12.0)
                        painter.setPen(QPen(QColor("#15803d"), 1.8, Qt.PenStyle.SolidLine))
                        painter.drawLine(current_tip, arrow_tip)
                else:
                    # Lápiz en el aire: halo azulado tenue
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(37, 99, 235, 45)))
                    painter.drawEllipse(current_tip, 11.0, 11.0)

                    painter.setPen(QPen(QColor("#ffffff"), 1.8))
                    painter.setBrush(QBrush(QColor("#2563eb")))
                    painter.drawEllipse(current_tip, 5.5, 5.5)

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

        # 5. Círculos numerados (1, 2, 3...) en el punto de inicio de cada trazo
        for idx, sp in enumerate(self._start_points):
            sp_px = QPointF(cx + sp[0] * box_side, cy + sp[1] * box_side)

            is_active_badge = (idx == active_badge_idx) and self._show_anim

            if is_active_badge:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(22, 163, 74, 60)))
                painter.drawEllipse(sp_px, 17.5, 17.5)

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
    """Vista completa de 'Guía de letras' con animación a velocidad de escritura natural,

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
            ("Velocidad: 1x (Normal ~2.8s)", 1.0),
            ("Velocidad: 0.5x (Lento ~5.5s)", 0.5),
            ("Velocidad: 1.5x (Rápido ~1.8s)", 1.5),
        ]
        self._current_speed_idx: int = 0

        self._build_ui()
        self.select_letter("a")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        # Encabezado
        header_box = QVBoxLayout()
        header_box.setSpacing(4)
        title = QLabel("Guía de letras")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #1e293b;")
        subtitle = QLabel(
            "Animación a velocidad de escritura humana. "
            "Sigue el punto numerado (1, 2) y el recorrido del lápiz sobre el renglón."
        )
        subtitle.setStyleSheet("font-size: 13px; color: #64748b;")
        header_box.addWidget(title)
        header_box.addWidget(subtitle)
        layout.addLayout(header_box)

        # Área principal dividida
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

        self.lbl_status = QLabel("Trazando en vivo...")
        self.lbl_status.setStyleSheet("font-size: 12px; color: #16a34a; font-weight: 500; margin-left: 8px;")
        bottom_bar.addWidget(self.lbl_status)

        bottom_bar.addStretch()

        # Botón Pausar / Reanudar
        self.btn_play_pause = QPushButton("⏸ Pausar")
        self.btn_play_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_play_pause.setStyleSheet(
            "QPushButton { background-color: #ffffff; color: #334155; border: 1px solid #cbd5e1; "
            "border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 500; } "
            "QPushButton:hover { background-color: #f1f5f9; }"
        )
        self.btn_play_pause.clicked.connect(self._on_play_pause_clicked)
        bottom_bar.addWidget(self.btn_play_pause)

        # Botón Velocidad
        self.btn_speed = QPushButton(self._speed_modes[0][0])
        self.btn_speed.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_speed.setStyleSheet(
            "QPushButton { background-color: #ffffff; color: #334155; border: 1px solid #cbd5e1; "
            "border-radius: 6px; padding: 6px 10px; font-size: 12px; font-weight: 500; } "
            "QPushButton:hover { background-color: #f1f5f9; }"
        )
        self.btn_speed.clicked.connect(self._on_speed_clicked)
        bottom_bar.addWidget(self.btn_speed)

        # Botón Reiniciar
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

        # Columna Derecha: Cuadrícula de 27 tarjetas
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        lbl_grid_title = QLabel("Selecciona una letra:")
        lbl_grid_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #475569;")
        right_col.addWidget(lbl_grid_title)

        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(8)
        grid.setContentsMargins(0, 0, 0, 0)

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
        elif "levantando" in state.lower():
            self.lbl_status.setStyleSheet("font-size: 12px; color: #2563eb; font-weight: 500; margin-left: 8px;")
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
        """Selecciona una letra, actualiza el resaltado visual y recalcula la animación."""
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
