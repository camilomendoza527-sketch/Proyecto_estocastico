"""Pantalla de Inicio (gui/MainMenu.py) — Vista 'Elige un modo'.

Diseño minimalista y limpio acorde a la Imagen 1:
- Título principal: 'Elige un modo'
- Subtítulo: 'Dibuja una letra y el modelo oculto de Markov la reconoce mientras la trazas.'
- Dos tarjetas interactivas grandes:
  1. Modo 2D (Plano cartesiano con icono de recuadro y letra 'a').
  2. Modo 3D (Superficie esférica con icono de esfera en malla).
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _render_2d_icon(size: int = 72) -> QPixmap:
    """Dibuja el icono de Modo 2D: un marco cuadrado con la letra 'a'."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin = 8.0
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    painter.setPen(QPen(QColor("#94a3b8"), 1.8, Qt.PenStyle.DashLine))
    painter.setBrush(QBrush(QColor("#f8fafc")))
    painter.drawRoundedRect(rect, 8, 8)

    painter.setPen(QColor("#1e293b"))
    font = QFont("Georgia", int(size * 0.38), QFont.Weight.Normal)
    painter.setFont(font)
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "a")
    painter.end()
    return pix


def _render_3d_icon(size: int = 72) -> QPixmap:
    """Dibuja el icono de Modo 3D: esfera en malla alámbrica."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    r = size * 0.38
    c = size * 0.5

    # Círculo exterior
    painter.setPen(QPen(QColor("#1e293b"), 1.8))
    painter.setBrush(QBrush(QColor("#f8fafc")))
    painter.drawEllipse(QPointF(c, c), r, r)

    # Meridianos y paralelos
    painter.setPen(QPen(QColor("#94a3b8"), 1.2))
    painter.drawEllipse(QPointF(c, c), r * 0.5, r)
    painter.drawEllipse(QPointF(c, c), r, r * 0.4)
    painter.drawLine(QPointF(c - r, c), QPointF(c + r, c))
    painter.drawLine(QPointF(c, c - r), QPointF(c, c + r))
    painter.end()
    return pix


class _ModeCard(QFrame):
    """Tarjeta interactiva grande para seleccionar un modo."""

    def __init__(
        self,
        title: str,
        subtitle: str,
        icon_pixmap: QPixmap,
        on_click: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(240)

        self._normal_style = (
            "QFrame { "
            "background-color: #ffffff; "
            "border: 1.5px solid #e2e8f0; "
            "border-radius: 12px; "
            "} "
            "QFrame:hover { "
            "border-color: #16a34a; "
            "background-color: #f8fafc; "
            "}"
        )
        self.setStyleSheet(self._normal_style)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(14)

        # Icono
        lbl_icon = QLabel()
        lbl_icon.setPixmap(icon_pixmap)
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_icon.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(lbl_icon)

        # Título
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #1e293b; background: transparent; border: none;"
        )
        layout.addWidget(lbl_title)

        # Subtítulo
        lbl_sub = QLabel(subtitle)
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet(
            "font-size: 13px; color: #64748b; background: transparent; border: none;"
        )
        layout.addWidget(lbl_sub)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click()
        super().mousePressEvent(event)


class MainMenu(QWidget):
    """Pantalla 'Inicio' de bienvenida y selección de modo."""

    def __init__(
        self,
        on_select_2d: Callable[[], None],
        on_select_3d: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 48, 40, 48)
        outer.setSpacing(32)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Encabezado
        header_box = QVBoxLayout()
        header_box.setSpacing(6)

        title = QLabel("Elige un modo")
        title.setStyleSheet("font-size: 26px; font-weight: 700; color: #1e293b;")

        subtitle = QLabel(
            "Dibuja una letra y el modelo oculto de Markov la reconoce mientras la trazas."
        )
        subtitle.setStyleSheet("font-size: 14px; color: #64748b;")

        header_box.addWidget(title)
        header_box.addWidget(subtitle)
        outer.addLayout(header_box)

        # Fila de tarjetas grandes
        cards_row = QHBoxLayout()
        cards_row.setSpacing(24)

        card_2d = _ModeCard(
            title="Modo 2D",
            subtitle="Plano cartesiano",
            icon_pixmap=_render_2d_icon(80),
            on_click=on_select_2d,
        )
        card_3d = _ModeCard(
            title="Modo 3D",
            subtitle="Superficie esférica",
            icon_pixmap=_render_3d_icon(80),
            on_click=on_select_3d,
        )

        cards_row.addWidget(card_2d)
        cards_row.addWidget(card_3d)
        outer.addLayout(cards_row)

        outer.addStretch(1)
