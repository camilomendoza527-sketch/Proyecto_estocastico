"""Panel lateral derecho de clasificación y probabilidades en tiempo real (gui/RankingPanel.py).

Diseño idéntico a la referencia visual (Image 2):
- Se eliminaron las cajas guía con puntos verdes.
- Cada letra se muestra como su propio glifo manuscrito (la forma exacta que se dibuja, no tipografía de teclado).
- Cada fila contiene: Glifo manuscrito -> Barra de progreso estilizada (píldora) -> Porcentaje.
- Fila ganadora resaltada con fondo verde suave (#e6f4ea) y barra/glifo en verde oscuro (#0f766e).
- Cabecera compacta: "Esperando trazo" o "Reconociendo: 17.1%".
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Sequence

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.config import PROJECT_ROOT, get_config


@lru_cache(maxsize=1)
def load_canonical_glyphs() -> dict[str, dict]:
    """Carga y almacena en caché los glifos canónicos de data/canonical_glyphs.json."""
    path = PROJECT_ROOT / "data" / "canonical_glyphs.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def build_handwritten_glyph_pixmap(letter: str, color: QColor, w: int = 24, h: int = 30) -> QPixmap:
    """Genera un glifo manuscrito transparente de la letra (como se dibuja, sin caja ni punto verde)."""
    pixmap = QPixmap(w, h)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    pad_x = 2.0
    pad_y = 3.0
    span_x = w - 2.0 * pad_x
    span_y = h - 2.0 * pad_y

    glyph_info = load_canonical_glyphs().get(letter)
    strokes = glyph_info.get("strokes", []) if glyph_info else []

    pen = QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    for stroke in strokes:
        if len(stroke) == 0:
            continue
        if len(stroke) == 1:
            # Punto aislado (ej. punto de la 'i' o 'j')
            pt = stroke[0]
            px = pad_x + float(pt[0]) * span_x
            py = pad_y + float(pt[1]) * span_y
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(px, py), 1.5, 1.5)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            continue

        path = QPainterPath()
        p0 = stroke[0]
        path.moveTo(pad_x + float(p0[0]) * span_x, pad_y + float(p0[1]) * span_y)
        for p in stroke[1:]:
            path.lineTo(pad_x + float(p[0]) * span_x, pad_y + float(p[1]) * span_y)
        painter.drawPath(path)

    painter.end()
    return pixmap


class _LetterRankRow(QFrame):
    """Fila individual: glifo manuscrito a la izquierda, barra al centro, porcentaje a la derecha."""

    def __init__(self, letter: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.letter = letter
        self.setFixedHeight(34)

        # Generar las dos versiones del glifo (normal en carbón oscuro, destacada en verde)
        self.pixmap_normal = build_handwritten_glyph_pixmap(letter, QColor("#1e293b"))
        self.pixmap_highlight = build_handwritten_glyph_pixmap(letter, QColor("#0f766e"))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(10)

        # Glifo manuscrito (la forma exacta que se dibuja)
        self.lbl_glyph = QLabel()
        self.lbl_glyph.setFixedSize(24, 30)
        self.lbl_glyph.setPixmap(self.pixmap_normal)
        self.lbl_glyph.setScaledContents(True)
        layout.addWidget(self.lbl_glyph)

        # Label de compatibilidad con tests (mantiene lbl_letter accesible para row.lbl_letter.text())
        self.lbl_letter = QLabel(letter)
        self.lbl_letter.setVisible(False)
        layout.addWidget(self.lbl_letter)

        # Barra de progreso tipo píldora
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(7)
        layout.addWidget(self.bar, stretch=1)

        # Porcentaje
        self.lbl_pct = QLabel("0.0%")
        self.lbl_pct.setFixedWidth(46)
        self.lbl_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_pct.setStyleSheet("font-size: 11px; font-weight: 500; color: #64748b;")
        layout.addWidget(self.lbl_pct)

        self._is_highlighted: bool | None = None
        self.set_highlight(False)

    def set_highlight(self, is_top: bool) -> None:
        if self._is_highlighted is is_top:
            return
        self._is_highlighted = is_top
        if is_top:
            self.setStyleSheet(
                "QFrame { background-color: #e6f4ea; border-radius: 6px; border: none; }"
            )
            self.lbl_glyph.setPixmap(self.pixmap_highlight)
            self.lbl_pct.setStyleSheet("font-size: 11px; font-weight: 700; color: #0f766e;")
            self.bar.setStyleSheet(
                "QProgressBar { background-color: #c7ebd1; border-radius: 3.5px; border: none; } "
                "QProgressBar::chunk { background-color: #0f766e; border-radius: 3.5px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { background-color: transparent; border: none; }"
            )
            self.lbl_glyph.setPixmap(self.pixmap_normal)
            self.lbl_pct.setStyleSheet("font-size: 11px; font-weight: 500; color: #64748b;")
            self.bar.setStyleSheet(
                "QProgressBar { background-color: #e2e8f0; border-radius: 3.5px; border: none; } "
                "QProgressBar::chunk { background-color: #94a3b8; border-radius: 3.5px; }"
            )


class RankingPanel(QFrame):
    """Panel lateral derecho de clasificación y probabilidades en tiempo real."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cfg = get_config()
        self.setFixedWidth(280)
        self.setStyleSheet(
            "RankingPanel { background-color: #ffffff; border-left: 1px solid #e2e8f0; }"
        )

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 16, 14, 16)
        root_layout.setSpacing(8)

        # Encabezado minimalista: "Esperando trazo" o "Reconociendo: 17.1%"
        self.lbl_top_val = QLabel("Esperando trazo")
        self.lbl_top_val.setStyleSheet("font-size: 14px; font-weight: 700; color: #1e293b; padding-left: 4px;")
        root_layout.addWidget(self.lbl_top_val)

        # Label de compatibilidad para verificar el carácter ganador
        self.lbl_top_char = QLabel("—")
        self.lbl_top_char.setVisible(False)
        root_layout.addWidget(self.lbl_top_char)

        # Área de desplazamiento con las 27 filas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self.rows_layout = QVBoxLayout(container)
        self.rows_layout.setContentsMargins(0, 0, 2, 0)
        self.rows_layout.setSpacing(2)

        self.rows: list[_LetterRankRow] = []
        self._rows_by_char: dict[str, _LetterRankRow] = {}
        for ch in self.cfg.letters:
            row = _LetterRankRow(ch)
            self.rows.append(row)
            self._rows_by_char[ch] = row
            self.rows_layout.addWidget(row)
        self.rows_layout.addStretch(1)

        scroll.setWidget(container)
        root_layout.addWidget(scroll, stretch=1)

    def reset(self) -> None:
        """Restablece todas las probabilidades a 0.0% y el texto a 'Esperando trazo'."""
        self.lbl_top_char.setText("—")
        self.lbl_top_val.setText("Esperando trazo")
        self.lbl_top_val.setStyleSheet("font-size: 14px; font-weight: 700; color: #1e293b; padding-left: 4px;")
        for row in self.rows:
            row.bar.setValue(0)
            row.lbl_pct.setText("0.0%")
            row.set_highlight(False)

    def update_probabilities(
        self,
        ranking: Sequence[tuple],
        viterbi_state: int | None = None,
        n_obs: int = 0,
    ) -> None:
        """Actualiza las probabilidades y reordena la vista mostrando el ganador."""
        if not ranking:
            self.reset()
            return

        top_char = ranking[0][0]
        top_prob = float(ranking[0][1])
        pct = top_prob * 100.0

        if n_obs > 0 and top_prob > 0.05:
            self.lbl_top_char.setText(top_char)
            self.lbl_top_val.setText(f"Reconociendo: {pct:.1f}%")
            self.lbl_top_val.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f766e; padding-left: 4px;")
        else:
            self.lbl_top_char.setText("—")
            self.lbl_top_val.setText("Esperando trazo")
            self.lbl_top_val.setStyleSheet("font-size: 14px; font-weight: 700; color: #1e293b; padding-left: 4px;")

        # Reordenar las filas en el layout según el ranking (desactivar repintados intermedios)
        self.setUpdatesEnabled(False)
        try:
            for rank_idx, item in enumerate(ranking):
                letter = item[0]
                prob = float(item[1])
                row = self._rows_by_char.get(letter)
                if row is not None:
                    val = int(round(prob * 1000))
                    row.bar.setValue(val)
                    row.lbl_pct.setText(f"{prob * 100.0:4.1f}%")
                    is_top = (rank_idx == 0 and n_obs > 0 and prob > 0.05)
                    row.set_highlight(is_top)
                    self.rows_layout.insertWidget(rank_idx, row)
        finally:
            self.setUpdatesEnabled(True)
