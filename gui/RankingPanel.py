"""Panel lateral derecho de clasificación y probabilidades en tiempo real (gui/RankingPanel.py).

Diseño limpio y moderno acorde a los mockups:
- Cabecera compacta: "Letra más probable" -> "Esperando trazo" / Letra reconocida (%).
- Etiqueta "P(c | O1:t) de cada letra".
- Listado de las 27 letras: imagen/glifo de referencia a la izquierda, barra de progreso estilizada y porcentaje a la derecha.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
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
from training.augment import canonical_unit_strokes


def _load_canonical_data() -> dict[str, dict]:
    path = PROJECT_ROOT / "data" / "canonical_glyphs.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_GLYPH_CACHE = _load_canonical_data()


def build_letter_reference_pixmap(letter: str, size: int = 34) -> QPixmap:
    """Genera un icono estático de referencia mostrando cómo debe dibujarse la letra."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Tarjeta base blanca con borde suave
    painter.setPen(QPen(QColor("#e2e8f0"), 1.0))
    painter.setBrush(QBrush(QColor("#f8fafc")))
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 6, 6)

    margin = 5.0
    span = size - 2.0 * margin

    glyph_info = _GLYPH_CACHE.get(letter)
    if glyph_info and "strokes" in glyph_info:
        strokes = glyph_info["strokes"]
        start_pts = glyph_info.get("start_points", [])
    else:
        try:
            strokes = canonical_unit_strokes(letter)
            start_pts = [[float(s[0][0]), float(s[0][1])] for s in strokes if len(s)]
        except Exception:
            strokes = []
            start_pts = []

    pen = QPen(QColor("#1e293b"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        path = QPainterPath()
        p0 = stroke[0]
        path.moveTo(margin + float(p0[0]) * span, margin + float(p0[1]) * span)
        for p in stroke[1:]:
            path.lineTo(margin + float(p[0]) * span, margin + float(p[1]) * span)
        painter.drawPath(path)

    # Punto verde en el inicio del trazo principal
    if start_pts:
        sp = start_pts[0]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#16a34a")))
        painter.drawEllipse(QPointF(margin + float(sp[0]) * span, margin + float(sp[1]) * span), 2.5, 2.5)

    painter.end()
    return pixmap


class _LetterRankRow(QFrame):
    """Fila individual de la lista: glifo a la izquierda, letra, barra al centro, porcentaje a la derecha."""

    def __init__(self, letter: str, icon_pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.letter = letter
        self.setFixedHeight(38)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(8)

        # Icono con la imagen de cómo trazar la letra
        self.lbl_icon = QLabel()
        self.lbl_icon.setFixedSize(30, 30)
        self.lbl_icon.setPixmap(icon_pixmap)
        layout.addWidget(self.lbl_icon)

        # Letra en texto limpio
        self.lbl_letter = QLabel(letter)
        self.lbl_letter.setFixedWidth(16)
        self.lbl_letter.setStyleSheet("font-size: 13px; font-weight: 600; color: #334155;")
        layout.addWidget(self.lbl_letter)

        # Barra de progreso estilizada
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        layout.addWidget(self.bar, stretch=1)

        # Porcentaje
        self.lbl_pct = QLabel("0.0%")
        self.lbl_pct.setFixedWidth(50)
        self.lbl_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_pct.setStyleSheet("font-size: 11px; font-weight: 500; color: #64748b;")
        layout.addWidget(self.lbl_pct)

        self.set_highlight(False)

    def set_highlight(self, is_top: bool) -> None:
        if is_top:
            self.setStyleSheet(
                "QFrame { background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; }"
            )
            self.lbl_letter.setStyleSheet("font-size: 13px; font-weight: 700; color: #15803d;")
            self.lbl_pct.setStyleSheet("font-size: 11px; font-weight: 700; color: #15803d;")
            self.bar.setStyleSheet(
                "QProgressBar { background-color: #dcfce7; border-radius: 4px; border: none; } "
                "QProgressBar::chunk { background-color: #16a34a; border-radius: 4px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { background-color: #ffffff; border: 1px solid #f1f5f9; border-radius: 6px; }"
            )
            self.lbl_letter.setStyleSheet("font-size: 13px; font-weight: 600; color: #334155;")
            self.lbl_pct.setStyleSheet("font-size: 11px; font-weight: 500; color: #64748b;")
            self.bar.setStyleSheet(
                "QProgressBar { background-color: #f1f5f9; border-radius: 4px; border: none; } "
                "QProgressBar::chunk { background-color: #94a3b8; border-radius: 4px; }"
            )


class RankingPanel(QFrame):
    """Panel lateral derecho de clasificación y probabilidades en tiempo real."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cfg = get_config()
        self.setFixedWidth(310)
        self.setStyleSheet(
            "RankingPanel { background-color: #ffffff; border-left: 1px solid #e2e8f0; }"
        )

        global _GLYPH_CACHE
        if not _GLYPH_CACHE:
            _GLYPH_CACHE = _load_canonical_data()

        self.icons: dict[str, QPixmap] = {
            ch: build_letter_reference_pixmap(ch, size=30) for ch in self.cfg.letters
        }

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(10)

        # Encabezado compacto
        self.lbl_top_hdr = QLabel("Letra más probable")
        self.lbl_top_hdr.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
        root_layout.addWidget(self.lbl_top_hdr)

        self.lbl_top_val = QLabel("Esperando trazo")
        self.lbl_top_val.setStyleSheet("font-size: 17px; font-weight: 700; color: #1e293b;")
        root_layout.addWidget(self.lbl_top_val)

        # Label de compatibilidad para tests
        self.lbl_top_char = QLabel("—")
        self.lbl_top_char.setVisible(False)
        root_layout.addWidget(self.lbl_top_char)

        self.lbl_sub = QLabel("P(c | O1:t) de cada letra")
        self.lbl_sub.setStyleSheet("font-size: 11px; color: #94a3b8; margin-top: 4px;")
        root_layout.addWidget(self.lbl_sub)

        # Área de desplazamiento con las 27 filas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self.rows_layout = QVBoxLayout(container)
        self.rows_layout.setContentsMargins(0, 0, 4, 0)
        self.rows_layout.setSpacing(3)

        self.rows: list[_LetterRankRow] = []
        self._rows_by_char: dict[str, _LetterRankRow] = {}
        for ch in self.cfg.letters:
            row = _LetterRankRow(ch, self.icons[ch])
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
        self.lbl_top_val.setStyleSheet("font-size: 17px; font-weight: 700; color: #1e293b;")
        for row in self.rows:
            row.bar.setValue(0)
            row.lbl_pct.setText("0.0%")
            row.set_highlight(False)

    def reset_ranking(self) -> None:
        self.reset()

    def update_ranking(
        self,
        ranking: Sequence[tuple[str, float]],
        top_letter: str | None = None,
        current_state: int | None = None,
        n_states: int | None = None,
        n_obs: int = 0,
    ) -> None:
        self.update_probabilities(ranking, current_state, n_obs)

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
            self.lbl_top_val.setText(f"{top_char}  ({pct:.1f}%)")
            self.lbl_top_val.setStyleSheet("font-size: 20px; font-weight: 700; color: #16a34a;")
        else:
            self.lbl_top_char.setText("—")
            self.lbl_top_val.setText("Esperando trazo")
            self.lbl_top_val.setStyleSheet("font-size: 17px; font-weight: 700; color: #1e293b;")

        # Reordenar las filas en el layout según el ranking
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
                self.rows_layout.removeWidget(row)
                self.rows_layout.insertWidget(rank_idx, row)
