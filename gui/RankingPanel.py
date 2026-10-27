"""Panel lateral derecho de clasificación y probabilidades en tiempo real (SPEC §2.2.2 y §2.3.2).

Muestra el listado ordenado descendentemente de las 27 letras del alfabeto español
según P(Carácter | O_{1:t}), junto con un icono estático de referencia del trazo,
porcentaje y valor decimal en [0.0, 1.0].
"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.config import get_config
from training.augment import canonical_unit_strokes


def build_letter_reference_pixmap(letter: str, size: int = 32) -> QPixmap:
    """Genera un icono estático de referencia con la forma canónica de la letra."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
    painter.setBrush(QColor("#f8fafc"))
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 5, 5)

    margin = 5.0
    span = size - 2.0 * margin
    try:
        strokes = canonical_unit_strokes(letter)
    except ValueError:
        strokes = []

    pen = QPen(QColor("#1e40af"), 2.0, Qt.PenStyle.SolidLine)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    for idx, stroke in enumerate(strokes):
        pts = [QPointF(margin + float(p[0]) * span, margin + float(p[1]) * span) for p in stroke]
        if len(pts) == 1:
            painter.drawEllipse(pts[0], 1.5, 1.5)
        elif len(pts) >= 2:
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            painter.drawPath(path)
        if idx == 0 and pts:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#16a34a"))
            painter.drawEllipse(pts[0], 2.0, 2.0)
            painter.setPen(pen)

    painter.end()
    return pixmap


class _LetterRankRow(QFrame):
    """Fila individual del ranking para una clase del alfabeto."""

    def __init__(self, rank: int, letter: str, icon_pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RankRow")
        self.setFixedHeight(42)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.lbl_rank = QLabel(f"{rank:2d}.")
        self.lbl_rank.setFixedWidth(26)
        self.lbl_rank.setFont(QFont("Segoe UI", 9))

        self.lbl_icon = QLabel()
        self.lbl_icon.setFixedSize(32, 32)
        self.lbl_icon.setPixmap(icon_pixmap)

        self.lbl_letter = QLabel(letter)
        self.lbl_letter.setFixedWidth(28)
        self.lbl_letter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_letter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)

        self.lbl_prob = QLabel("0.0% (0.000)")
        self.lbl_prob.setFixedWidth(98)
        self.lbl_prob.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_prob.setFont(QFont("Consolas", 9))

        layout.addWidget(self.lbl_rank)
        layout.addWidget(self.lbl_icon)
        layout.addWidget(self.lbl_letter)
        layout.addWidget(self.bar, stretch=1)
        layout.addWidget(self.lbl_prob)

        self.set_highlight(False)

    def set_highlight(self, is_top_active: bool) -> None:
        if is_top_active:
            self.setStyleSheet(
                "QFrame#RankRow { background-color: #eff6ff; border: 1px solid #93c5fd; border-radius: 6px; }"
            )
            self.lbl_rank.setStyleSheet("color: #1d4ed8; font-weight: bold; background: transparent;")
            self.lbl_letter.setStyleSheet("color: #1e3a8a; background: transparent;")
            self.lbl_prob.setStyleSheet("color: #1e3a8a; font-weight: bold; background: transparent;")
            self.bar.setStyleSheet(
                "QProgressBar { background-color: #dbeafe; border-radius: 4px; border: none; } "
                "QProgressBar::chunk { background-color: #2563eb; border-radius: 4px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame#RankRow { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; }"
            )
            self.lbl_rank.setStyleSheet("color: #64748b; background: transparent;")
            self.lbl_letter.setStyleSheet("color: #1e293b; background: transparent;")
            self.lbl_prob.setStyleSheet("color: #475569; background: transparent;")
            self.bar.setStyleSheet(
                "QProgressBar { background-color: #f1f5f9; border-radius: 4px; border: none; } "
                "QProgressBar::chunk { background-color: #64748b; border-radius: 4px; }"
            )


class RankingPanel(QFrame):
    """Panel derecho con la predicción Top-1 y el listado ordenado de las 27 clases."""

    def __init__(self, title: str = "Clasificación en Tiempo Real", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cfg = get_config()
        self.setObjectName("RankingRoot")
        self.setFixedWidth(365)
        self.setStyleSheet(
            "QFrame#RankingRoot { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 8px; }"
        )

        self.icons: dict[str, QPixmap] = {
            ch: build_letter_reference_pixmap(ch, size=32) for ch in self.cfg.letters
        }

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color: #1e293b; background: transparent;")
        root_layout.addWidget(lbl_title)

        # Recuadro superior con la letra reconocida actualmente
        self.top_card = QFrame()
        self.top_card.setObjectName("TopSummaryCard")
        self.top_card.setStyleSheet(
            "QFrame#TopSummaryCard { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; }"
        )
        card_layout = QHBoxLayout(self.top_card)
        card_layout.setContentsMargins(12, 8, 12, 8)
        card_layout.setSpacing(10)

        self.lbl_top_char = QLabel("—")
        self.lbl_top_char.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        self.lbl_top_char.setMinimumWidth(48)
        self.lbl_top_char.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_top_char.setStyleSheet("color: #1d4ed8; background: transparent;")

        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        self.lbl_top_prob = QLabel("Sin trazo activo")
        self.lbl_top_prob.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_top_prob.setStyleSheet("color: #1e293b; background: transparent;")

        self.lbl_viterbi = QLabel("Estado Viterbi: —  |  T = 0")
        self.lbl_viterbi.setFont(QFont("Segoe UI", 9))
        self.lbl_viterbi.setStyleSheet("color: #64748b; background: transparent;")

        info_col.addWidget(self.lbl_top_prob)
        info_col.addWidget(self.lbl_viterbi)
        card_layout.addWidget(self.lbl_top_char)
        card_layout.addLayout(info_col, stretch=1)
        root_layout.addWidget(self.top_card)

        hdr = QLabel("Pos.  Guía  Letra   Probabilidad P(c | O₁:ₜ)")
        hdr.setFont(QFont("Segoe UI", 9))
        hdr.setStyleSheet("color: #64748b; background: transparent;")
        root_layout.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self.rows_layout = QVBoxLayout(container)
        self.rows_layout.setContentsMargins(0, 0, 4, 0)
        self.rows_layout.setSpacing(4)

        self.rows: list[_LetterRankRow] = []
        for idx, ch in enumerate(self.cfg.letters, start=1):
            row = _LetterRankRow(idx, ch, self.icons[ch])
            self.rows.append(row)
            self.rows_layout.addWidget(row)
        self.rows_layout.addStretch(1)

        scroll.setWidget(container)
        root_layout.addWidget(scroll, stretch=1)

    def reset_ranking(self) -> None:
        """Restaura el panel al estado inicial."""
        self.lbl_top_char.setText("—")
        self.lbl_top_prob.setText("Dibuja una letra para clasificar")
        self.lbl_viterbi.setText("Estado Viterbi: —  |  T = 0")
        for idx, (row, ch) in enumerate(zip(self.rows, self.cfg.letters), start=1):
            row.lbl_rank.setText(f"{idx:2d}.")
            row.lbl_letter.setText(ch)
            row.lbl_icon.setPixmap(self.icons[ch])
            row.bar.setValue(0)
            row.lbl_prob.setText("0.0% (0.000)")
            row.set_highlight(False)

    def update_ranking(
        self,
        ranking: Sequence[tuple[str, float, float]],
        top_letter: str | None,
        current_state: int | None,
        n_states: int | None,
        n_obs: int,
    ) -> None:
        """Actualiza las 27 filas en orden descendente de probabilidad posterior."""
        if not ranking or n_obs == 0 or top_letter is None:
            self.reset_ranking()
            return

        top_prob = ranking[0][1]
        self.lbl_top_char.setText(top_letter)
        self.lbl_top_prob.setText(
            f"Letra más probable: '{top_letter}' ({top_prob * 100:5.1f}%)"
        )
        state_txt = (
            f"S_{current_state + 1}/{n_states}"
            if current_state is not None and n_states is not None
            else "—"
        )
        self.lbl_viterbi.setText(f"Estado Viterbi: {state_txt}  |  T = {n_obs}")

        for idx, (row, (letter, prob, _log_p)) in enumerate(zip(self.rows, ranking), start=1):
            row.lbl_rank.setText(f"{idx:2d}.")
            row.lbl_letter.setText(letter)
            if letter in self.icons:
                row.lbl_icon.setPixmap(self.icons[letter])
            bar_val = int(round(max(0.0, min(1.0, prob)) * 1000))
            row.bar.setValue(bar_val)
            row.lbl_prob.setText(f"{prob * 100:5.1f}% ({prob:.3f})")
            row.set_highlight(idx == 1)
