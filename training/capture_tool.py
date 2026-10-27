"""Herramienta independiente de captura de muestras 2D (training/capture_tool.py).

- Queda FUERA del menú de la GUI principal (cumple la regla de que la GUI nunca entrena).
- Lee el alfabeto de 27 letras, el tamaño del recuadro guía (expected_letter_size)
  y la convención de trazo desde config/config.toml.
- Levantar el ratón añade un sub-trazo pero NO cierra la letra; solo "Guardar"
  o "Limpiar" cierran la captura.
- Guarda cada muestra cruda en data/raw/2d/<alias>/user_NNN.json.

Ejecución:
    python -m training.capture_tool
"""

from __future__ import annotations

import sys
from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.config import get_config
from core.FeatureExtractor import FeatureExtractor2D
from training.augment import canonical_unit_strokes, load_raw_samples, save_raw_sample


class CaptureCanvas2D(QFrame):
    """Lienzo de dibujo 2D con recuadro guía y plantilla fantasma opcional."""

    def __init__(
        self,
        guide_size: float,
        on_stroke_updated: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.guide_size = float(guide_size)
        self.on_stroke_updated = on_stroke_updated
        self.strokes: list[list[tuple[float, float]]] = []
        self._current_stroke: list[tuple[float, float]] | None = None
        self.template_strokes: list[np.ndarray] = []
        self.show_template: bool = True

        self.setMinimumSize(int(self.guide_size + 140), int(self.guide_size + 140))
        self.setStyleSheet(
            "background-color: #181a20; border: 1px solid #2e323d; border-radius: 10px;"
        )
        self.setCursor(Qt.CursorShape.CrossCursor)

    def guide_rect(self) -> QRectF:
        """Recuadro guía centrado de lado expected_letter_size."""
        w, h = self.width(), self.height()
        s = self.guide_size
        return QRectF((w - s) / 2.0, (h - s) / 2.0, s, s)

    def set_template_letter(self, letter: str) -> None:
        self.template_strokes = canonical_unit_strokes(letter)
        self.update()

    def clear_canvas(self) -> None:
        self.strokes.clear()
        self._current_stroke = None
        self.update()
        if self.on_stroke_updated:
            self.on_stroke_updated()

    def undo_substroke(self) -> None:
        if self.strokes:
            self.strokes.pop()
            self.update()
            if self.on_stroke_updated:
                self.on_stroke_updated()

    def get_numpy_strokes(self) -> list[np.ndarray]:
        return [np.asarray(s, dtype=float) for s in self.strokes if len(s) > 0]

    # --- Eventos de ratón (levantar el ratón NO cierra la letra) --------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._current_stroke = [(pos.x(), pos.y())]
            self.strokes.append(self._current_stroke)
            self.update()
            if self.on_stroke_updated:
                self.on_stroke_updated()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._current_stroke is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self._current_stroke.append((pos.x(), pos.y()))
            self.update()
            if self.on_stroke_updated:
                self.on_stroke_updated()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._current_stroke = None
            self.update()
            if self.on_stroke_updated:
                self.on_stroke_updated()

    # --- Renderizado ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.guide_rect()

        # Recuadro guía de tamaño expected_letter_size
        guide_pen = QPen(QColor("#3c4456"), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(guide_pen)
        painter.drawRect(rect)

        # Líneas centrales sutiles
        axis_pen = QPen(QColor("#262b36"), 1.0, Qt.PenStyle.DotLine)
        painter.setPen(axis_pen)
        painter.drawLine(
            QPointF(rect.left(), rect.center().y()),
            QPointF(rect.right(), rect.center().y()),
        )
        painter.drawLine(
            QPointF(rect.center().x(), rect.top()),
            QPointF(rect.center().x(), rect.bottom()),
        )

        # Plantilla fantasma de referencia
        if self.show_template and self.template_strokes:
            tpl_pen = QPen(QColor(100, 140, 220, 55), 5.0, Qt.PenStyle.SolidLine)
            tpl_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(tpl_pen)
            for stroke in self.template_strokes:
                pts = [
                    QPointF(rect.left() + p[0] * rect.width(), rect.top() + p[1] * rect.height())
                    for p in stroke
                ]
                if len(pts) == 1:
                    painter.drawPoint(pts[0])
                elif len(pts) >= 2:
                    path = QPainterPath(pts[0])
                    for pt in pts[1:]:
                        path.lineTo(pt)
                    painter.drawPath(path)

        # Trazos dibujados por el usuario
        stroke_pen = QPen(QColor("#63b3ed"), 4.0, Qt.PenStyle.SolidLine)
        stroke_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        stroke_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(stroke_pen)

        for idx, stroke in enumerate(self.strokes):
            if not stroke:
                continue
            if len(stroke) == 1:
                painter.drawEllipse(QPointF(*stroke[0]), 3.5, 3.5)
            else:
                path = QPainterPath(QPointF(*stroke[0]))
                for x, y in stroke[1:]:
                    path.lineTo(QPointF(x, y))
                painter.drawPath(path)

            # Flecha punteada si hubo salto de pluma levantada entre sub-trazos
            if idx > 0 and self.strokes[idx - 1]:
                prev_pt = QPointF(*self.strokes[idx - 1][-1])
                curr_pt = QPointF(*stroke[0])
                jump_pen = QPen(QColor("#f6ad55"), 1.5, Qt.PenStyle.DotLine)
                painter.setPen(jump_pen)
                painter.drawLine(prev_pt, curr_pt)
                painter.setPen(stroke_pen)


class CaptureToolWindow(QMainWindow):
    """Ventana principal de la herramienta offline de captura."""

    def __init__(self) -> None:
        super().__init__()
        self.cfg = get_config()
        self.extractor = FeatureExtractor2D()
        self.setWindowTitle("Herramienta Offline de Captura 2D — HMM Estocásticos")
        self.resize(880, 560)
        self.setStyleSheet("background-color: #111318; color: #e2e8f0;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        # Lienzo izquierdo
        self.canvas = CaptureCanvas2D(
            guide_size=self.cfg.features.expected_letter_size,
            on_stroke_updated=self._on_canvas_changed,
        )
        layout.addWidget(self.canvas, stretch=3)

        # Panel derecho de controles y convenciones
        panel = QVBoxLayout()
        panel.setSpacing(12)
        layout.addLayout(panel, stretch=2)

        title = QLabel("Captura de Muestras 2D")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        panel.addWidget(title)

        # Selector de letra
        row_letter = QHBoxLayout()
        lbl_letter = QLabel("Letra objetivo:")
        lbl_letter.setFont(QFont("Segoe UI", 11))
        self.combo_letter = QComboBox()
        self.combo_letter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.combo_letter.setStyleSheet(
            "background-color: #1e222d; padding: 6px; border: 1px solid #3c4456; border-radius: 6px;"
        )
        for ch in self.cfg.letters:
            self.combo_letter.addItem(ch)
        self.combo_letter.currentTextChanged.connect(self._on_letter_changed)
        row_letter.addWidget(lbl_letter)
        row_letter.addWidget(self.combo_letter, stretch=1)
        panel.addLayout(row_letter)

        self.lbl_count = QLabel()
        self.lbl_count.setStyleSheet("color: #90cdf4; font-size: 13px;")
        panel.addWidget(self.lbl_count)

        # Recuadro con la convención de trazo obligatoria
        conv_box = QFrame()
        conv_box.setStyleSheet(
            "background-color: #1a1e29; border: 1px solid #2d3748; border-radius: 8px; padding: 10px;"
        )
        conv_layout = QVBoxLayout(conv_box)
        conv_title = QLabel("Convención obligatoria de trazo:")
        conv_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        conv_title.setStyleSheet("color: #fbd38d;")
        conv_layout.addWidget(conv_title)
        for rule in self.cfg.stroke_convention:
            item = QLabel(f"• {rule}")
            item.setWordWrap(True)
            item.setStyleSheet("color: #cbd5e0; font-size: 12px;")
            conv_layout.addWidget(item)
        panel.addWidget(conv_box)

        # Vista previa de símbolos discretizados
        self.lbl_symbols = QLabel("Símbolos extraídos: (dibuja en el recuadro guía)")
        self.lbl_symbols.setWordWrap(True)
        self.lbl_symbols.setStyleSheet(
            "background-color: #181a20; border: 1px solid #2d3748; border-radius: 6px; "
            "padding: 8px; font-family: Consolas; font-size: 11px; color: #a0aec0;"
        )
        panel.addWidget(self.lbl_symbols)

        panel.addStretch(1)

        # Botones de acción
        btn_style = (
            "QPushButton { background-color: #2b6cb0; color: white; border-radius: 6px; "
            "padding: 9px; font-weight: bold; } "
            "QPushButton:hover { background-color: #3182ce; }"
        )
        sec_style = (
            "QPushButton { background-color: #2d3748; color: #e2e8f0; border-radius: 6px; "
            "padding: 8px; } "
            "QPushButton:hover { background-color: #4a5568; }"
        )

        self.btn_save = QPushButton("Guardar Muestra (Enter / Espacio)")
        self.btn_save.setStyleSheet(btn_style)
        self.btn_save.clicked.connect(self.save_current_sample)
        panel.addWidget(self.btn_save)

        row_btns = QHBoxLayout()
        self.btn_undo = QPushButton("Deshacer sub-trazo (Ctrl+Z)")
        self.btn_undo.setStyleSheet(sec_style)
        self.btn_undo.clicked.connect(self.canvas.undo_substroke)

        self.btn_clear = QPushButton("Limpiar Lienzo (Esc)")
        self.btn_clear.setStyleSheet(sec_style)
        self.btn_clear.clicked.connect(self.canvas.clear_canvas)
        row_btns.addWidget(self.btn_undo)
        row_btns.addWidget(self.btn_clear)
        panel.addLayout(row_btns)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #68d391; font-size: 12px;")
        panel.addWidget(self.lbl_status)

        # Atajos de teclado
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, self.save_current_sample)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.save_current_sample)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.canvas.clear_canvas)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.canvas.undo_substroke)

        self._on_letter_changed(self.combo_letter.currentText())

    def current_letter(self) -> str:
        return self.combo_letter.currentText()

    def _on_letter_changed(self, letter: str) -> None:
        self.canvas.set_template_letter(letter)
        self.canvas.clear_canvas()
        self._refresh_count()

    def _refresh_count(self) -> None:
        letter = self.current_letter()
        n = len(load_raw_samples(letter, mode="2d"))
        alias = self.cfg.file_name(letter)
        self.lbl_count.setText(f"Muestras guardadas para '{letter}' (data/raw/2d/{alias}): {n}")

    def _on_canvas_changed(self) -> None:
        strokes = self.canvas.get_numpy_strokes()
        if not strokes:
            self.lbl_symbols.setText("Símbolos extraídos: —")
            return
        res = self.extractor.extract(strokes, realtime=False)
        labels = [self.extractor.symbol_label(int(s)) for s in res.symbols]
        preview = " ".join(labels[:32]) + (" ..." if len(labels) > 32 else "")
        self.lbl_symbols.setText(
            f"Sub-trazos: {len(strokes)} | T={len(res.symbols)} símbolos:\n{preview}"
        )

    def save_current_sample(self) -> None:
        strokes = self.canvas.get_numpy_strokes()
        res = self.extractor.extract(strokes, realtime=False)
        if len(res.symbols) < 2:
            self.lbl_status.setText("Trazo demasiado corto para guardar.")
            return
        letter = self.current_letter()
        path = save_raw_sample(letter, strokes, mode="2d", prefix="user")
        self.canvas.clear_canvas()
        self._refresh_count()
        self.lbl_status.setText(f"Guardado: {path.name}")


def main() -> None:
    app = QApplication(sys.argv)
    win = CaptureToolWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
