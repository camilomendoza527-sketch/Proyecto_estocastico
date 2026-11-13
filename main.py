"""Punto de entrada principal de la aplicación de escritorio HMM 2D/3D (main.py).

Estructura en tres vistas principales según SPEC §2.1–2.3:
- Ventana A: Menú Principal de Inicio (gui/MainMenu.py)
- Ventana B: Reconocedor y Visualizador 2D (gui/View2D.py)
- Ventana C: Reconocedor y Visualizador 3D (gui/View3D.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.config import PROJECT_ROOT, get_config
from core.FeatureExtractor import FeatureExtractor2D, FeatureExtractor3D
from core.HMMEngine import HMMClassifier
from gui.LetterGuide import LetterGuideView
from gui.MainMenu import MainMenu
from gui.View2D import View2D
from gui.View3D import View3D


class HMMRecognizerApp(QMainWindow):
    """Ventana principal con barra lateral de navegación y área de contenido apilada."""

    IDX_INICIO = 0
    IDX_MENU = 0
    IDX_2D = 1
    IDX_3D = 2
    IDX_GUIA = 3

    def __init__(self, models_dir: Path | None = None) -> None:
        super().__init__()
        self.cfg = get_config()
        mdir = models_dir if models_dir is not None else PROJECT_ROOT / "models"

        self.classifier_2d = HMMClassifier.load_from_dir(mdir, pattern="2d_*.json")
        self.classifier_3d = HMMClassifier.load_from_dir(mdir, pattern="3d_*.json")
        self.extractor_2d = FeatureExtractor2D()
        self.extractor_3d = FeatureExtractor3D()

        self.setWindowTitle("Reconocedor HMM — Procesos Estocásticos")
        self.resize(1180, 750)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(
            "QMainWindow { background-color: #f8fafc; font-family: 'Segoe UI', sans-serif; }"
        )

        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central_widget)

        # 1. Barra lateral izquierda
        sidebar = QFrame()
        sidebar.setFixedWidth(205)
        sidebar.setStyleSheet(
            "QFrame { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }"
        )
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 24, 16, 20)
        side_layout.setSpacing(6)

        lbl_app = QLabel("Reconocedor HMM")
        lbl_app.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #0f172a; background: transparent; border: none;"
        )
        lbl_sub = QLabel("Procesos Estocásticos")
        lbl_sub.setStyleSheet(
            "font-size: 11px; color: #64748b; background: transparent; border: none; margin-bottom: 16px;"
        )
        side_layout.addWidget(lbl_app)
        side_layout.addWidget(lbl_sub)

        self.btn_inicio = QPushButton("Inicio")
        self.btn_2d = QPushButton("Modo 2D")
        self.btn_3d = QPushButton("Modo 3D")
        self.btn_guia = QPushButton("Guía de letras")

        self.btn_inicio.clicked.connect(self.show_menu)
        self.btn_2d.clicked.connect(self.show_2d)
        self.btn_3d.clicked.connect(self.show_3d)
        self.btn_guia.clicked.connect(self.show_guia)

        self.nav_buttons = [self.btn_inicio, self.btn_2d, self.btn_3d, self.btn_guia]
        for btn in self.nav_buttons:
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(38)
            side_layout.addWidget(btn)

        side_layout.addStretch(1)
        lbl_footer = QLabel("Universidad de los Llanos")
        lbl_footer.setStyleSheet(
            "font-size: 10px; color: #94a3b8; background: transparent; border: none;"
        )
        side_layout.addWidget(lbl_footer)
        central_layout.addWidget(sidebar)

        # 2. Vistas principales (Ventanas A, B y C)
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("QStackedWidget { background-color: #ffffff; }")
        central_layout.addWidget(self.stack, stretch=1)

        self.view_inicio = MainMenu(on_select_2d=self.show_2d, on_select_3d=self.show_3d)
        self.view_2d = View2D(classifier=self.classifier_2d, extractor=self.extractor_2d)
        self.view_3d = View3D(classifier=self.classifier_3d, extractor=self.extractor_3d)
        self.view_guia = LetterGuideView()

        self.stack.addWidget(self.view_inicio)
        self.stack.addWidget(self.view_2d)
        self.stack.addWidget(self.view_3d)
        self.stack.addWidget(self.view_guia)
        self.navigate_to(self.IDX_INICIO)

    def navigate_to(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            if i == index:
                btn.setStyleSheet(
                    "QPushButton { background-color: #dcfce7; color: #166534; font-size: 13px; "
                    "font-weight: 600; border: none; border-radius: 6px; text-align: left; padding-left: 12px; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background-color: transparent; color: #334155; font-size: 13px; "
                    "font-weight: 500; border: none; border-radius: 6px; text-align: left; padding-left: 12px; } "
                    "QPushButton:hover { background-color: #f1f5f9; }"
                )

    def show_menu(self) -> None:
        self.navigate_to(self.IDX_MENU)

    def show_2d(self) -> None:
        self.navigate_to(self.IDX_2D)

    def show_3d(self) -> None:
        self.navigate_to(self.IDX_3D)

    def show_guia(self) -> None:
        self.navigate_to(self.IDX_GUIA)


def main() -> None:
    app = QApplication(sys.argv)
    window = HMMRecognizerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
