"""Punto de entrada principal de la aplicación de escritorio HMM 2D/3D (main.py).

Arquitectura de navegación moderna con barra lateral izquierda:
- Inicio (Menú de bienvenida)
- Modo 2D (Plano cartesiano)
- Modo 3D (Superficie esférica)
- Guía de letras (Trazos canónicos, animación caligráfica y selector de 27 letras)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
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

        # Carga de modelos pre-entrenados
        self.classifier_2d = HMMClassifier.load_from_dir(mdir, pattern="2d_*.json")
        self.classifier_3d = HMMClassifier.load_from_dir(mdir, pattern="3d_*.json")

        self.extractor_2d = FeatureExtractor2D()
        self.extractor_3d = FeatureExtractor3D()

        self.setWindowTitle("Reconocedor HMM — Procesos Estocásticos")
        self.resize(1180, 750)
        self.setMinimumSize(960, 640)
        self.setStyleSheet("QMainWindow { background-color: #f8fafc; font-family: 'Segoe UI', sans-serif; }")

        # Widget central con división horizontal: Sidebar | Contenido
        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central_widget)

        # --- 1. Barra lateral izquierda (Sidebar) -----------------------------
        sidebar = QFrame()
        sidebar.setFixedWidth(205)
        sidebar.setStyleSheet(
            "QFrame { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }"
        )
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 24, 16, 20)
        side_layout.setSpacing(6)

        # Encabezado institucional
        lbl_app = QLabel("Reconocedor HMM")
        lbl_app.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a; background: transparent; border: none;")
        lbl_sub = QLabel("Procesos Estocásticos")
        lbl_sub.setStyleSheet("font-size: 11px; color: #64748b; background: transparent; border: none; margin-bottom: 16px;")
        side_layout.addWidget(lbl_app)
        side_layout.addWidget(lbl_sub)

        # Botones de navegación
        self.nav_buttons: list[QPushButton] = []

        self.btn_inicio = QPushButton("Inicio")
        self.btn_2d = QPushButton("Modo 2D")
        self.btn_3d = QPushButton("Modo 3D")
        self.btn_guia = QPushButton("Guía de letras")

        self.btn_inicio.clicked.connect(lambda: self.navigate_to(self.IDX_INICIO))
        self.btn_2d.clicked.connect(lambda: self.navigate_to(self.IDX_2D))
        self.btn_3d.clicked.connect(lambda: self.navigate_to(self.IDX_3D))
        self.btn_guia.clicked.connect(lambda: self.navigate_to(self.IDX_GUIA))

        for btn in (self.btn_inicio, self.btn_2d, self.btn_3d, self.btn_guia):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(38)
            side_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        side_layout.addStretch(1)

        # Pie de página institucional en la barra lateral
        lbl_footer = QLabel("Universidad de los Llanos")
        lbl_footer.setStyleSheet("font-size: 10px; color: #94a3b8; background: transparent; border: none;")
        side_layout.addWidget(lbl_footer)

        central_layout.addWidget(sidebar)

        # --- 2. Área principal de contenido (QStackedWidget) ------------------
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("QStackedWidget { background-color: #ffffff; }")
        central_layout.addWidget(self.stack, stretch=1)

        # Inicialización de las vistas
        self.view_inicio = MainMenu(
            on_select_2d=lambda: self.navigate_to(self.IDX_2D),
            on_select_3d=lambda: self.navigate_to(self.IDX_3D),
        )
        self.view_2d = View2D(
            classifier=self.classifier_2d,
            extractor=self.extractor_2d,
        )
        self.view_3d = View3D(
            classifier=self.classifier_3d,
            extractor=self.extractor_3d,
        )
        self.view_guia = LetterGuideView()

        self.stack.addWidget(self.view_inicio)  # 0
        self.stack.addWidget(self.view_2d)      # 1
        self.stack.addWidget(self.view_3d)      # 2
        self.stack.addWidget(self.view_guia)    # 3

        self.navigate_to(self.IDX_INICIO)

    def navigate_to(self, index: int) -> None:
        """Cambia la vista activa y actualiza el estilo de los botones de la barra lateral."""
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            if i == index:
                btn.setStyleSheet(
                    "QPushButton { "
                    "background-color: #dcfce7; "
                    "color: #166534; "
                    "font-size: 13px; "
                    "font-weight: 600; "
                    "border: none; "
                    "border-radius: 6px; "
                    "text-align: left; "
                    "padding-left: 12px; "
                    "}"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { "
                    "background-color: transparent; "
                    "color: #334155; "
                    "font-size: 13px; "
                    "font-weight: 500; "
                    "border: none; "
                    "border-radius: 6px; "
                    "text-align: left; "
                    "padding-left: 12px; "
                    "} "
                    "QPushButton:hover { "
                    "background-color: #f1f5f9; "
                    "}"
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
