"""Punto de entrada principal de la aplicación de escritorio HMM 2D/3D (main.py).

- Carga exclusivamente los modelos pre-entrenados desde models/2d_*.json y models/3d_*.json.
- NO contiene lógica matemática HMM ni rutinas de entrenamiento o calibración.
- Estructura la navegación entre:
  * Ventana A (gui/MainMenu.py): Menú Principal de Inicio.
  * Ventana B (gui/View2D.py): Reconocedor y Visualizador 2D.
  * Ventana C (gui/View3D.py): Reconocedor y Visualizador 3D (Lienzo Esférico).

Ejecución:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from core.config import PROJECT_ROOT, get_config
from core.FeatureExtractor import FeatureExtractor2D, FeatureExtractor3D
from core.HMMEngine import HMMClassifier
from gui.MainMenu import MainMenu
from gui.View2D import View2D
from gui.View3D import View3D


class HMMRecognizerApp(QMainWindow):
    """Ventana principal contenedora de las Ventanas A (Menú), B (2D) y C (3D)."""

    IDX_MENU = 0
    IDX_2D = 1
    IDX_3D = 2

    def __init__(self, models_dir: Path | None = None) -> None:
        super().__init__()
        self.cfg = get_config()
        mdir = models_dir if models_dir is not None else PROJECT_ROOT / "models"

        # Carga estricta de modelos pre-entrenados desde models/*.json
        self.classifier_2d = HMMClassifier.load_from_dir(mdir, pattern="2d_*.json")
        self.classifier_3d = HMMClassifier.load_from_dir(mdir, pattern="3d_*.json")

        self.extractor_2d = FeatureExtractor2D()
        self.extractor_3d = FeatureExtractor3D()

        self.setWindowTitle(
            "Reconocedor y Clasificador de Caracteres 2D/3D mediante Cadenas Ocultas de Markov (HMM)"
        )
        self.resize(1120, 720)
        self.setStyleSheet("QMainWindow { background-color: #f3f4f6; }")

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: #f3f4f6; color: #1e293b;")
        self.setCentralWidget(self.stack)

        self.view_menu = MainMenu(
            on_select_2d=self.show_2d,
            on_select_3d=self.show_3d,
            n_models_2d=len(self.classifier_2d.models),
            n_models_3d=len(self.classifier_3d.models),
        )
        self.view_2d = View2D(
            classifier=self.classifier_2d,
            extractor=self.extractor_2d,
            on_back_menu=self.show_menu,
            on_switch_3d=self.show_3d,
        )
        self.view_3d = View3D(
            classifier=self.classifier_3d,
            extractor=self.extractor_3d,
            on_back_menu=self.show_menu,
            on_switch_2d=self.show_2d,
        )

        self.stack.addWidget(self.view_menu)
        self.stack.addWidget(self.view_2d)
        self.stack.addWidget(self.view_3d)
        self.show_menu()

    def show_menu(self) -> None:
        self.stack.setCurrentIndex(self.IDX_MENU)

    def show_2d(self) -> None:
        self.stack.setCurrentIndex(self.IDX_2D)

    def show_3d(self) -> None:
        self.stack.setCurrentIndex(self.IDX_3D)


def main() -> None:
    app = QApplication(sys.argv)
    window = HMMRecognizerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
