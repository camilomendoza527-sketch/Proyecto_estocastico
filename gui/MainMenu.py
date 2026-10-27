"""Ventana A: Menú Principal de Inicio (SPEC §2.1).

Pantalla de bienvenida institucional sencilla y limpia que permite seleccionar:
1. Modo 2D: Reconocimiento y proyección de trazos en plano cartesiano bidimensional.
2. Modo 3D: Reconocimiento y proyección de trazos sobre la superficie de una esfera 3D.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.config import get_config


class MainMenu(QWidget):
    """Ventana A: Menú Principal de Inicio."""

    def __init__(
        self,
        on_select_2d: Callable[[], None],
        on_select_3d: Callable[[], None],
        n_models_2d: int = 27,
        n_models_3d: int = 27,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cfg = get_config()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Contenedor central con ancho máximo fijo para que al maximizar la ventana
        # no se estire ni se deformen los textos.
        container = QWidget()
        container.setMaximumWidth(860)
        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)

        # --- Encabezado institucional -----------------------------------------
        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        header_card.setStyleSheet(
            "QFrame#HeaderCard { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 8px; }"
        )
        hdr_layout = QVBoxLayout(header_card)
        hdr_layout.setContentsMargins(24, 20, 24, 20)
        hdr_layout.setSpacing(6)

        lbl_uni = QLabel("UNIVERSIDAD DE LOS LLANOS")
        lbl_uni.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_uni.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        lbl_uni.setStyleSheet("color: #1e293b; background: transparent;")

        lbl_fac = QLabel(
            "Facultad de Ciencias Básicas e Ingeniería — Programa de Ingeniería de Sistemas"
        )
        lbl_fac.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_fac.setFont(QFont("Segoe UI", 10))
        lbl_fac.setStyleSheet("color: #475569; background: transparent;")

        lbl_course = QLabel("Procesos Estocásticos (603504) — Proyecto Final")
        lbl_course.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_course.setFont(QFont("Segoe UI", 10))
        lbl_course.setStyleSheet("color: #64748b; background: transparent;")

        lbl_title = QLabel(
            "Reconocedor y Clasificador de Caracteres en Tiempo Real (2D / 3D)\n"
            "mediante Cadenas Ocultas de Markov (HMM)"
        )
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color: #0f172a; background: transparent;")

        hdr_layout.addWidget(lbl_uni)
        hdr_layout.addWidget(lbl_fac)
        hdr_layout.addWidget(lbl_course)
        hdr_layout.addSpacing(4)
        hdr_layout.addWidget(lbl_title)
        root.addWidget(header_card)

        # --- Selección de modo (2D y 3D) --------------------------------------
        cards_row = QHBoxLayout()
        cards_row.setSpacing(18)

        card_2d = self._build_mode_card(
            title="Modo 2D — Lienzo Plano",
            description=(
                "Dibuja caracteres sobre un lienzo bidimensional con clasificación "
                "y autocompletado punteado en tiempo real."
            ),
            status=f"Modelos 2D listos: {n_models_2d}/{len(self.cfg.letters)}",
            btn_text="Abrir Modo 2D",
            on_click=on_select_2d,
        )

        card_3d = self._build_mode_card(
            title="Modo 3D — Lienzo Esférico",
            description=(
                "Dibuja caracteres directamente sobre la superficie de una esfera 3D "
                "interactiva con proyección geodésica en tiempo real."
            ),
            status=f"Modelos 3D listos: {n_models_3d}/{len(self.cfg.letters)}",
            btn_text="Abrir Modo 3D (Esfera)",
            on_click=on_select_3d,
        )

        cards_row.addWidget(card_2d, stretch=1)
        cards_row.addWidget(card_3d, stretch=1)
        root.addLayout(cards_row)

        # --- Indicaciones de uso ----------------------------------------------
        conv_card = QFrame()
        conv_card.setObjectName("ConvCard")
        conv_card.setStyleSheet(
            "QFrame#ConvCard { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 8px; }"
        )
        conv_layout = QVBoxLayout(conv_card)
        conv_layout.setContentsMargins(20, 14, 20, 14)
        conv_layout.setSpacing(4)

        conv_title = QLabel(
            f"Indicaciones de trazo (27 letras minúsculas: {' '.join(self.cfg.letters)}):"
        )
        conv_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        conv_title.setStyleSheet("color: #334155; background: transparent;")
        conv_layout.addWidget(conv_title)

        for rule in self.cfg.stroke_convention:
            lbl_rule = QLabel(f"• {rule}")
            lbl_rule.setFont(QFont("Segoe UI", 9))
            lbl_rule.setStyleSheet("color: #475569; background: transparent;")
            conv_layout.addWidget(lbl_rule)

        root.addWidget(conv_card)
        outer.addWidget(container)

    @staticmethod
    def _build_mode_card(
        title: str,
        description: str,
        status: str,
        btn_text: str,
        on_click: Callable[[], None],
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("ModeCard")
        card.setStyleSheet(
            "QFrame#ModeCard { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 8px; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color: #1e293b; background: transparent;")

        lbl_desc = QLabel(description)
        lbl_desc.setWordWrap(True)
        lbl_desc.setFont(QFont("Segoe UI", 10))
        lbl_desc.setStyleSheet("color: #475569; background: transparent;")

        lbl_status = QLabel(status)
        lbl_status.setFont(QFont("Segoe UI", 9))
        lbl_status.setStyleSheet("color: #16a34a; background: transparent;")

        btn = QPushButton(btn_text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        btn.setFixedHeight(38)
        btn.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: #ffffff; border: none; "
            "border-radius: 6px; padding: 6px 14px; } "
            "QPushButton:hover { background-color: #1d4ed8; }"
        )
        btn.clicked.connect(on_click)

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_desc)
        layout.addSpacing(4)
        layout.addWidget(lbl_status)
        layout.addWidget(btn)
        return card
