"""Pruebas de integración de la GUI (Ventanas A, B y C) y separación arquitectónica."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from core.config import PROJECT_ROOT, get_config
from main import HMMRecognizerApp
from training.augment import canonical_unit_strokes


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_gui_has_no_training_or_capture_links():
    """Verifica que ni main.py ni gui/*.py invoquen entrenamiento ni capture_tool."""
    files = [PROJECT_ROOT / "main.py", *sorted((PROJECT_ROOT / "gui").glob("*.py"))]
    forbidden = ("capture_tool", "train_all", "train_letter", ".fit(")
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"Token prohibido {token!r} encontrado en {path.name}"


def test_app_loads_models_and_navigates_windows(qapp):
    win = HMMRecognizerApp()
    assert len(win.classifier_2d.models) == 27
    assert len(win.classifier_3d.models) == 27

    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_MENU
    win.show_2d()
    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_2D
    win.show_3d()
    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_3D
    win.show_menu()
    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_MENU


def test_view2d_realtime_projection_and_fullscreen_resize(qapp):
    win = HMMRecognizerApp()
    win.show_2d()
    v2d = win.view_2d
    box = get_config().features.expected_letter_size

    # Sub-trazo 1: cuerpo de 'i' (mitad superior)
    strokes_i = [s * box + 80.0 for s in canonical_unit_strokes("i")]
    half_stem = strokes_i[0][: len(strokes_i[0]) // 2]
    v2d.canvas.strokes = [[(float(x), float(y)) for x, y in half_stem]]
    v2d._evaluate_realtime()

    assert len(v2d.canvas.projected_strokes) >= 1
    assert v2d.ranking_panel.lbl_top_char.text() != "—"

    # Simular cambio a pantalla completa (1920x1080) y verificar que las 27 letras se ven
    win.resize(1920, 1080)
    qapp.processEvents()
    letters_shown = [row.lbl_letter.text() for row in v2d.ranking_panel.rows]
    assert set(letters_shown) == set(get_config().letters)

    # Añadir segundo sub-trazo (simula levantar el ratón y volver a trazar) NO borra el primero
    dot_stroke = strokes_i[1]
    v2d.canvas.strokes.append([(float(x), float(y)) for x, y in dot_stroke])
    v2d._evaluate_realtime()
    assert len(v2d.canvas.strokes) == 2

    # Únicamente Limpiar / Reiniciar cierran la captura
    v2d.btn_clear.click()
    assert len(v2d.canvas.strokes) == 0
    assert len(v2d.canvas.projected_strokes) == 0
    assert v2d.ranking_panel.lbl_top_char.text() == "—"


def test_view3d_mouse_raycast_drawing_and_projection(qapp):
    """Verifica que arrastrar el ratón con clic izquierdo sobre SphereCanvas3D traza en la esfera."""
    win = HMMRecognizerApp()
    win.show_3d()
    v3d = win.view_3d
    canvas = v3d.canvas3d
    canvas.resize(600, 600)

    # Simular clic izquierdo y arrastre desde el centro de la esfera hacia abajo/derecha
    pts_px = [(280.0, 240.0 + i * 5.0) for i in range(25)]

    press_ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(*pts_px[0]),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(press_ev)

    for x, y in pts_px[1:]:
        move_ev = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(x, y),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mouseMoveEvent(move_ev)

    rel_ev = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(*pts_px[-1]),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseReleaseEvent(rel_ev)

    assert len(canvas.strokes) == 1
    assert len(canvas.strokes[0]) >= 10
    assert len(canvas.projected_strokes) >= 1
    assert v3d.ranking_panel.lbl_top_char.text() != "—"

    # Verificar botón de giro de la esfera (Modo Rotar ON/OFF)
    assert canvas.rotate_mode is False
    v3d.btn_rotate_mode.click()
    assert canvas.rotate_mode is True
    assert "ON" in v3d.btn_rotate_mode.text()

    # Con rotate_mode activo, arrastrar con clic izquierdo orbita la cámara sin añadir nuevos trazos
    az_before, el_before = canvas.current_camera_orientation()
    rot_press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(300.0, 300.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(rot_press)
    rot_move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(360.0, 260.0),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseMoveEvent(rot_move)
    rot_rel = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(360.0, 260.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseReleaseEvent(rot_rel)
    az_after, el_after = canvas.current_camera_orientation()
    assert (az_after != az_before) or (el_after != el_before)
    assert len(canvas.strokes) == 1  # No agregó otro sub-trazo mientras rotaba

    # Desactivar modo rotar
    v3d.btn_rotate_mode.click()
    assert canvas.rotate_mode is False

    v3d.btn_restart.click()
    assert len(canvas.strokes) == 0
    assert len(canvas.projected_strokes) == 0
    assert v3d.ranking_panel.lbl_top_char.text() == "—"


def test_letter_guide_view(qapp):
    """Verifica que la pantalla Guía de letras cargue las 27 letras y actualice la animación."""
    win = HMMRecognizerApp()
    win.show_guia()
    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_GUIA

    guia = win.view_guia
    assert len(guia._buttons) == 27
    assert "a" in guia._buttons
    assert "ñ" in guia._buttons

    # Seleccionar 'a'
    guia.select_letter("a")
    assert guia.current_letter == "a"
    assert "Letra a" in guia.lbl_info.text()
    assert guia.demo_canvas._letter == "a"
    assert len(guia.demo_canvas._strokes) >= 1

    # Seleccionar 'ñ'
    guia.select_letter("ñ")
    assert guia.current_letter == "ñ"
    assert "Letra ñ" in guia.lbl_info.text()
    assert guia.demo_canvas._letter == "ñ"
    assert len(guia.demo_canvas._start_points) >= 1

    # Probar trazado interactivo del usuario con el ratón
    canvas = guia.demo_canvas
    assert len(canvas.user_strokes) == 0

    p1 = QPointF(100.0, 100.0)
    press_ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        p1,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(press_ev)
    assert len(canvas.user_strokes) == 1
    assert len(canvas.user_strokes[0]) == 1

    p2 = QPointF(105.0, 105.0)
    move_ev = QMouseEvent(
        QEvent.Type.MouseMove,
        p2,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseMoveEvent(move_ev)
    assert len(canvas.user_strokes[0]) == 2

    rel_ev = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        p2,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseReleaseEvent(rel_ev)
    assert canvas._current_user_stroke is None
    assert len(canvas.user_strokes) == 1

    # Probar botón de limpiar trazo
    guia.btn_clear.click()
    assert len(canvas.user_strokes) == 0

    # Probar botón de reiniciar animación
    guia.btn_restart.click()
    assert canvas._show_anim is True

    # Probar botón de pausar / reanudar
    guia.btn_play_pause.click()
    assert canvas._is_paused is True
    assert "Reanudar" in guia.btn_play_pause.text()

    guia.btn_play_pause.click()
    assert canvas._is_paused is False
    assert "Pausar" in guia.btn_play_pause.text()

    # Probar ciclado de velocidad (1x -> 0.5x -> 1.5x -> 1x)
    guia.btn_speed.click()
    assert canvas._speed_multiplier == 0.5
    guia.btn_speed.click()
    assert canvas._speed_multiplier == 1.5
    guia.btn_speed.click()
    assert canvas._speed_multiplier == 1.0

