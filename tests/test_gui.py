"""Pruebas de integración de la GUI (Ventanas A, B y C) y separación arquitectónica."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from core.config import PROJECT_ROOT, get_config
from main import HMMRecognizerApp
from training.augment import canonical_unit_strokes


_KEEP_ALIVE: list = []


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _KEEP_ALIVE.append(app)
    return app


@pytest.fixture(scope="session")
def app_window(qapp):
    win = HMMRecognizerApp()
    _KEEP_ALIVE.append(win)
    return win


def test_gui_has_no_training_or_capture_links():
    """Verifica que ni main.py ni gui/*.py invoquen entrenamiento ni capture_tool."""
    files = [PROJECT_ROOT / "main.py", *sorted((PROJECT_ROOT / "gui").glob("*.py"))]
    forbidden = ("capture_tool", "train_all", "train_letter", ".fit(")
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"Token prohibido {token!r} encontrado en {path.name}"


def test_app_loads_models_and_navigates_windows(app_window):
    win = app_window
    assert len(win.classifier_2d.models) == 27
    assert len(win.classifier_3d.models) == 27

    win.show_menu()
    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_MENU
    win.show_2d()
    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_2D
    win.show_3d()
    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_3D
    win.show_guia()
    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_GUIA
    win.view_guia.select_letter("v")
    assert len(win.view_guia.demo_canvas._strokes) == 1
    win.view_guia.select_letter("d")
    assert len(win.view_guia.demo_canvas._strokes) == 2
    win.view_guia.select_letter("ñ")
    assert len(win.view_guia.demo_canvas._strokes) == 2
    win.show_menu()
    assert win.stack.currentIndex() == HMMRecognizerApp.IDX_MENU


def test_view2d_realtime_projection_and_fullscreen_resize(qapp, app_window):
    win = app_window
    win.show_2d()
    v2d = win.view_2d
    box = get_config().features.expected_letter_size

    strokes_i = [s * box + 80.0 for s in canonical_unit_strokes("i")]
    half_stem = strokes_i[0][: len(strokes_i[0]) // 2]
    v2d.canvas.strokes = [[(float(x), float(y)) for x, y in half_stem]]
    v2d._evaluate_realtime()

    assert len(v2d.canvas.projected_strokes) >= 1
    assert v2d.ranking_panel.lbl_top_char.text() != "—"

    win.resize(1920, 1080)
    qapp.processEvents()
    letters_shown = [row.lbl_letter.text() for row in v2d.ranking_panel.rows]
    assert set(letters_shown) == set(get_config().letters)

    dot_stroke = strokes_i[1]
    v2d.canvas.strokes.append([(float(x), float(y)) for x, y in dot_stroke])
    v2d._evaluate_realtime()
    assert len(v2d.canvas.strokes) == 2

    v2d.btn_clear.click()
    assert len(v2d.canvas.strokes) == 0
    assert len(v2d.canvas.projected_strokes) == 0
    assert v2d.ranking_panel.lbl_top_char.text() == "—"


def test_view3d_mouse_raycast_drawing_and_projection(app_window):
    """Verifica que arrastrar el ratón con clic izquierdo sobre SphereCanvas3D traza en la esfera."""
    win = app_window
    win.show_3d()
    v3d = win.view_3d
    canvas = v3d.canvas3d
    canvas.resize(600, 600)

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
    assert canvas.strokes[0][-1][2] < canvas.strokes[0][0][2]
    assert len(canvas.projected_strokes) >= 1
    assert v3d.ranking_panel.lbl_top_char.text() != "—"

    assert canvas.rotate_mode is False
    v3d.btn_rotate_mode.click()
    assert canvas.rotate_mode is True

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
    assert len(canvas.strokes) == 1

    v3d.btn_rotate_mode.click()
    assert canvas.rotate_mode is False

    v3d.btn_restart.click()
    assert len(canvas.strokes) == 0
    assert len(canvas.projected_strokes) == 0
    assert v3d.ranking_panel.lbl_top_char.text() == "—"


def test_view2d_arbitrary_stroke_always_shows_statistical_ranking(app_window):
    """Incluso un trazo irregular completo siempre muestra la estadística de las 27 letras."""
    import numpy as np

    win = app_window
    win.show_2d()
    v2d = win.view_2d

    t_zz = np.linspace(0.0, 10.0 * np.pi, 180)
    zz = [(float(300.0 + 8.0 * t), float(400.0 + 90.0 * np.sin(t))) for t in t_zz]
    v2d.canvas.strokes = [zz]
    v2d.canvas._current_stroke = None
    v2d._evaluate_realtime()

    assert v2d.ranking_panel.lbl_top_char.text() != "—"
    assert "Reconociendo:" in v2d.ranking_panel.lbl_top_val.text()

    v2d.btn_clear.click()


def test_view2d_undo_button_and_live_telemetry_bar(app_window):
    """El botón 'Deshacer trazo' elimina solo el último sub-trazo y la barra inferior muestra telemetría en vivo."""
    win = app_window
    win.show_2d()
    v2d = win.view_2d

    assert not hasattr(v2d, "btn_restart")
    assert v2d.btn_undo.text() == "Deshacer trazo"
    assert v2d.btn_clear.text() == "Limpiar lienzo"
    assert v2d.lbl_obs.text() == v2d._IDLE_OBS_TEXT

    stroke_1 = [(300.0, float(150 + 10 * i)) for i in range(21)]
    stroke_2 = [(float(230 + 10 * i), 220.0) for i in range(15)]

    v2d.canvas.strokes = [stroke_1]
    v2d.canvas._current_stroke = None
    v2d._evaluate_realtime()
    obs_1 = v2d.lbl_obs.text()
    assert "Sub-trazos: 1" in obs_1
    assert "T = " in obs_1
    assert "Proyección '" in obs_1
    assert v2d.ranking_panel.lbl_top_char.text() != "—"

    v2d.canvas.strokes.append(stroke_2)
    v2d._evaluate_realtime()
    obs_2 = v2d.lbl_obs.text()
    assert "Sub-trazos: 2" in obs_2

    v2d.btn_undo.click()
    assert len(v2d.canvas.strokes) == 1
    assert "Sub-trazos: 1" in v2d.lbl_obs.text()
    assert v2d.ranking_panel.lbl_top_char.text() != "—"

    v2d.btn_undo.click()
    assert len(v2d.canvas.strokes) == 0
    assert v2d.lbl_obs.text() == v2d._IDLE_OBS_TEXT
    assert v2d.ranking_panel.lbl_top_char.text() == "—"


def test_letter_guide_modal_trajectories_and_q_3_strokes(app_window):
    """La guía refleja los modos reales del dataset (x abajo->arriba, k=2, p=2, a=2, g=2, q=3)."""
    win = app_window
    win.show_guia()
    guide = win.view_guia

    guide.select_letter("q")
    assert len(guide.demo_canvas._strokes) == 3
    assert "3 sub-trazos" in guide.lbl_info.text()
    assert "3 (morado)" in guide.lbl_info.text()

    guide.select_letter("x")
    st_x = guide.demo_canvas._strokes
    assert len(st_x) == 2
    assert st_x[0][-1][0] - st_x[0][0][0] > 0.15 and st_x[0][-1][1] - st_x[0][0][1] > 0.15
    assert st_x[1][-1][0] - st_x[1][0][0] > 0.15 and st_x[1][-1][1] - st_x[1][0][1] < -0.15

    for ch, expected_n in (("a", 2), ("g", 2), ("k", 2), ("p", 2), ("o", 1)):
        guide.select_letter(ch)
        assert len(guide.demo_canvas._strokes) == expected_n
