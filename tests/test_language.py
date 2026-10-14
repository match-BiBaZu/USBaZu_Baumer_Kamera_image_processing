"""Switch presentation while live acquisition, calibration and replay keep their state."""

import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog

from bibazu_vision.app import MainWindow
from bibazu_vision.recording import Clip


@pytest.fixture(scope="module")
def qt_app():
    application = QApplication.instance() or QApplication([])
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if not QFontDatabase.families() and font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            application.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0]))
    return application


def wait_for(application, predicate):
    deadline = time.perf_counter() + 2
    while time.perf_counter() < deadline:
        application.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    raise AssertionError("GUI condition did not become true")


def dispose(window, application):
    window.timer.stop()
    window.play_timer.stop()
    if window.worker is not None:
        window.worker.stop()
        assert window.worker.wait(3000)
    application.processEvents()
    window.close()
    window.deleteLater()
    application.processEvents()


def test_switch_during_live_keeps_worker_controls_image_and_native_preferences(qt_app, tmp_path):
    native = {
        "processing": {
            "threshold": 78,
            "min_area": 10000,
            "method": "pca",
            "roi": [20, 110, 610, 245],
            "zero_offset_deg": -2.63,
        },
        "camera": {
            "mode": "frame_trigger",
            "trigger_line": "Line2",
            "trigger_activation": "FallingEdge",
        },
        "background_pending": True,
    }
    profile_path = tmp_path / "settings.json"
    profile_path.write_text(json.dumps(native), encoding="utf-8")
    window = MainWindow(tmp_path)
    reopened = None
    try:
        window.connect_camera(True)
        wait_for(qt_app, lambda: window.last_snapshot is not None)
        window.start_mode.setCurrentIndex(window.start_mode.findData("manual"))
        window.command("capture_config", asdict(window.capture_config()))
        window.command("trigger")
        wait_for(qt_app, lambda: window.last_snapshot["status"].state == "waiting")
        assert not window.processing_group.isEnabled()
        window.timer.stop()
        worker = window.worker
        before_id = worker.snapshot()["frame_id"]
        image_key = window.live_view.source.cacheKey()
        controls = (
            window.processing_config(),
            window.capture_config(),
            window.camera_mode.currentData(),
            window.line.currentText(),
            window.edge.currentText(),
        )
        commands = []
        window.command = lambda *args: commands.append(args)
        QTest.mouseClick(window.language_button, Qt.MouseButton.LeftButton)
        assert window.language == "en"
        assert window.language_button.text() == "Deutsch"
        assert window.connect_button.text() == "Connect camera"
        assert window.processing_group.title() == "2 · Detection & reference"
        assert window.tabs.tabText(1) == "Passage · slow motion"
        assert "background subtraction" in window.segmentation_label.text()
        assert "ID gaps" in window.rate_label.text()
        assert "Recording:" in window.state_label.text()
        assert "angle history" in window.trace.caption
        assert window.live_view.source.cacheKey() == image_key
        assert window.worker is worker and worker.isRunning()
        assert worker.snapshot()["status"].state == "waiting"
        assert not window.processing_group.isEnabled()
        assert controls == (
            window.processing_config(),
            window.capture_config(),
            window.camera_mode.currentData(),
            window.line.currentText(),
            window.edge.currentText(),
        )
        assert not commands  # Localization must not send new hardware/processing settings.
        wait_for(qt_app, lambda: worker.snapshot()["frame_id"] > before_id)
        worker.message.emit("Hintergrund aufgenommen. Nur bei leerer Rutsche verwenden.")
        assert "Background captured." in window.log.toPlainText()
        window.camera_error("RuntimeError: Kameramerkmal ExposureTime ist nicht schreibbar.")
        assert "Camera feature ExposureTime is not writable." in window.state_label.text()
        QTest.mouseClick(window.language_button, Qt.MouseButton.LeftButton)
        assert window.language == "de"
        assert "Kameramerkmal ExposureTime ist nicht schreibbar." in window.state_label.text()
        assert "Hintergrund aufgenommen." in window.log.toPlainText()
        window.toggle_language()
        assert json.loads(profile_path.read_text()) == {**native, "language": "en"}
        dispose(window, qt_app)
        window = None
        reopened = MainWindow(tmp_path)
        assert reopened.language == "en"
        assert reopened.connect_button.text() == "Connect camera"
        assert reopened.processing_config().zero_offset_deg == -2.63
        assert reopened.camera_mode.currentData() == "frame_trigger"
        assert reopened.line.currentText() == "Line2"
        assert reopened.edge.currentText() == "FallingEdge"
    finally:
        if window is not None:
            dispose(window, qt_app)
        if reopened is not None:
            dispose(reopened, qt_app)


def test_switch_preserves_playback_and_translates_exports_and_nested_messages(
    qt_app, tmp_path, monkeypatch
):
    frames = [np.full((80, 120), 100 + index, np.uint8) for index in range(3)]
    measurements = [
        {
            "angle_deg": index * 3,
            "axis_deg": index * 3,
            "center": [60, 40],
            "box": [[20, 20], [100, 20], [100, 60], [20, 60]],
        }
        for index in range(3)
    ]
    clip = Clip(
        tmp_path,
        frames,
        [100, 100.05, 100.1],
        [17, 18, 19],
        measurements,
        {"stop_reason": "part_left"},
    )
    monkeypatch.setattr("bibazu_vision.app.load_clip", lambda path: clip)
    window = MainWindow(tmp_path)
    try:
        window.open_clip(tmp_path)
        window.slider.setValue(1)
        window.play_timer.setInterval(60000)
        window.toggle_play()
        position = window.review_position
        window.speed.setCurrentIndex(2)
        QTest.mouseClick(window.language_button, Qt.MouseButton.LeftButton)
        assert window.play_timer.isActive()
        assert window.play_button.text() == "Ⅱ Pause"
        assert window.clip is clip and window.slider.value() == 1
        assert window.review_position == position
        assert window.speed.currentData() == 0.25
        assert "Frame 2/3" in window.review_info.text()
        assert "Angle +3.00°" in window.review_info.text()
        assert "part_left" in window.review_info.text()
        captions = []
        csv_path = tmp_path / "Messungen" / "Rutsche.csv"

        def choose_csv(parent, caption, *args, **kwargs):
            captions.append(caption)
            assert kwargs["options"] == QFileDialog.Option.DontUseNativeDialog
            return str(csv_path), "CSV (*.csv)"

        monkeypatch.setattr(QFileDialog, "getSaveFileName", choose_csv)
        window.export_csv()
        assert captions == ["Export measurements"]
        assert f"CSV saved: {csv_path}" in window.log.toPlainText()
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert [int(row["frame_id"]) for row in rows] == [17, 18, 19]
        assert [float(row["angle_deg"]) for row in rows] == [0, 3, 6]
        window.note(
            "Aufzeichnung fehlgeschlagen: RuntimeError: "
            "Kameramerkmal ExposureTime ist nicht schreibbar."
        )
        assert (
            "Recording failed: RuntimeError: Camera feature ExposureTime is not writable."
            in window.log.toPlainText()
        )
        assert np.array_equal(clip.frames[1], frames[1])
        if artifact_dir := os.environ.get("BIBAZU_GUI_ARTIFACTS"):
            window.show()
            qt_app.processEvents()
            assert window.grab().save(str(Path(artifact_dir) / "gui_english.png"))
        window.toggle_language()
        assert "Bild 2/3" in window.review_info.text()
        assert f"CSV gespeichert: {csv_path}" in window.log.toPlainText()
        assert window.play_timer.isActive() and window.slider.value() == 1
    finally:
        dispose(window, qt_app)
