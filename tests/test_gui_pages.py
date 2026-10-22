"""The three workspaces share acquisition, but keep their displayed measurements separate."""

import os
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from test_gui_pipeline import dispose, screenshot, wait_for
from test_gui_pipeline import qt_app as qt_app

from bibazu_vision.app import MainWindow
from bibazu_vision.processing import Detection, ProcessingConfig
from bibazu_vision.recording import CaptureStatus, Clip


@pytest.mark.parametrize("size", [(1420, 920), (1280, 800)])
def test_pages_own_their_controls_and_disclosures_remember_session_state(qt_app, tmp_path, size):
    window = MainWindow(tmp_path)
    try:
        window.resize(*size)
        window.show()
        qt_app.processEvents()
        assert window.size().width() == size[0]
        assert window.size().height() == size[1]
        assert window.tabs.count() == 3
        assert window.tabs.currentWidget() is window.live_page
        assert window.threshold.isVisible()
        assert not window.arm_button.isVisible()
        assert not window.recordings.isVisible()
        assert window.processing_group.header.isChecked()
        assert not window.camera_section.header.isChecked()
        assert not window.processing_advanced.header.isChecked()
        assert not window.capture_advanced.header.isChecked()
        assert not window.details_panel.isVisible()
        assert window.live_page.isAncestorOf(window.exposure)
        assert window.live_page.isAncestorOf(window.threshold)
        assert window.capture_page.isAncestorOf(window.arm_button)
        assert window.review_page.isAncestorOf(window.recordings)
        assert not window.live_page.isAncestorOf(window.connect_button)
        assert not window.capture_page.isAncestorOf(window.details_panel)
        QTest.mouseClick(window.camera_section.header, Qt.MouseButton.LeftButton)
        assert window.exposure.isVisible()
        window.exposure.setValue(4321)
        window.processing_advanced.set_expanded(True)
        qt_app.processEvents()
        assert window.live_sidebar.verticalScrollBar().maximum() > 0
        window.tabs.setCurrentWidget(window.capture_page)
        assert window.capture_view.isVisible()
        assert window.arm_button.isVisible()
        assert not window.threshold.isVisible()
        assert not window.trace.isVisible()
        QTest.mouseClick(window.capture_advanced.header, Qt.MouseButton.LeftButton)
        assert window.quiet.isVisible()
        window.tabs.setCurrentWidget(window.review_page)
        assert window.recordings.isVisible()
        assert window.recordings.height() >= 240
        assert window.review_angle_label.isVisible()
        assert not window.angle_label.isVisible()
        assert not window.capture_angle_label.isVisible()
        QTest.mouseClick(window.details_button, Qt.MouseButton.LeftButton)
        assert window.details_panel.isVisible()
        assert window.size().height() == size[1]
        window.tabs.setCurrentWidget(window.live_page)
        assert window.camera_section.header.isChecked()
        assert window.processing_advanced.header.isChecked()
        assert window.exposure.value() == 4321
        window.set_language("en", persist=False)
        assert window.camera_section.title() == "Camera"
        assert window.processing_advanced.title() == "More detection settings"
        assert window.capture_advanced.title() == "More recording settings"
        assert window.details_button.text() == "Log & details"
        assert window.camera_section.header.isChecked()
        assert window.processing_advanced.header.isChecked()
        window.tabs.setCurrentWidget(window.capture_page)
        assert window.capture_advanced.header.isChecked()
    finally:
        dispose(window, qt_app)


def test_recording_and_api_locks_apply_independently_of_selected_page(qt_app, tmp_path):
    window = MainWindow(tmp_path)
    window.timer.stop()
    api_busy = False
    timestamp = time.perf_counter()
    status = CaptureStatus("idle", None, 0, 0, 0, 0, None)
    snapshot = {
        "image": np.full((64, 96), 150, np.uint8),
        "detection": Detection(mask=np.zeros((64, 96), np.uint8)),
        "config": asdict(ProcessingConfig()),
        "has_background": False,
        "status": status,
        "fps": 120,
        "frame_id": 1,
        "frame_gaps": 0,
        "timestamp": timestamp,
        "last_clip": None,
        "stale_seconds": 0,
    }
    connected = threading.Event()
    connected.set()
    window.worker = SimpleNamespace(
        snapshot=lambda: snapshot,
        isRunning=lambda: True,
        demo=False,
        camera_idle=connected,
        api_capture=SimpleNamespace(
            busy=lambda: api_busy,
            history_measurements=lambda *args: {"measurements": [], "overflow": False},
        ),
        camera_settings_available=lambda: not api_busy and status.state == "idle",
    )
    window.trace.reset(timestamp)
    try:
        window.refresh_live()
        assert window.processing_group.isEnabled()
        assert window.capture_group.isEnabled()
        assert window.apply_camera_button.isEnabled()
        status = replace(status, state="waiting")
        snapshot["status"] = status
        window.refresh_live()
        assert not window.processing_group.isEnabled()
        assert not window.apply_camera_button.isEnabled()
        window.tabs.setCurrentWidget(window.capture_page)
        assert window.stop_button.isEnabled()
        api_busy = True
        status = replace(status, state="armed")
        snapshot["status"] = status
        window.refresh_live()
        for page in (window.live_page, window.capture_page, window.review_page):
            window.tabs.setCurrentWidget(page)
            assert not window.processing_group.isEnabled()
            assert not window.capture_group.isEnabled()
            assert not window.apply_camera_button.isEnabled()
        window.set_language("en", persist=False)
        assert "Armed" in window.state_label.text()
        assert "Armed" in window.capture_status.text()
        api_busy = False
        status = replace(status, state="idle")
        snapshot["status"] = status
        window.refresh_live()
        assert window.processing_group.isEnabled()
        assert window.capture_group.isEnabled()
    finally:
        window.worker = None
        dispose(window, qt_app)


def test_recording_continues_across_pages_and_saving_preserves_review(
    qt_app, tmp_path, monkeypatch
):
    window = MainWindow(tmp_path)
    existing = Clip(
        tmp_path / "existing",
        [np.full((480, 800), 160, np.uint8) for _ in range(2)],
        [100, 100.05],
        [1, 2],
        [{"angle_deg": -12}, {"angle_deg": -10}],
        {"stop_reason": "manual_stop"},
    )
    try:
        window.show()
        window.tabs.setCurrentWidget(window.capture_page)
        window.connect_camera(True)
        wait_for(qt_app, lambda: window.last_snapshot is not None)
        window.command("background")
        wait_for(qt_app, lambda: "Hintergrund aufgenommen" in window.log.toPlainText())
        QTest.mouseClick(window.arm_button, Qt.MouseButton.LeftButton)
        wait_for(
            qt_app,
            lambda: (
                window.last_snapshot["detection"].angle_deg is not None
                and not window.last_snapshot["detection"].touching_border
            ),
            timeout=2,
        )
        worker = window.worker
        window.tabs.setCurrentWidget(window.live_page)
        historical = window.trace.samples[-1]
        window.trace.select_at(historical.timestamp - window.trace.origin)
        wait_for(qt_app, lambda: window.live_view.source is not None)
        historical_image = window.live_view.source.cacheKey()
        historical_angle = window.angle_label.text()
        with monkeypatch.context() as patch:
            patch.setattr("bibazu_vision.app.load_clip", lambda path: existing)
            window.open_clip(existing.path)
        window.slider.setValue(1)
        assert window.review_angle_label.text() == "-10.00 °"
        review_image = window.review_view.source.cacheKey()
        # Direct navigation renders the existing snapshot, without adding samples or commands.
        count = len(window.trace.samples)
        with monkeypatch.context() as patch:
            commands = []
            patch.setattr(window, "command", lambda *args: commands.append(args))
            for page in (window.live_page, window.review_page, window.capture_page):
                window.tabs.setCurrentWidget(page)
            assert not commands
        assert len(window.trace.samples) == count
        assert window.worker is worker
        assert window.capture_view.source is not None
        assert "Historie" not in window.capture_angle_label.text()
        wait_for(qt_app, lambda: window.last_snapshot["frame_id"] > historical.frame_id + 10)
        snapshot = window.last_snapshot
        angle = snapshot["detection"].angle_deg
        expected = "— °" if angle is None else f"{angle:+.2f} °"
        assert window.capture_angle_label.text() == expected
        assert window.history_selection is historical
        assert window.live_view.source.cacheKey() == historical_image
        assert window.angle_label.text() == historical_angle
        assert window.clip is existing
        assert window.review_index == 1
        assert window.review_view.source.cacheKey() == review_image
        assert "Aufzeichnung:" in window.state_label.text()
        wait_for(qt_app, lambda: window.last_saved is not None, timeout=4)
        assert window.tabs.currentWidget() is window.capture_page
        assert window.clip is existing and window.review_index == 1
        assert window.open_saved_button.isEnabled()
        assert Path(window.last_saved).name in window.last_saved_label.text()
        QTest.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
        for width, height in ((1420, 920), (1280, 800)):
            window.resize(width, height)
            qt_app.processEvents()
            for name, page in (
                ("live", window.live_page),
                ("capture", window.capture_page),
                ("review", window.review_page),
            ):
                window.tabs.setCurrentWidget(page)
                qt_app.processEvents()
                assert window.width() == width and window.height() == height
                screenshot(window, f"pages_{name}_{width}x{height}.png")
        assert window.review_angle_label.text() == "-10.00 °"
        window.tabs.setCurrentWidget(window.capture_page)
        QTest.mouseClick(window.open_saved_button, Qt.MouseButton.LeftButton)
        assert window.tabs.currentWidget() is window.review_page
        assert window.clip is not existing
        assert window.clip.path == window.last_saved
        assert window.review_index == 0
        assert window.history_selection is historical
        window.select_image("roi")
        assert window.tabs.currentWidget() is window.live_page
        assert window.history_selection is None
        assert window.live_view.mode == "roi"
        window.set_language("en", persist=False)
        assert "Recording:" in window.state_label.text()
        assert "frames" in window.capture_status.text()
    finally:
        dispose(window, qt_app)
