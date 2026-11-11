"""Presentation limits must not thin camera measurements or original recordings."""

import json
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from test_gui_pipeline import dispose, wait_for
from test_gui_pipeline import qt_app as qt_app
from test_worker import FakeAcquisitionCamera

from bibazu_vision import worker as worker_module
from bibazu_vision.api_capture import ApiCapture
from bibazu_vision.app import MainWindow
from bibazu_vision.camera import CameraFrame
from bibazu_vision.processing import Detection, ProcessingConfig
from bibazu_vision.recording import CaptureStatus, load_clip


@pytest.fixture
def presentation_harness(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("BIBAZU_API_DISABLED", "1")
    window = MainWindow(tmp_path)
    window.timer.stop()
    origin = time.perf_counter()
    connected = threading.Event()
    connected.set()
    capture = SimpleNamespace(status=CaptureStatus("idle", None, 0, 0, 0, 0))
    snapshot = None
    worker = SimpleNamespace(
        measurement_epoch=0, demo=True, camera_idle=connected,
        isRunning=lambda: True, snapshot=lambda: snapshot,
        camera_settings_available=lambda: False,
    )
    worker.api_capture = ApiCapture(worker)
    window.worker = worker
    window.start_history()
    archive_calls, display_calls = [], []
    monkeypatch.setattr(
        window.history_archive, "submit",
        lambda sample, image: archive_calls.append(sample) or True,
    )
    monkeypatch.setattr(
        window.live_view, "set_image", lambda image: display_calls.append(snapshot["frame_id"]),
    )
    image = np.full((64, 96), 150, np.uint8)

    def send(index):
        nonlocal snapshot
        detection = Detection(mask=np.zeros_like(image), axis_deg=12, angle_deg=12)
        timestamp = origin + index / 200
        worker.api_capture.tick(capture, CameraFrame(image, timestamp, index, False), detection)
        snapshot = {
            "image": image, "detection": detection, "timestamp": timestamp, "frame_id": index,
            "status": capture.status, "fps": 200, "frame_gaps": 0, "last_clip": None,
            "config": asdict(ProcessingConfig()), "has_background": False, "stale_seconds": 0,
        }
        window.refresh_live()

    yield window, send, display_calls, archive_calls
    window.worker = None
    dispose(window, qt_app)


@pytest.mark.parametrize("live_fps,history_fps,display_count,history_count", [
    (10, 5, 10, 5), (5, 20, 5, 20), (0, -1, 0, 0), (0, 0, 0, 200),
])
def test_display_and_history_limits_preserve_every_measurement(
    presentation_harness, live_fps, history_fps, display_count, history_count,
):
    window, send, displayed, archived = presentation_harness
    window.preview_fps.setCurrentIndex(window.preview_fps.findData(live_fps))
    window.history_fps.setCurrentIndex(window.history_fps.findData(history_fps))
    for index in range(200):
        send(index)
    assert len(displayed) == display_count
    assert len(archived) == history_count
    assert len(window.trace.samples) == history_count
    assert len(window.trace.measurements) == 200
    assert len(window.worker.api_capture.measurements(0, 8192)["measurements"]) == 200
    assert window.angle_label.text() == "+12.00 °"


def test_preview_preferences_are_global_and_do_not_replace_native_calibration(
    qt_app, tmp_path, monkeypatch,
):
    monkeypatch.setenv("BIBAZU_API_DISABLED", "1")
    path = tmp_path / "settings.json"
    original = {"language": "de", "processing": {"zero_offset_deg": 5.25}}
    path.write_text(json.dumps(original), encoding="utf-8")
    window = MainWindow(tmp_path)
    try:
        window.preview_fps.setCurrentIndex(window.preview_fps.findData(20))
        window.history_fps.setCurrentIndex(window.history_fps.findData(-1))
        assert json.loads(path.read_text(encoding="utf-8")) == {
            **original, "preview": {"live_fps": 20, "history_fps": -1, "gap_ms": 100},
        }
        window.set_language("en", persist=False)
        assert window.preview_section.title() == "Preview & image history"
        assert window.timed_button.text() == "Record 1 minute"
    finally:
        dispose(window, qt_app)
    reopened = MainWindow(tmp_path)
    try:
        assert reopened.preview_settings() == {"live_fps": 20, "history_fps": -1, "gap_ms": 100}
        assert reopened.camera_profiles["usb"]["processing"]["zero_offset_deg"] == 5.25
    finally:
        dispose(reopened, qt_app)


def test_minute_button_and_early_stop_keep_originals_when_all_previews_are_off(
    qt_app, tmp_path, monkeypatch,
):
    monkeypatch.setenv("BIBAZU_API_DISABLED", "1")
    cameras = []

    def factory(config):
        camera = FakeAcquisitionCamera(config)
        cameras.append(camera)
        return camera

    monkeypatch.setattr(worker_module, "BaumerCamera", factory)
    window = MainWindow(tmp_path)
    try:
        window.preview_fps.setCurrentIndex(window.preview_fps.findData(0))
        window.history_fps.setCurrentIndex(window.history_fps.findData(-1))
        window.threshold.setValue(100)
        window.connect_camera(False)
        wait_for(qt_app, lambda: bool(cameras))
        cameras[0].send_frame()
        wait_for(qt_app, lambda: window.last_snapshot is not None)
        assert window.timed_button.isEnabled()
        window.timed_button.click()
        wait_for(qt_app, lambda: "Zeitaufnahme gestartet" in window.log.toPlainText())
        for index in range(3):
            cameras[0].send_frame(x=20 + index * 5)
            wait_for(
                qt_app, lambda number=index + 2: window.worker.snapshot()["frame_id"] == number
            )
        assert window.worker.snapshot()["status"].timed_active
        assert not window.timed_button.isEnabled()
        assert not window.apply_camera_button.isEnabled()
        assert "Zeitaufnahme" in window.capture_status.text()
        window.stop_button.click()
        wait_for(qt_app, lambda: window.worker.snapshot()["last_clip"] is not None)
        clip = load_clip(window.worker.snapshot()["last_clip"])
        assert clip.manifest["capture_config"]["duration_seconds"] == 60
        assert clip.manifest["stop_reason"] == "manual_stop"
        assert clip.frame_ids == [2, 3, 4]
        assert all(row["angle_deg"] is not None for row in clip.measurements)
        assert all(pixels.shape == (64, 96) for pixels in clip.frames)
        assert not window.trace.samples
        assert window.live_view.source is None
        window.open_clip(clip.path)
        assert window.review_view.source is not None
    finally:
        dispose(window, qt_app)
