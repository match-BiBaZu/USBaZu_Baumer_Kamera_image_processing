"""Dense camera telemetry, independent previews, and private epoch provenance."""

import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from test_angle_history import rows
from test_gui_pipeline import dispose, wait_for
from test_gui_pipeline import qt_app as qt_app
from test_worker import FakeAcquisitionCamera

from bibazu_vision import worker as worker_module
from bibazu_vision.api_capture import ApiCapture
from bibazu_vision.app import MainWindow
from bibazu_vision.camera import CameraConfig, CameraFrame
from bibazu_vision.history_trace import AngleTrace
from bibazu_vision.processing import Detection, ProcessingConfig
from bibazu_vision.recording import CaptureConfig, CaptureStatus


def test_all_142_measurements_arrive_without_30hz_preview_gate(qt_app, tmp_path):
    window = MainWindow(tmp_path)
    window.timer.stop()
    origin = time.perf_counter()
    connected = threading.Event()
    connected.set()
    capture = SimpleNamespace(status=CaptureStatus("idle", None, 0, 0, 0, 0, None))
    snapshot = None
    worker = SimpleNamespace(
        measurement_epoch=0,
        demo=True,
        camera_idle=connected,
        isRunning=lambda: True,
        snapshot=lambda: snapshot,
        camera_settings_available=lambda: False,
    )
    worker.api_capture = ApiCapture(worker)
    window.worker = worker
    image = np.full((64, 96), 150, np.uint8)

    def send(index):
        nonlocal snapshot
        raw = ((-180 * index / 142 + 90) % 180) - 90
        detection = Detection(
            mask=np.zeros_like(image),
            center=(48, 32),
            area=200,
            confidence=0.4,
            axis_deg=raw,
            angle_deg=raw,
        )
        timestamp = origin + index / 142
        worker.api_capture.tick(capture, CameraFrame(image, timestamp, index + 1, False), detection)
        snapshot = {
            "image": image,
            "detection": detection,
            "timestamp": timestamp,
            "frame_id": index + 1,
            "status": capture.status,
            "fps": 142,
            "frame_gaps": 0,
            "last_clip": None,
            "config": asdict(ProcessingConfig()),
            "has_background": False,
            "stale_seconds": 0,
        }

    try:
        window.start_history()
        window.trace.reset(origin)
        assert window.timer.interval() == 17
        assert window.timer.timerType() == Qt.TimerType.PreciseTimer
        window.tabs.setCurrentWidget(window.capture_page)
        for index in range(142):
            send(index)
        window.refresh_live()
        assert len(window.trace.measurements) == 142
        assert window.trace.measurements.cursor == 142
        assert len(window.trace.samples) == 1
        assert window.trace.measurements.last.continuous_angle == pytest.approx(-180 * 141 / 142)
        historical = window.trace.samples[0]
        window.trace.select_at(historical.timestamp - origin)
        wait_for(qt_app, lambda: window.live_view.source is not None)
        pinned_image = window.live_view.source.cacheKey()
        assert "Fortlaufend -178.73°" in window.history_status.text()
        assert window.trace.measurements.for_preview(historical).raw_angle == historical.angle
        for index in range(142, 146):
            send(index)
            window.refresh_live()
        # These frames are only 7 ms apart: the previous extra gate dropped them.
        assert len(window.trace.samples) == 5
        assert len(window.trace.measurements) == 146
        assert window.live_view.source.cacheKey() == pinned_image
        assert window.history_selection == historical
        assert window.capture_angle_label.text() == f"{snapshot['detection'].angle_deg:+.2f} °"
        window.refresh_live()
        assert len(window.trace.samples) == 5 and len(window.trace.measurements) == 146
        window.set_language("en", persist=False)
        assert "Continuous" in window.history_status.text()
        assert "measurements retained" in window.diagnostics_label.text()
        public = worker.api_capture.measurements(0, 2048)["measurements"]
        assert all("epoch" not in row and "_history_epoch" not in row for row in public)
        assert public[-1]["angle_deg"] == snapshot["detection"].angle_deg
    finally:
        window.worker = None
        dispose(window, qt_app)


def test_dense_trace_snaps_only_to_preview_and_marks_hover_reasons(qt_app):
    trace = AngleTrace()
    try:
        trace.resize(1100, 180)
        trace.reset(100)
        angles = [
            (((-180 * min(1, max(0, i / 142 - 1) / 0.6)) + 90) % 180) - 90 for i in range(568)
        ]
        batch = rows(angles)
        batch[400].update(angle_deg=None, axis_deg=0, touching_border=True)
        trace.add_measurements({"measurements": batch, "overflow": False})
        for index in range(0, len(batch), 3):
            row = batch[index]
            trace.add(row["timestamp"], row["frame_id"], row["angle_deg"])
        trace.set_span(5)
        trace.show()
        qt_app.processEvents()
        trace.select_at(1.33)
        sample = trace.selection
        assert sample in trace.samples
        assert trace.measurements.for_preview(sample).continuous_angle == pytest.approx(
            -180 * (sample.timestamp - 101) / 0.6
        )
        rect = trace.plot.plot_rect()
        point = QPoint(
            round(rect.left() + (400 / 142) / 5 * rect.width()), round(rect.center().y())
        )
        trace.language = "en"
        QTest.mouseMove(trace.plot, point)
        assert "Border contact" in trace.plot.toolTip()
        low, high, _ = trace.plot.axis_limits(trace.plot.view_data()[0])
        assert low < -180 and high > 0
        artifact_dir = os.environ.get("BIBAZU_GUI_ARTIFACTS")
        if artifact_dir:
            assert trace.grab().save(str(Path(artifact_dir) / "continuous_flip.png"))
    finally:
        trace.close()


def test_worker_epochs_match_accepted_processing_commands_without_api_changes(
    qt_app, tmp_path, monkeypatch
):
    cameras = []

    def factory(config):
        camera = FakeAcquisitionCamera(config)
        cameras.append(camera)
        return camera

    monkeypatch.setattr(worker_module, "BaumerCamera", factory)
    worker = worker_module.PipelineWorker(
        CameraConfig(), ProcessingConfig(), CaptureConfig(), tmp_path
    )
    worker.start()

    def frame(number):
        cameras[0].send_frame()
        wait_for(
            qt_app,
            lambda: worker.snapshot() is not None and worker.snapshot()["frame_id"] == number,
        )

    try:
        wait_for(qt_app, lambda: bool(cameras))
        frame(1)
        worker.command("config", {"threshold": 100})
        wait_for(qt_app, lambda: worker.measurement_epoch == 1)
        frame(2)
        assert worker.measurement_epoch == 1
        worker.command("zero")
        wait_for(qt_app, lambda: worker.measurement_epoch == 2)
        frame(3)
        assert worker.measurement_epoch == 2
        worker.command("background")
        wait_for(qt_app, lambda: worker.measurement_epoch == 3)
        frame(4)
        assert worker.measurement_epoch == 3
        assert worker.snapshot()["detection"].angle_deg is None
        worker.command("zero")  # Invalid detection: rejected zero must not create a new epoch.
        frame(5)
        assert worker.measurement_epoch == 3
        internal = worker.api_capture.history_measurements(0, 2048)["measurements"]
        assert [row["epoch"] for row in internal] == [0, 1, 2, 3, 3]
        public = worker.api_capture.measurements(0, 2048)["measurements"]
        assert all("epoch" not in row and "_history_epoch" not in row for row in public)
    finally:
        worker.stop()
        assert worker.wait(3000)


def test_ten_minute_plot_remains_responsive_and_preserves_extrema(qt_app):
    trace = AngleTrace()
    try:
        trace.resize(1100, 180)
        trace.reset(100)
        angles = [0] * 85201
        angles[42001], angles[42002] = 40, -40
        trace.add_measurements({"measurements": rows(angles), "overflow": False})
        trace.set_span(600)
        trace.show()
        started = time.perf_counter()
        pixmap = trace.grab()
        elapsed = time.perf_counter() - started
        assert elapsed < 0.5, f"Painting the full history took {elapsed:.3f}s"
        points, _ = trace.plot.view_data()
        assert len(points) <= 4 * 1100
        assert min(row.continuous_angle for row in points) == -40
        assert max(row.continuous_angle for row in points) == 40
        artifact_dir = os.environ.get("BIBAZU_GUI_ARTIFACTS")
        if artifact_dir:
            assert pixmap.save(str(Path(artifact_dir) / "full_history.png"))
    finally:
        trace.close()
