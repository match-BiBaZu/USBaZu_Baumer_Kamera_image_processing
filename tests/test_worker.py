"""Native AcquisitionStart lifecycle through the real Qt worker and writer."""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import asdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from bibazu_vision import worker as worker_module
from bibazu_vision.camera import CameraConfig, CameraFrame
from bibazu_vision.processing import ProcessingConfig
from bibazu_vision.recording import CaptureConfig, CaptureManager, load_clip


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def wait_for(application, predicate, timeout=2.0, description="condition"):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        application.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    raise AssertionError(f"Timed out waiting for {description}")


class FakeAcquisitionCamera:
    """Pending hardware acquisition yields no frames until a test sends one."""

    def __init__(self, config):
        self.config = config
        self.diagnostics = {"model": "Native AcquisitionStart test", "mode": config.mode}
        self.opened = threading.Event()
        self.closed = threading.Event()
        self.items = queue.Queue()
        self.rearm_count = 0
        self.read_count = 0
        self.frame_id = 0

    def open(self):
        self.opened.set()

    def rearm_acquisition(self):
        self.rearm_count += 1

    def read(self, timeout=0.2):
        self.read_count += 1
        try:
            value = self.items.get(timeout=min(timeout, 0.01))
        except queue.Empty:
            return None
        if isinstance(value, Exception):
            raise value
        image, triggered = value
        self.frame_id += 1
        return CameraFrame(image, time.perf_counter(), self.frame_id, triggered)

    def send_frame(self, x=20, triggered=False):
        image = np.full((64, 96), 185, dtype=np.uint8)
        image[25:35, x : x + 30] = 30
        self.items.put((image, triggered))

    def close(self):
        self.closed.set()


@pytest.fixture
def worker_harness(qt_app, tmp_path, monkeypatch):
    workers = []
    cameras = []

    def camera_factory(config):
        camera = FakeAcquisitionCamera(config)
        cameras.append(camera)
        return camera

    monkeypatch.setattr(worker_module, "BaumerCamera", camera_factory)

    def start(*, camera_mode="acquisition_trigger", commands=("arm",), **capture_values):
        options = {
            "start_mode": "hardware",
            "pre_roll_seconds": 0,
            "quiet_seconds": 0.03,
            "wait_seconds": 0.5,
            "max_seconds": 2,
        }
        options.update(capture_values)
        processor_config = ProcessingConfig(threshold=100, min_area=20)
        worker = worker_module.PipelineWorker(
            CameraConfig(mode=camera_mode),
            processor_config,
            CaptureConfig(**options),
            tmp_path / f"worker_{len(workers)}",
        )
        failures, messages = [], []
        worker.failed.connect(failures.append)
        worker.message.connect(messages.append)
        workers.append(worker)
        for command in commands:
            worker.command(command)
        worker.start()
        wait_for(qt_app, lambda: len(cameras) == len(workers), description="camera construction")
        camera = cameras[-1]
        wait_for(qt_app, camera.opened.is_set, description="camera open")
        return worker, camera, failures, messages

    yield start
    for worker in workers:
        worker.stop()
        assert worker.wait(2500), "worker did not close camera and flush writer"
    qt_app.processEvents()


def state(worker):
    snapshot = worker.snapshot()
    return snapshot["status"].state if snapshot is not None else None


def completed_clip(application, worker):
    wait_for(
        application,
        lambda: worker.snapshot() is not None and worker.snapshot()["last_clip"] is not None,
        description="asynchronous recording completion",
    )
    return load_clip(worker.snapshot()["last_clip"])


@pytest.mark.parametrize("camera_mode,expected_rearms", [("acquisition_trigger", 1), ("live", 0)])
def test_arm_rearms_only_native_acquisition_and_pending_read_stays_alive(
    qt_app, worker_harness, camera_mode, expected_rearms
):
    worker, camera, failures, _ = worker_harness(camera_mode=camera_mode)
    wait_for(qt_app, lambda: camera.read_count > 3, description="pending camera reads")
    assert camera.rearm_count == expected_rearms
    assert worker.snapshot() is None
    assert worker.isRunning()
    assert failures == []


def test_arm_during_active_capture_does_not_restart_native_acquisition(qt_app, worker_harness):
    worker, camera, failures, _ = worker_harness()
    wait_for(qt_app, lambda: camera.rearm_count == 1)
    camera.send_frame(triggered=True)
    wait_for(qt_app, lambda: state(worker) == "waiting", description="triggered recording")
    reads = camera.read_count
    worker.command("arm")
    wait_for(qt_app, lambda: worker.commands.empty() and camera.read_count > reads + 2)
    assert state(worker) == "waiting"
    assert camera.rearm_count == 1
    assert failures == []


def test_no_frames_poll_timeout_rearms_native_acquisition_once(qt_app, worker_harness):
    worker, camera, failures, _ = worker_harness(wait_seconds=0.1)
    wait_for(qt_app, lambda: camera.rearm_count == 1)
    camera.send_frame(triggered=True)
    wait_for(qt_app, lambda: state(worker) == "waiting", description="first trigger frame")
    wait_for(qt_app, lambda: camera.rearm_count == 2, description="timeout native rearm")
    clip = completed_clip(qt_app, worker)
    assert clip.manifest["stop_reason"] == "no_motion_timeout"
    assert clip.manifest["motion_seen"] is False
    assert clip.manifest["trigger_source"] == "hardware"
    assert clip.frame_ids == [1]
    reads = camera.read_count
    wait_for(qt_app, lambda: camera.read_count > reads + 3)
    assert camera.rearm_count == 2
    assert state(worker) == "armed"
    assert failures == []


def test_disarm_moving_capture_rearms_camera_and_flushes_manual_stop(qt_app, worker_harness):
    worker, camera, failures, _ = worker_harness()
    wait_for(qt_app, lambda: camera.rearm_count == 1)
    camera.send_frame(triggered=True)
    wait_for(qt_app, lambda: state(worker) == "waiting")
    camera.send_frame(x=35)
    wait_for(qt_app, lambda: state(worker) == "moving", description="part motion")
    worker.command("disarm")
    wait_for(qt_app, lambda: camera.rearm_count == 2, description="disarm native rearm")
    clip = completed_clip(qt_app, worker)
    assert clip.manifest["stop_reason"] == "manual_stop"
    assert clip.manifest["motion_seen"] is True
    assert clip.frame_ids == [1, 2]
    assert state(worker) == "idle"
    assert failures == []


def test_processing_changes_blocked_midcapture_keep_saved_reference_stable(qt_app, worker_harness):
    worker, camera, failures, messages = worker_harness()
    original_config = asdict(worker.processing_config)
    wait_for(qt_app, lambda: camera.rearm_count == 1)
    camera.send_frame(triggered=True)
    wait_for(qt_app, lambda: state(worker) == "waiting")
    worker.command("config", {"threshold": 250, "zero_offset_deg": 37})
    for command in ("background", "clear_background", "zero"):
        worker.command(command)
    wait_for(
        qt_app,
        lambda: sum("Messreferenz erst" in message for message in messages) == 4,
        description="all reference changes rejected during active capture",
    )
    assert asdict(worker.processing_config) == original_config
    assert not (worker.data_dir / "background.png").exists()
    camera.send_frame()
    wait_for(qt_app, lambda: worker.snapshot()["frame_id"] == 2)
    assert worker.snapshot()["config"] == original_config
    worker.command("disarm")
    clip = completed_clip(qt_app, worker)
    assert clip.manifest["metadata"]["processing"] == original_config
    assert len(clip.measurements) == 2
    assert all(measurement["angle_deg"] == pytest.approx(0) for measurement in clip.measurements)
    worker.command("config", {"threshold": 90})
    wait_for(qt_app, lambda: worker.commands.empty())
    camera.send_frame()
    wait_for(qt_app, lambda: worker.snapshot()["frame_id"] == 3)
    assert worker.snapshot()["config"]["threshold"] == 90
    assert clip.manifest["metadata"]["processing"]["threshold"] == 100
    assert failures == []


def test_capture_configuration_error_is_reported_without_stopping_acquisition(
    qt_app, worker_harness
):
    worker, camera, failures, messages = worker_harness()
    wait_for(qt_app, lambda: camera.rearm_count == 1)
    camera.send_frame(triggered=True)
    wait_for(qt_app, lambda: state(worker) == "waiting")
    worker.command("capture_config", {"start_mode": "manual"})
    wait_for(qt_app, lambda: any("Aufzeichnungseinstellungen:" in item for item in messages))
    assert state(worker) == "waiting"
    assert worker.isRunning()
    assert failures == []


def test_manual_trigger_before_first_frame_snapshots_processing_and_camera_metadata(
    qt_app, worker_harness
):
    worker, camera, failures, _ = worker_harness(commands=("trigger",))
    wait_for(qt_app, lambda: camera.read_count > 1 and worker.commands.empty())
    camera.send_frame()
    wait_for(qt_app, lambda: state(worker) == "waiting")
    worker.command("disarm")
    clip = completed_clip(qt_app, worker)
    assert clip.manifest["metadata"]["processing"] == asdict(worker.processing_config)
    assert clip.manifest["metadata"]["camera"] == camera.diagnostics
    assert clip.manifest["metadata"]["demo"] is False
    assert failures == []


def test_read_failure_emits_error_closes_camera_and_preserves_active_clip(qt_app, worker_harness):
    worker, camera, failures, _ = worker_harness()
    wait_for(qt_app, lambda: camera.rearm_count == 1)
    camera.send_frame(triggered=True)
    wait_for(qt_app, lambda: state(worker) == "waiting")
    camera.items.put(OSError("USB disconnected"))
    wait_for(qt_app, lambda: not worker.isRunning(), description="worker error shutdown")
    qt_app.processEvents()
    assert failures == ["OSError: USB disconnected"]
    assert camera.closed.is_set()
    sessions = list((worker.data_dir / "recordings").glob("session_*"))
    assert len(sessions) == 1
    clip = load_clip(sessions[0])
    assert clip.manifest["stop_reason"] == "closed"
    assert clip.frame_ids == [1]


def test_trigger_and_immediate_memory_abort_still_rearms_native_acquisition(qt_app, worker_harness):
    worker, camera, failures, _ = worker_harness(max_memory_mb=0.001)
    wait_for(qt_app, lambda: camera.rearm_count == 1)
    camera.send_frame(triggered=True)
    clip = completed_clip(qt_app, worker)
    assert clip.manifest["stop_reason"] == "memory_limit"
    assert clip.manifest["dropped_frames"] == 1
    assert clip.frames == []
    assert camera.rearm_count == 2
    assert failures == []


def test_unarmed_hardware_frames_rearm_without_starting_a_clip(qt_app, worker_harness):
    worker, camera, failures, _ = worker_harness(commands=())
    wait_for(qt_app, lambda: camera.read_count > 2)
    assert camera.rearm_count == 0
    camera.send_frame(triggered=True)
    wait_for(qt_app, lambda: camera.rearm_count == 1, description="unarmed trigger rearm")
    assert state(worker) == "idle"
    assert worker.snapshot()["last_clip"] is None
    camera.send_frame(triggered=True)
    wait_for(qt_app, lambda: camera.rearm_count == 2)
    assert state(worker) == "idle"
    assert not list((worker.data_dir / "recordings").glob("session_*"))
    assert failures == []


def test_writer_busy_trigger_rejection_rearms_and_retains_original_clip(
    qt_app, worker_harness, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    persist = CaptureManager._persist

    def blocked_writer(self, capture):
        entered.set()
        assert release.wait(5)
        return persist(self, capture)

    monkeypatch.setattr(CaptureManager, "_persist", blocked_writer)
    worker, camera, failures, _ = worker_harness(writer_queue_depth=1)
    try:
        wait_for(qt_app, lambda: camera.rearm_count == 1)
        camera.send_frame(triggered=True)
        wait_for(qt_app, lambda: state(worker) == "waiting")
        worker.command("disarm")
        wait_for(qt_app, lambda: camera.rearm_count == 2 and entered.is_set())
        worker.command("arm")
        wait_for(qt_app, lambda: camera.rearm_count == 3)
        camera.send_frame(triggered=True)
        wait_for(qt_app, lambda: worker.snapshot()["frame_id"] == 2)
        assert worker.snapshot()["status"].reason == "writer_busy"
        assert worker.snapshot()["status"].rejected_triggers == 1
        assert camera.rearm_count == 4
        assert worker.isRunning()
        assert failures == []
    finally:
        release.set()
    clip = completed_clip(qt_app, worker)
    assert clip.manifest["stop_reason"] == "manual_stop"
    assert clip.frame_ids == [1]


def track_recorders(monkeypatch):
    recorders = []

    def capture_factory(*args, **kwargs):
        manager = CaptureManager(*args, **kwargs)
        recorders.append(manager)
        return manager

    monkeypatch.setattr(worker_module, "CaptureManager", capture_factory)
    return recorders


def test_disk_full_is_reported_without_new_frames_and_camera_remains_live(
    qt_app, worker_harness, monkeypatch
):
    recorders = track_recorders(monkeypatch)

    def disk_full(_self, _capture):
        raise OSError("disk full")

    monkeypatch.setattr(CaptureManager, "_persist", disk_full)
    worker, camera, failures, _ = worker_harness(
        camera_mode="live", commands=("trigger",), start_mode="manual"
    )
    recording_errors = []
    worker.recording_failed.connect(recording_errors.append)
    wait_for(qt_app, lambda: worker.commands.empty() and camera.read_count > 1)
    camera.send_frame()
    wait_for(qt_app, lambda: state(worker) == "waiting")
    worker.command("disarm")
    wait_for(
        qt_app,
        lambda: bool(recording_errors),
        description="disk-full recording error while camera has no new frame",
    )
    assert recording_errors == ["OSError: disk full"]
    assert worker.isRunning()
    assert not camera.closed.is_set()
    assert failures == []
    assert recorders[0].status.writer_pending == 0
    assert recorders[0].status.buffered_bytes == 0
    assert recorders[0].completed_clips[-1].error == "OSError: disk full"
    assert worker.snapshot()["last_clip"] is None

    camera.send_frame(x=35)
    wait_for(qt_app, lambda: worker.snapshot()["frame_id"] == 2)
    assert worker.isRunning()
    assert failures == []
    worker.stop()
    assert worker.wait(2500)
    qt_app.processEvents()
    assert camera.closed.is_set()
    assert not recorders[0]._thread.is_alive()
    assert recorders[0].status.writer_pending == 0
    assert failures == []


def test_shutdown_waits_for_writer_and_surfaces_its_disk_full_error(
    qt_app, worker_harness, monkeypatch
):
    recorders = track_recorders(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def slow_disk_full(_self, _capture):
        entered.set()
        assert release.wait(5)
        raise OSError("disk full")

    monkeypatch.setattr(CaptureManager, "_persist", slow_disk_full)
    worker, camera, failures, _ = worker_harness(
        camera_mode="live", commands=("trigger",), start_mode="manual"
    )
    recording_errors = []
    worker.recording_failed.connect(recording_errors.append)
    try:
        wait_for(qt_app, lambda: worker.commands.empty() and camera.read_count > 1)
        camera.send_frame()
        wait_for(qt_app, lambda: state(worker) == "waiting")
        worker.command("disarm")
        wait_for(qt_app, entered.is_set, description="pending writer")
        worker.stop()
        wait_for(qt_app, camera.closed.is_set, description="camera closed before writer flush")
        assert worker.isRunning()
        assert recorders[0]._thread.is_alive()
        assert recorders[0].status.writer_pending == 1
        assert failures == []
    finally:
        release.set()
    wait_for(qt_app, lambda: not worker.isRunning(), description="writer-aware worker shutdown")
    assert recording_errors == ["OSError: disk full"]
    assert not recorders[0]._thread.is_alive()
    assert recorders[0].status.writer_pending == 0
    assert recorders[0].status.buffered_bytes == 0
    assert failures == []
