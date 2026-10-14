"""HTTP + Qt pipeline integration using a synthetic queued camera only."""

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PyQt6.QtWidgets import QApplication, QWidget
from test_worker import FakeAcquisitionCamera

from bibazu_vision import worker as worker_module
from bibazu_vision.api import VisionApi
from bibazu_vision.api_capture import ApiCapture
from bibazu_vision.camera import CameraConfig
from bibazu_vision.local_api_runtime import SessionLease
from bibazu_vision.processing import ProcessingConfig
from bibazu_vision.recording import CaptureConfig, CaptureManager


@pytest.fixture
def harness(tmp_path, monkeypatch):
    qt = QApplication.instance() or QApplication([])
    created = []
    monkeypatch.setattr(
        worker_module,
        "BaumerCamera",
        lambda config: created.append(FakeAcquisitionCamera(config)) or created[-1],
    )
    workers = []
    pools = []
    clients = []
    windows = []

    def wait(predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            qt.processEvents()
            if predicate():
                return
            time.sleep(0.002)
        raise AssertionError("condition timed out")

    def start(mode):
        worker = worker_module.PipelineWorker(
            CameraConfig(mode="live"),
            ProcessingConfig(threshold=100, min_area=20),
            CaptureConfig(
                start_mode=mode,
                pre_roll_seconds=0,
                quiet_seconds=0.03,
                wait_seconds=1,
                max_seconds=2,
            ),
            tmp_path / mode,
        )
        workers.append(worker)
        worker.start()
        wait(lambda: len(created) == len(workers))
        camera = created[-1]
        camera.items.put((np.full((64, 96), 185, dtype=np.uint8), False))
        wait(lambda: worker.snapshot() is not None)
        window = QWidget()
        window.worker = worker
        windows.append(window)
        api = VisionApi(window)
        client = TestClient(api.app)
        pool = ThreadPoolExecutor(1)
        pools.append(pool)
        clients.append(client)

        def request(method, path, **kwargs):
            future = pool.submit(client.request, method, "/api/v1/" + path, **kwargs)
            wait(future.done)
            return future.result()

        session = request("POST", "sessions", json={})
        assert session.status_code == 201, session.text
        return worker, camera, api, request, {"X-Session-Id": session.json()["session_id"]}

    yield start, wait
    for worker in workers:
        worker.stop()
        assert worker.wait(3000)
    for pool in pools:
        pool.shutdown()
    for client in clients:
        client.close()
    for window in windows:
        window.deleteLater()
    qt.processEvents()


def blank(camera):
    camera.items.put((np.full((64, 96), 185, dtype=np.uint8), False))


@pytest.mark.parametrize("mode", ["motion", "hardware", "manual"])
def test_trial_single_shot_ids_full_measurements_and_locked_reference(harness, mode):
    start, wait = harness
    worker, camera, api, request, headers = start(mode)
    body = {"run_id": "run-1", "trial_id": "trial-1", "context": {"part_name": "Dk1i"}}
    response = request("POST", "trials", json=body, headers=headers)
    assert response.status_code == 202, response.text
    wait(lambda: worker.api_capture.trials["trial-1"]["status"] == "armed")
    worker.command("config", {"threshold": 200})
    worker.command("capture_config", {"start_mode": "manual"})
    if mode == "manual":
        assert request("POST", "trials/trial-1/trigger", headers=headers).status_code == 202
    camera.send_frame(x=20, triggered=mode == "hardware")
    wait(lambda: worker.snapshot()["status"].state in {"waiting", "moving"})
    camera.send_frame(x=25)
    blank(camera)
    time.sleep(0.06)
    blank(camera)
    wait(lambda: not worker.api_capture.busy())
    result = request("GET", "trials/trial-1").json()
    assert result["status"] == "completed", result
    assert result["manifest"]["metadata"]["run_id"] == "run-1"
    assert result["manifest"]["metadata"]["trial_id"] == "trial-1"
    assert result["manifest"]["metadata"]["processing"]["threshold"] == 100
    assert len(result["measurements"]) == result["manifest"]["frame_count"]
    assert worker.snapshot()["status"].state == "idle"
    previous = worker.snapshot()["last_clip"]
    camera.send_frame(x=20, triggered=True)
    blank(camera)
    time.sleep(0.06)
    blank(camera)
    wait(lambda: camera.items.empty())
    assert worker.snapshot()["last_clip"] == previous
    assert request("POST", "trials", json=body, headers=headers).status_code == 409
    assert request("GET", "trials/missing").status_code == 404


def test_session_expiry_aborts_active_capture_and_reports_buffer_loss(harness):
    start, wait = harness
    worker, camera, api, request, headers = start("hardware")
    worker.api_capture.samples = deque(maxlen=2)
    request("POST", "trials", json={"run_id": "r", "trial_id": "t"}, headers=headers)
    wait(lambda: worker.api_capture.trials["t"]["status"] == "armed")
    camera.send_frame(triggered=True)
    camera.send_frame(x=25)
    camera.send_frame(x=30)
    wait(lambda: worker.api_capture.sequence >= 4)
    batch = request("GET", "measurements?after=0").json()
    assert batch["overflow"]
    assert len(batch["measurements"]) == 2
    assert "valid" in batch["measurements"][0]
    api.lease.invalidate()
    wait(lambda: not worker.api_capture.busy())
    result = request("GET", "trials/t").json()
    assert result["status"] == "aborted"
    assert result["manifest"]["aborted"]
    assert result["manifest"]["stop_reason"] == "session_expired"
    assert request("POST", "trials/t/trigger", headers=headers).status_code == 409


def test_writer_failure_is_not_completed(harness, monkeypatch):
    start, wait = harness
    worker, camera, api, request, headers = start("manual")

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(CaptureManager, "_persist", fail)
    request("POST", "trials", json={"run_id": "r", "trial_id": "t"}, headers=headers)
    wait(lambda: worker.api_capture.trials["t"]["status"] == "armed")
    request("POST", "trials/t/trigger", headers=headers)
    camera.send_frame()
    camera.send_frame(x=25)
    blank(camera)
    time.sleep(0.06)
    blank(camera)
    wait(lambda: not worker.api_capture.busy())
    result = request("GET", "trials/t").json()
    assert result["status"] == "failed"
    assert "disk full" in result["error"]
    assert result["clip_path"] is None


def test_abort_while_armed_and_request_validation(harness):
    start, wait = harness
    worker, camera, api, request, headers = start("motion")
    assert (
        request("POST", "trials", json={"run_id": "", "trial_id": "t"}, headers=headers).status_code
        == 422
    )
    assert request("GET", "measurements?after=-1").status_code == 422
    request("POST", "trials", json={"run_id": "r", "trial_id": "t"}, headers=headers)
    wait(lambda: worker.api_capture.trials["t"]["status"] == "armed")
    assert request("POST", "trials/t/trigger", headers=headers).status_code == 409
    assert request("POST", "trials/t/abort", headers=headers).status_code == 202
    wait(lambda: not worker.api_capture.busy())
    assert request("GET", "trials/t").json()["status"] == "aborted"


def test_completion_between_worker_ticks_is_still_reported(tmp_path):
    config = CaptureConfig(start_mode="manual", wait_seconds=0.01, max_seconds=0.02)
    worker = SimpleNamespace(command=lambda *args: True, capture_config=config)
    api = ApiCapture(worker)
    lease = SessionLease()
    token = lease.acquire()["session_id"]
    request = {
        "run_id": "r",
        "trial_id": "t",
        "context": {},
        "timeout_seconds": 60,
        "capture_config": asdict(config),
    }
    capture = CaptureManager(config, tmp_path)
    try:
        api.create(request, lease, token)
        api.start(capture, request)
        api.trigger(capture)
        capture.poll(time.perf_counter() + 1)
        assert capture.wait_until_saved(2)
        api.tick(capture)
        result = api.get("t")
        assert result["status"] == "failed"
        assert result["clip_path"] is not None
        assert result["manifest"]["frame_count"] == 0
    finally:
        capture.close()


def test_latest_gui_capture_mode_and_results_after_disconnect(harness):
    start, wait = harness
    worker, camera, api, request, headers = start("hardware")
    api.window.capture_config = lambda: CaptureConfig(start_mode="manual", max_seconds=2)
    request("POST", "trials", json={"run_id": "r", "trial_id": "gui-mode"}, headers=headers)
    wait(lambda: worker.api_capture.trials["gui-mode"]["status"] == "armed")
    assert worker.capture_config.start_mode == "manual"
    assert request("POST", "trials/gui-mode/trigger", headers=headers).status_code == 202
    camera.send_frame()
    wait(lambda: worker.snapshot()["status"].state in {"waiting", "moving"})
    worker.stop()
    assert worker.wait(3000)
    result = request("GET", "trials/gui-mode").json()
    assert result["status"] == "aborted"
    assert result["manifest"]["metadata"]["trial_id"] == "gui-mode"
    assert result["manifest"]["capture_config"]["start_mode"] == "manual"
