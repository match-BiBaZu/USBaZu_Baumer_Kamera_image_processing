"""Exercise the API installed by the actual camera window, without camera hardware."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from bibazu_vision.app import MainWindow


def test_window_starts_reachable_api_and_demo_session(tmp_path, monkeypatch):
    monkeypatch.delenv("BIBAZU_API_DISABLED", raising=False)
    qt = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    pool = ThreadPoolExecutor(1)
    client = None

    def wait(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            qt.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("camera API did not become ready")

    def request(method, path, **kwargs):
        future = pool.submit(client.request, method, path, **kwargs)
        wait(future.done)
        return future.result()

    try:
        server = window.control_api.server
        assert server is not None
        wait(lambda: server.server.started)
        client = httpx.Client(
            base_url=f"http://127.0.0.1:{server.port}/api/v1/", trust_env=False, timeout=3
        )
        assert request("GET", "capabilities").json()["api_version"] == 1
        assert request("GET", "health").json()["connected"] is False
        # The API is reachable before capture; an unconnected camera is an explicit 503.
        assert request("GET", "state").status_code == 503
        window.connect_camera(True)
        wait(lambda: window.worker is not None and window.worker.snapshot() is not None)
        response = request("POST", "sessions", json={"client": "AutoTune regression"})
        assert response.status_code == 201, response.text
        token = response.json()["session_id"]
        state = request("GET", "state").json()
        assert state["connected"]
        assert state["demo"]
        assert request("GET", "measurements").status_code == 200
        assert (
            request("DELETE", "sessions/current", headers={"X-Session-Id": token}).status_code
            == 200
        )
    finally:
        if client:
            client.close()
        pool.shutdown()
        window.close()
        qt.processEvents()
        if window.control_api.server:
            wait(lambda: not window.control_api.server.thread.is_alive())


def test_api_can_still_be_explicitly_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BIBAZU_API_DISABLED", "1")
    qt = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    try:
        assert window.control_api.server is None
    finally:
        window.close()
        qt.processEvents()
