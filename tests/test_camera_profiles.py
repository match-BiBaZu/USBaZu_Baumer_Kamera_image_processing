"""Camera selection, legacy migration and calibration isolation through the real worker."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import httpx
import numpy as np
import pytest
from test_gui_pipeline import dispose, wait_for
from test_gui_pipeline import qt_app as qt_app
from test_worker import FakeAcquisitionCamera

from bibazu_vision import worker as worker_module
from bibazu_vision.app import MainWindow
from bibazu_vision.local_api_runtime import ApiError
from bibazu_vision.profiles import VLXT_SERIAL, camera_data_dir, migrate_settings


def test_legacy_preferences_are_migrated_only_to_usb_without_aliasing(tmp_path):
    legacy = {
        "cti": "custom_usb.cti", "serial": "usb123", "exposure": 3000, "fps": 125,
        "language": "en", "output_dir": str(tmp_path / "clips"),
        "processing": {"roi": [10, 20, 30, 40], "zero_offset_deg": 4.25},
        "camera": {"trigger_line": "Line2"}, "capture": {"start_mode": "manual"},
    }
    settings = migrate_settings(legacy)
    assert settings["selected_camera"] == "vlxt"
    usb, vlxt = (settings["camera_profiles"][key] for key in ("usb", "vlxt"))
    assert usb["cti"] == "custom_usb.cti" and usb["serial"] == "usb123"
    assert (usb["exposure"], usb["fps"]) == (3000, 125)
    assert usb["processing"]["zero_offset_deg"] == 4.25
    assert usb["camera"]["trigger_line"] == "Line2"
    assert usb["capture"]["start_mode"] == "manual"
    assert vlxt["serial"] == VLXT_SERIAL and vlxt["cti"].endswith("bgapi2_gige.cti")
    assert (vlxt["exposure"], vlxt["fps"], vlxt["camera"]["mode"]) == (1000, 200, "live")
    assert vlxt["processing"]["roi"] is None
    assert vlxt["processing"]["zero_offset_deg"] == 0
    assert vlxt["processing"]["chute_angle_deg"] == 0
    usb["processing"]["roi"][0] = 99
    assert legacy["processing"]["roi"][0] == 10
    assert settings["language"] == "en" and settings["output_dir"] == legacy["output_dir"]
    assert camera_data_dir(tmp_path, "usb") == tmp_path / "camera"
    assert camera_data_dir(tmp_path, "vlxt") != tmp_path / "camera"
    assert migrate_settings(settings) == settings


def test_switch_and_restart_preserve_profiles_and_global_preferences(qt_app, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "exposure": 3000,
        "processing": {"roi": [5, 6, 30, 40], "zero_offset_deg": 3.5},
    }), encoding="utf-8")
    window = MainWindow(tmp_path)
    try:
        assert window.selected_camera == "vlxt"
        assert window.camera_selector.currentData() == "vlxt"
        assert window.serial.text() == VLXT_SERIAL
        window.exposure.setValue(2345)
        window.fps.setValue(150)
        window.threshold.setValue(45)
        window.start_mode.setCurrentIndex(window.start_mode.findData("manual"))
        window.profile["processing"].update(roi=[1, 2, 50, 60], zero_offset_deg=-2)
        output = str(tmp_path / "shared_output")
        window.output_path.setText(output)
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("usb"))
        assert window.exposure.value() == 3000
        assert window.processing_config().roi == [5, 6, 30, 40]
        assert window.processing_config().zero_offset_deg == 3.5
        assert "3,50" in window.reference_label.text() or "3.50" in window.reference_label.text()
        window.exposure.setValue(4321)
        window.cti.setText("custom_usb.cti")
        window.line.setCurrentText("Line2")
        window.edge.setCurrentText("FallingEdge")
        window.set_language("en")
        assert window.camera_selector.itemText(0) == "Small USB camera (VCXU-02C)"
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("vlxt"))
        assert window.exposure.value() == 2345 and window.fps.value() == 150
        assert window.threshold.value() == 45
        assert window.start_mode.currentData() == "manual"
        assert window.processing_config().roi == [1, 2, 50, 60]
        assert window.processing_config().zero_offset_deg == -2
        assert window.output_path.text() == output and window.language == "en"
        assert window.line.currentText() == "Line0"
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("usb"))
    finally:
        dispose(window, qt_app)
    reopened = MainWindow(tmp_path)
    try:
        assert reopened.selected_camera == "usb"
        assert reopened.exposure.value() == 4321
        assert reopened.cti.text() == "custom_usb.cti"
        assert reopened.line.currentText() == "Line2"
        assert reopened.edge.currentText() == "FallingEdge"
        assert reopened.processing_config().zero_offset_deg == 3.5
        assert "3.50" in reopened.reference_label.text()
        assert reopened.output_path.text() == output and reopened.language == "en"
        assert reopened.camera_profiles["vlxt"]["processing"]["zero_offset_deg"] == -2
    finally:
        dispose(reopened, qt_app)


def test_real_worker_switch_isolates_background_identity_history_and_api(
    qt_app, tmp_path, monkeypatch,
):
    usb_directory = tmp_path / "camera"
    usb_directory.mkdir()
    usb_background = np.full((64, 96), 185, np.uint8)
    assert cv2.imwrite(str(usb_directory / "background.png"), usb_background)
    cameras = []
    open_release, close_entered, close_release = (
        threading.Event(), threading.Event(), threading.Event()
    )

    class ControlledCamera(FakeAcquisitionCamera):
        def open(self):
            assert open_release.wait(5)
            super().open()

        def close(self):
            close_entered.set()
            assert close_release.wait(5)
            super().close()

    def factory(config):
        camera = ControlledCamera(config)
        camera.diagnostics["devices"] = [
            {"model": "Wrong first device", "serial": "wrong"},
        ]
        camera.diagnostics["selected_device"] = {
            "model": "VLXT-28M.I" if config.serial == VLXT_SERIAL else "VCXU-02C",
            "serial": config.serial or "usb123",
        }
        cameras.append(camera)
        return camera

    monkeypatch.setattr(worker_module, "BaumerCamera", factory)
    window = MainWindow(tmp_path)
    pool = ThreadPoolExecutor(max_workers=1)
    client = None
    failures = []
    try:
        window.connect_camera(False)
        window.worker.failed.connect(failures.append)
        assert not window.camera_selector.isEnabled()
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("usb"))
        assert window.selected_camera == "vlxt" and window.camera_selector.currentData() == "vlxt"
        open_release.set()
        wait_for(qt_app, lambda: len(cameras) == 1 and cameras[0].opened.is_set())
        camera = cameras[0]
        assert camera.config.serial == VLXT_SERIAL
        assert camera.config.cti_path.endswith("bgapi2_gige.cti")
        assert camera.config.pixel_format == "Mono8"
        camera.send_frame()
        wait_for(qt_app, lambda: window.last_snapshot is not None)
        assert not window.last_snapshot["has_background"]
        assert "VLXT-28M.I" in window.connection_label.text()
        assert VLXT_SERIAL in window.connection_label.text()
        assert "Wrong first device" not in window.connection_label.text()
        previous_epoch = window.worker.measurement_epoch
        window.command("background")
        window.command("config", {"roi": [2, 3, 80, 50], "zero_offset_deg": 7})
        wait_for(qt_app, lambda: window.worker.measurement_epoch >= previous_epoch + 2)
        camera.send_frame()
        wait_for(qt_app, lambda: window.last_snapshot["config"]["zero_offset_deg"] == 7)
        assert window.last_snapshot["has_background"]
        vlxt_background = camera_data_dir(tmp_path, "vlxt") / "background.png"
        assert vlxt_background.exists()
        assert np.array_equal(cv2.imread(str(usb_directory / "background.png"), 0), usb_background)
        old_worker = window.worker
        token = window.control_api.lease.acquire()["session_id"]
        completed = {
            "run_id": "run", "trial_id": "done", "context": {}, "timeout_seconds": 1.0,
            "capture_config": {}, "status": "completed", "clip_path": None,
        }
        old_worker.api_capture.trials["done"] = completed
        window.control_api.trial_sources["done"] = old_worker.api_capture
        window.disconnect_camera()
        wait_for(qt_app, close_entered.is_set)
        assert not window.camera_selector.isEnabled()
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("usb"))
        assert window.selected_camera == "vlxt"
        close_release.set()
        wait_for(qt_app, lambda: window.camera_selector.isEnabled())
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("usb"))
        assert window.worker is None
        assert window.history_archive is None and not window.trace.samples
        assert window.trace.measurements.total_count == 0
        assert window.last_snapshot is None and window.live_view.source is None
        assert window.capture_view.source is None
        assert window.processing_config().roi is None
        assert window.processing_config().zero_offset_deg == 0
        with pytest.raises(ApiError, match="Sitzung"):
            window.control_api.lease.require(token)
        server = window.control_api.server
        wait_for(qt_app, lambda: server.server.started)
        client = httpx.Client(base_url=f"http://127.0.0.1:{server.port}", trust_env=False)
        future = pool.submit(client.get, "/api/v1/trials/done")
        wait_for(qt_app, future.done)
        assert future.result().status_code == 200
        assert future.result().json()["status"] == "completed"
        window.connect_camera(False)
        wait_for(qt_app, lambda: len(cameras) == 2 and cameras[1].opened.is_set())
        assert cameras[1].config.cti_path.endswith("bgapi2_usb.cti")
        assert cameras[1].config.pixel_format is None
        cameras[1].send_frame()
        wait_for(qt_app, lambda: window.last_snapshot is not None)
        assert window.last_snapshot["has_background"]
        assert "VCXU-02C" in window.connection_label.text()
        assert "usb123" in window.connection_label.text()
        assert failures == []
    finally:
        open_release.set()
        close_release.set()
        if client is not None:
            client.close()
        pool.shutdown()
        dispose(window, qt_app)


def test_stopped_demo_does_not_contaminate_selected_profile(qt_app, tmp_path):
    window = MainWindow(tmp_path)
    try:
        window.threshold.setValue(83)
        window.connect_camera(True)
        wait_for(qt_app, lambda: window.last_snapshot is not None)
        window.disconnect_camera()
        wait_for(qt_app, lambda: window.camera_selector.isEnabled())
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("usb"))
        assert window.camera_profiles["vlxt"]["processing"]["threshold"] == 83
        assert window.camera_profiles["vlxt"]["processing"]["chute_angle_deg"] == 0
        assert window.camera_profiles["vlxt"]["capture"]["start_mode"] == "hardware"
        assert window.processing_config().roi is None
    finally:
        dispose(window, qt_app)


def test_failed_connection_unlocks_profile_selection(qt_app, tmp_path, monkeypatch):
    class MissingCamera(FakeAcquisitionCamera):
        def open(self):
            raise RuntimeError(f"Keine Kamera: {self.config.serial}")

    monkeypatch.setattr(worker_module, "BaumerCamera", MissingCamera)
    window = MainWindow(tmp_path)
    try:
        window.connect_camera(False)
        assert not window.camera_selector.isEnabled()
        wait_for(qt_app, lambda: window.camera_selector.isEnabled())
        assert "Kamerafehler" in window.state_label.text()
        assert VLXT_SERIAL in window.state_label.text()
        assert window.cti.isEnabled() and window.serial.isEnabled()
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("usb"))
        assert window.selected_camera == "usb"
        assert window.cti.text().endswith("bgapi2_usb.cti")
        assert window.worker is None
    finally:
        dispose(window, qt_app)
