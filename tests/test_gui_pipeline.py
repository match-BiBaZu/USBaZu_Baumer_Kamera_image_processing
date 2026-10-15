"""Offscreen UI checks using the actual Qt worker, writer and replay controls."""

import json
import math
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from test_worker import FakeAcquisitionCamera

from bibazu_vision import worker as worker_module
from bibazu_vision.app import ImageView, MainWindow
from bibazu_vision.camera import CameraFrame
from bibazu_vision.demo import DemoCamera
from bibazu_vision.processing import wrap_axis_angle


@pytest.fixture(scope="module")
def qt_app():
    application = QApplication.instance() or QApplication([])
    # Qt's Windows offscreen plugin has no system font database. Register a
    # local font only for headless rendering; the normal desktop uses its OS font.
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if not QFontDatabase.families() and font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            application.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0]))
    return application


def wait_for(application, predicate, timeout=2.0, description="condition"):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        application.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    raise AssertionError(f"Timed out waiting for {description}")


def screenshot(window, name):
    artifact_dir = os.environ.get("BIBAZU_GUI_ARTIFACTS")
    if artifact_dir:
        destination = Path(artifact_dir)
        destination.mkdir(parents=True, exist_ok=True)
        assert window.grab().save(str(destination / name))


def dispose(window, application):
    window.timer.stop()
    window.play_timer.stop()
    if window.worker is not None:
        window.worker.stop()
        assert window.worker.wait(2500), "worker did not stop safely"
    application.processEvents()
    window.close()
    window.deleteLater()
    application.processEvents()


def test_apply_camera_button_without_frames_pending_edits_language_and_recording_guards(
    qt_app, tmp_path, monkeypatch
):
    cameras = []
    apply_entered, apply_release = threading.Event(), threading.Event()

    class PendingCamera(FakeAcquisitionCamera):
        def apply_settings(self, exposure_us, fps):
            apply_entered.set()
            assert apply_release.wait(5)
            return super().apply_settings(exposure_us, fps)

    def factory(config):
        camera = PendingCamera(config)
        cameras.append(camera)
        return camera

    monkeypatch.setattr(worker_module, "BaumerCamera", factory)
    window = MainWindow(tmp_path)
    failures = []
    try:
        assert not window.apply_camera_button.isEnabled()
        window.connect_camera(False)
        window.worker.failed.connect(failures.append)
        wait_for(qt_app, lambda: window.apply_camera_button.isEnabled())
        camera = cameras[0]
        assert window.last_snapshot is None
        window.exposure.setValue(5000)
        window.fps.setValue(80)
        window.set_language("en", persist=False)
        assert window.exposure.value() == 5000 and window.fps.value() == 80
        assert window.apply_camera_button.text() == "Apply camera settings"
        assert camera.settings_calls == []
        worker = window.worker
        QTest.mouseClick(window.apply_camera_button, Qt.MouseButton.LeftButton)
        wait_for(qt_app, apply_entered.is_set)
        assert window.camera_settings_pending
        assert not window.apply_camera_button.isEnabled()
        # Further edits while an update is in flight stay pending in the form.
        window.exposure.setValue(6000)
        window.set_language("de", persist=False)
        assert window.exposure.value() == 6000
        assert window.apply_camera_button.text() == "Kameraeinstellungen anwenden"
        apply_release.set()
        wait_for(qt_app, lambda: not window.camera_settings_pending)
        assert len(camera.settings_calls) == 1
        assert camera.settings_calls[0][1:] == (5000, 80)
        assert window.worker is worker
        assert window.exposure.value() == 6000
        window.set_language("en", persist=False)
        assert "Camera settings applied: 5000 µs" in window.log.toPlainText()
        assert "Requested frame rate 80 FPS" in window.log.toPlainText()
        assert "chute empty" in window.log.toPlainText()
        window.command("arm")
        wait_for(qt_app, lambda: not window.apply_camera_button.isEnabled())
        assert window.last_snapshot is None
        window.command("disarm")
        wait_for(qt_app, lambda: window.apply_camera_button.isEnabled())
        camera.send_frame()
        wait_for(qt_app, lambda: window.last_snapshot is not None)
        apply_entered.clear()
        apply_release.clear()
        window.apply_camera_settings()
        wait_for(qt_app, apply_entered.is_set)
        # Preview/status refresh must remain responsive during slow SDK writes.
        window.refresh_live()
        assert window.camera_settings_pending and not failures
        assert worker.camera_settings_updating
        apply_release.set()
        wait_for(qt_app, lambda: not window.camera_settings_pending)
        assert camera.settings_calls[-1][1:] == (6000, 80)
        wait_for(qt_app, lambda: window.apply_camera_button.isEnabled())
        # Failed enqueue must clear the pending UI state immediately.
        with monkeypatch.context() as patcher:
            patcher.setattr(worker, "command", lambda *args: False)
            window.apply_camera_settings()
        assert not window.camera_settings_pending and window.apply_camera_button.isEnabled()
        window.disconnect_camera()
        assert not window.apply_camera_button.isEnabled()
        wait_for(qt_app, lambda: not worker.isRunning())
        assert camera.closed.is_set() and failures == []
    finally:
        apply_release.set()
        dispose(window, qt_app)


def test_apply_camera_button_disabled_in_demo(qt_app, tmp_path):
    window = MainWindow(tmp_path)
    try:
        window.connect_camera(True)
        wait_for(qt_app, lambda: window.last_snapshot is not None)
        assert not window.apply_camera_button.isEnabled()
    finally:
        dispose(window, qt_app)


def test_demo_motion_capture_persists_then_replays_and_steps(qt_app, tmp_path):
    window = MainWindow(tmp_path)
    failures = []
    try:
        if os.environ.get("BIBAZU_GUI_ARTIFACTS"):
            window.show()
        window.connect_camera(True)
        window.worker.failed.connect(failures.append)
        wait_for(qt_app, lambda: window.last_snapshot is not None, description="first demo preview")
        # Capture the genuinely empty initial demo scene through the same command
        # as the GUI's background button, then arm motion based passage capture.
        window.command("background")
        wait_for(
            qt_app,
            lambda: "Hintergrund aufgenommen" in window.log.toPlainText(),
            description="background capture command",
        )
        window.arm()
        capture_deadline = time.perf_counter() + 4.8
        wait_for(
            qt_app,
            lambda: (
                window.last_snapshot["detection"].angle_deg is not None
                and not window.last_snapshot["detection"].touching_border
                and window.last_snapshot["detection"].confidence > 0.6
            ),
            timeout=2,
            description="fully visible demo part",
        )
        screenshot(window, "gui_live_demo.png")
        wait_for(
            qt_app,
            lambda: window.last_saved is not None or bool(failures),
            timeout=max(0.1, capture_deadline - time.perf_counter()),
            description="complete demo passage and asynchronous save",
        )
        assert failures == []
        assert window.last_saved is not None
        window.command("disarm")
        window.open_clip(window.last_saved)
        clip = window.clip
        assert clip is not None
        assert clip.manifest["stop_reason"] == "part_left"
        assert clip.manifest["motion_seen"]
        assert len(clip.frames) > 40
        timestamp_differences = np.diff(clip.timestamps)
        assert np.all(timestamp_differences > 0)
        assert not clip.frames[0].flags.writeable
        assert window.tabs.currentIndex() == 1
        assert window.slider.maximum() == len(clip.frames) - 1
        angles = [
            (i, measurement["angle_deg"])
            for i, measurement in enumerate(clip.measurements)
            if measurement.get("angle_deg") is not None
        ]
        assert angles and max(angle for _, angle in angles) > 30
        assert max(angle for _, angle in angles) < 39
        clipped_measurements = [
            measurement for measurement in clip.measurements if measurement["touching_border"]
        ]
        assert clipped_measurements
        assert all(measurement["angle_deg"] is None for measurement in clipped_measurements)
        selected = angles[len(angles) // 2][0]
        window.slider.setValue(selected)
        assert window.review_index == selected
        assert f"Bild {selected + 1}/" in window.review_info.text()
        assert "part_left" in window.review_info.text()
        assert window.review_view.source is not None
        qt_app.processEvents()
        screenshot(window, "gui_review_demo.png")
        rendered = window.review_image(selected)
        original = cv2.cvtColor(clip.frames[selected], cv2.COLOR_GRAY2BGR)
        assert rendered.shape == original.shape
        assert not np.array_equal(rendered, original)
        assert np.array_equal(clip.frames[selected], original[:, :, 0])
        window.slider.setValue(selected + 1)
        assert window.review_index == selected + 1
        window.speed.setCurrentIndex(0)
        before = window.slider.value()
        window.toggle_play()
        assert window.play_timer.isActive()
        window.advance_review()
        assert window.slider.value() > before
        window.toggle_play()
        assert not window.play_timer.isActive()
        window.disconnect_camera()
        wait_for(qt_app, lambda: not window.worker.isRunning(), description="camera shutdown")
        qt_app.processEvents()
        assert window.connect_button.isEnabled()
        assert (
            window.review_view.source is not None
        )  # Replay remains available after disconnection.
    finally:
        dispose(window, qt_app)


class StaticDemo(DemoCamera):
    """Keep an irregular part stationary so repeated zeroing is deterministic."""

    def read(self, timeout=0.5):
        time.sleep(0.006)
        self.frame_id += 1
        image = self.background()
        points = np.array([[-105, -25], [70, -25], [70, -10], [105, -10], [105, 25], [-105, 25]])
        angle = math.radians(28)
        rotation = np.array(
            [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]
        )
        points = np.rint(points @ rotation.T + [390, 230]).astype(np.int32)
        cv2.fillPoly(image, [points], 35)
        return CameraFrame(image, time.perf_counter(), self.frame_id, False)


def test_native_profile_does_not_hide_demo_and_is_preserved_on_close(qt_app, tmp_path, monkeypatch):
    native_processing = {
        "threshold": 74,
        "min_area": 10000,
        "roi": [50, 110, 575, 240],
        "chute_angle_deg": -1.6,
        "zero_offset_deg": -0.99,
        "method": "pca",
        "foreground": "light",
        "scale": 0.5,
    }
    profile = {"processing": native_processing, "capture": {"start_mode": "hardware"}}
    stored_profile = json.dumps(profile, indent=2) + "\n"
    profile_path = tmp_path / "settings.json"
    profile_path.write_text(stored_profile, encoding="utf-8")

    class OutsideNativeROI(DemoCamera):
        def read(self, timeout=0.5):
            time.sleep(0.008)
            self.frame_id += 1
            image = self.background()
            polygon = cv2.boxPoints(((710, 210), (160, 50), -18))
            cv2.fillPoly(image, [np.rint(polygon).astype(np.int32)], 35)
            return CameraFrame(image, time.perf_counter(), self.frame_id, False)

    class NativeStandIn(DemoCamera):
        """Exercise switching profiles without opening an actual camera."""

        def read(self, timeout=0.5):
            time.sleep(0.008)
            self.frame_id += 1
            image = np.zeros((480, 640), np.uint8)
            cv2.rectangle(image, (150, 150), (500, 280), 220, -1)
            return CameraFrame(image, time.perf_counter(), self.frame_id, False)

    monkeypatch.setattr(worker_module, "DemoCamera", OutsideNativeROI)
    monkeypatch.setattr(worker_module, "BaumerCamera", NativeStandIn)
    window = MainWindow(tmp_path)
    failures = []
    try:
        assert window.start_mode.currentData() == "hardware"
        assert window.threshold.value() == 74
        window.connect_camera(True)
        window.worker.failed.connect(failures.append)
        wait_for(
            qt_app, lambda: window.last_snapshot is not None, description="isolated demo preset"
        )
        demo_config = window.last_snapshot["config"]
        detection = window.last_snapshot["detection"]
        assert demo_config["threshold"] == 25
        assert demo_config["min_area"] == 150
        assert demo_config["roi"] is None
        assert demo_config["zero_offset_deg"] == 0
        assert demo_config["chute_angle_deg"] == 2.86
        assert demo_config["method"] == "obb"
        assert demo_config["foreground"] == "dark"
        assert demo_config["scale"] == 1
        assert window.start_mode.currentData() == "motion"
        assert detection.center is not None
        assert detection.center[0] > native_processing["roi"][0] + native_processing["roi"][2]
        assert detection.area < native_processing["min_area"]
        assert detection.angle_deg == pytest.approx(18 - 2.86, abs=0.8)
        assert not detection.touching_border
        original_angle = detection.angle_deg
        frame_id = window.last_snapshot["frame_id"]
        wait_for(qt_app, lambda: window.last_snapshot["frame_id"] > frame_id + 4)
        assert window.last_snapshot["detection"].angle_deg == pytest.approx(
            original_angle, abs=0.01
        )
        window.close()
        qt_app.processEvents()
        assert not window.worker.isRunning()
        assert profile_path.read_text(encoding="utf-8") == stored_profile
        assert failures == []

        # The same window can return from the demo to the saved native controls.
        window.connect_camera(False)
        window.worker.failed.connect(failures.append)
        wait_for(
            qt_app, lambda: window.last_snapshot is not None, description="native profile restore"
        )
        assert window.last_snapshot["config"] == native_processing
        assert window.start_mode.currentData() == "hardware"
        assert window.threshold.value() == 74
        assert window.area.value() == 10000
        assert window.method.currentData() == "pca"
        assert window.foreground.currentData() == "light"
        assert window.scale.currentData() == 0.5
        assert failures == []
    finally:
        dispose(window, qt_app)


def test_zero_twice_keeps_calibration_and_close_stops_worker(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(worker_module, "DemoCamera", StaticDemo)
    stored_profile = '{"processing": {"chute_angle_deg": -4, "zero_offset_deg": 6}}\n'
    profile_path = tmp_path / "settings.json"
    profile_path.write_text(stored_profile, encoding="utf-8")
    window = MainWindow(tmp_path)
    failures = []
    try:
        window.connect_camera(True)
        window.worker.failed.connect(failures.append)
        wait_for(qt_app, lambda: window.last_snapshot is not None, description="stationary part")
        assert failures == []
        original_axis = window.last_snapshot["detection"].axis_deg
        assert original_axis is not None
        window.command("zero")
        wait_for(
            qt_app,
            lambda: abs(window.last_snapshot["detection"].angle_deg or 0) < 0.01,
            description="first zero calibration",
        )
        offset = window.last_snapshot["config"]["zero_offset_deg"]
        assert wrap_axis_angle(offset) == pytest.approx(
            wrap_axis_angle(original_axis - window.chute_angle.value()), abs=0.01
        )
        previous_frame = window.last_snapshot["frame_id"]
        window.command("zero")
        wait_for(
            qt_app,
            lambda: window.last_snapshot["frame_id"] > previous_frame + 2,
            description="second zero command",
        )
        assert window.last_snapshot["config"]["zero_offset_deg"] == pytest.approx(offset, abs=0.01)
        assert window.last_snapshot["detection"].angle_deg == pytest.approx(0, abs=0.01)
        window.close()
        assert not window.worker.isRunning()
        qt_app.processEvents()
        # Demo calibration must not overwrite the user's hardware settings.
        assert profile_path.read_text(encoding="utf-8") == stored_profile
    finally:
        dispose(window, qt_app)


def test_image_coordinates_account_for_letterbox_and_selection(qt_app):
    view = ImageView()
    try:
        view.resize(1000, 800)
        view.set_image(np.zeros((480, 800), np.uint8))
        view.show()
        qt_app.processEvents()
        assert view.image_point(QPointF(500, 400)) == (400, 240)
        assert view.image_point(QPointF(500, 10)) is None
        selections = []
        view.selected.connect(lambda mode, points: selections.append((mode, points)))
        view.select("roi")
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(200, 200))
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(800, 600))
        assert selections == [("roi", ((160, 80), (640, 400)))]
        assert view.mode is None
    finally:
        view.close()
        view.deleteLater()
        qt_app.processEvents()


def test_roi_and_chute_selection_send_original_coordinates(qt_app, tmp_path, monkeypatch):
    window = MainWindow(tmp_path)
    commands = []
    monkeypatch.setattr(window, "command", lambda name, value=None: commands.append((name, value)))
    try:
        window.image_selection("roi", ((640, 390), (120, 110)))
        assert commands[-1] == ("config", {"roi": (120, 110, 520, 280)})
        window.image_selection("chute", ((30, 150), (230, 130)))
        command = commands[-1]
        assert command[0] == "config"
        assert command[1]["chute_angle_deg"] == pytest.approx(math.degrees(math.atan2(20, 200)))
        assert command[1]["zero_offset_deg"] == 0
        assert window.chute_angle.value() == pytest.approx(5.71, abs=0.001)
        sent_commands = len(commands)
        window.image_selection("roi", ((100, 100), (100, 200)))
        window.image_selection("roi", ((100, 100), (200, 100)))
        assert (
            len(commands) == sent_commands
        )  # Invalid ROI selection must keep acquisition running.
    finally:
        dispose(window, qt_app)
