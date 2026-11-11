"""Sensor crop constraints, safe restore, and background/reference coordinate mapping."""

import json
from dataclasses import replace

import cv2
import numpy as np
import pytest
from test_camera import FakeAcquirer, Node, patches
from test_gui_pipeline import dispose, wait_for
from test_gui_pipeline import qt_app as qt_app
from test_worker import FakeAcquisitionCamera

from bibazu_vision import worker as worker_module
from bibazu_vision.app import MainWindow
from bibazu_vision.camera import BaumerCamera, CameraConfig
from bibazu_vision.image_geometry import (
    aligned_background,
    load_background,
    save_background,
    translated_roi,
)


class RoiNode:
    def __init__(self, state, name, writes, reject=None):
        self.state, self.name, self.writes, self.reject = state, name, writes, reject
        self.min = 4 if name == "Height" else 0
        self.inc = 4
        self.choices = []

    @property
    def max(self):
        return 1464 - self.state["OffsetY" if self.name == "Height" else "Height"]

    @property
    def value(self):
        return self.state[self.name]

    @value.setter
    def value(self, value):
        if value == self.reject:
            raise RuntimeError("ROI rejected")
        if not self.min <= value <= self.max or (value - self.min) % self.inc:
            raise RuntimeError("ROI outside sensor bounds/increment")
        self.state[self.name] = value
        self.writes.append((self.name, value))


def roi_camera(top, bottom, *, height=1464, offset=0, reject=None):
    acquirer = FakeAcquirer()
    state, writes = {"Height": height, "OffsetY": offset}, []
    acquirer.node_map.Width = Node(1920)
    acquirer.node_map.HeightMax = Node(1464)
    acquirer.node_map.OffsetX = Node(0)
    acquirer.node_map.Height = RoiNode(state, "Height", writes)
    acquirer.node_map.OffsetY = RoiNode(state, "OffsetY", writes, reject=reject)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti", crop_top=top, crop_bottom=bottom))
    return camera, acquirer, state, writes


@pytest.mark.parametrize("height,offset", [(1464, 0), (1064, 200), (800, 600)])
def test_hardware_crop_is_absolute_and_restores_even_a_previously_cropped_camera(height, offset):
    camera, acquirer, state, writes = roi_camera(200, 200, height=height, offset=offset)
    p1, p2, p3 = patches(acquirer)
    with p1, p2, p3:
        camera.open()
        assert state == {"Height": 1064, "OffsetY": 200}
        assert camera.diagnostics["actual"]["Width"] == 1920
        assert camera.diagnostics["vertical_roi"]["top"] == 200
        assert camera.diagnostics["vertical_roi"]["bottom"] == 200
        camera.close()
    assert state == {"Height": height, "OffsetY": offset}
    assert acquirer.destroyed


def test_hardware_crop_rounds_to_sensor_steps_and_restores_after_partial_failure():
    camera, acquirer, state, writes = roi_camera(201, 200)
    p1, p2, p3 = patches(acquirer)
    with p1, p2, p3:
        camera.open()
        assert state == {"Height": 1060, "OffsetY": 204}
        camera.close()
    assert state == {"Height": 1464, "OffsetY": 0}
    camera, acquirer, state, writes = roi_camera(200, 200, reject=200)
    p1, p2, p3 = patches(acquirer)
    with p1, p2, p3, pytest.raises(RuntimeError, match="ROI rejected"):
        camera.open()
    assert state == {"Height": 1464, "OffsetY": 0}
    assert acquirer.destroyed


@pytest.mark.parametrize("exposure", [None, 2000])
@pytest.mark.parametrize("region_mode", ["On", "Off"])
def test_automatic_control_roi_is_unlocked_before_crop_and_fully_restored(exposure, region_mode):
    camera, acquirer, state, writes = roi_camera(200, 200)
    camera.config.exposure_us = exposure
    state.update(ExposureAuto="Continuous", GainAuto="Continuous", auto_height=1464)

    class AutomaticMode(Node):
        def __init__(self, name):
            self.name = name
            self.choices = ["Off", "Continuous"]

        @property
        def value(self):
            return state[self.name]

        @value.setter
        def value(self, value):
            state[self.name] = value
            writes.append((self.name, value))

    class AutomaticHeight(Node):
        choices = []

        def __init__(self):
            pass

        @property
        def value(self):
            return state["auto_height"]

        @value.setter
        def value(self, value):
            if value > state["Height"]:
                raise RuntimeError("automatic ROI exceeds camera height")
            state["auto_height"] = value
            writes.append(("AutoFeatureHeight", value))

    class AutoLockedHeight(RoiNode):
        def __init__(self):
            self.state, self.name, self.writes, self.reject = state, "Height", writes, None
            self.inc, self.choices = 4, []

        @property
        def min(self):
            return state["auto_height"] if region_mode == "On" and any(
                state[name] != "Off" for name in ("ExposureAuto", "GainAuto")
            ) else 4

        @RoiNode.value.setter
        def value(self, value):
            RoiNode.value.fset(self, value)
            # Real firmware silently shrinks its automatic ROI along with Height.
            state["auto_height"] = min(state["auto_height"], value)

    nm = acquirer.node_map
    nm.Height = AutoLockedHeight()
    nm.ExposureAuto = AutomaticMode("ExposureAuto")
    nm.GainAuto = AutomaticMode("GainAuto")
    nm.ExposureTime = Node(10000)
    nm.AutoFeatureRegionSelector = Node("BrightnessAuto", ["BrightnessAuto"])
    nm.AutoFeatureRegionMode = Node(region_mode, ["Off", "On"])
    nm.AutoFeatureHeight = AutomaticHeight()
    nm.AutoFeatureWidth = Node(1920)
    nm.AutoFeatureOffsetX = Node(0)
    nm.AutoFeatureOffsetY = Node(0)
    original = state.copy()
    p1, p2, p3 = patches(acquirer)
    with p1, p2, p3:
        camera.open()
        assert state["Height"] == state["auto_height"] == 1064
        assert state["OffsetY"] == 200
        assert state["ExposureAuto"] == state["GainAuto"] == "Off"
        assert writes.index(("GainAuto", "Off")) < writes.index(("Height", 1064))
        camera.close()
    assert state == original
    assert writes.index(("Height", 1464)) < writes.index(("AutoFeatureHeight", 1464))
    assert writes.index(("AutoFeatureHeight", 1464)) < writes.index(("ExposureAuto", "Continuous"))
    assert nm.AutoFeatureRegionMode.value == region_mode
    assert not any("wiederhergestellt" in warning for warning in camera.diagnostics["warnings"])


@pytest.mark.parametrize("top,bottom", [(1400, 200), (-1, 0), (0.5, 0)])
def test_impossible_camera_crop_does_not_leave_changed_hardware(top, bottom):
    camera, acquirer, state, writes = roi_camera(top, bottom)
    p1, p2, p3 = patches(acquirer)
    with p1, p2, p3, pytest.raises(RuntimeError):
        camera.open()
    assert state == {"Height": 1464, "OffsetY": 0}


def test_roi_translation_preserves_physical_region_across_crop_and_uncrop():
    original = dict(width=1920, height=1464, offset_x=0, offset_y=0)
    cropped = dict(width=1920, height=1064, offset_x=0, offset_y=200)
    roi = [2, 360, 1454, 862]
    assert translated_roi(roi, original, cropped) == [2, 160, 1454, 862]
    assert translated_roi([2, 160, 1454, 862], cropped, original) == roi
    assert translated_roi([0, 0, 100, 100], original, cropped) is None


def test_background_alignment_uses_origin_not_just_equal_image_dimensions(tmp_path):
    base = dict(width=96, height=64, offset_x=0, offset_y=0)
    target = {**base, "height": 48, "offset_y": 8}
    image = np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 96))
    np.testing.assert_array_equal(aligned_background(image, None, base, target), image[8:56])
    path = tmp_path / "background.png"
    save_background(path, image[8:56], target)
    saved, geometry = load_background(path)
    np.testing.assert_array_equal(aligned_background(saved, geometry, base, target), image[8:56])
    shifted = {**target, "offset_y": 12}
    assert aligned_background(saved, geometry, base, shifted) is None
    assert aligned_background(saved, None, base, target) is None
    # A partially published PNG cannot be used with the preceding crop metadata.
    assert cv2.imwrite(str(path), image[12:60])
    with pytest.raises(ValueError, match="Metadaten"):
        load_background(path)


def test_real_worker_crops_legacy_background_and_translates_reference_once(
    qt_app, tmp_path, monkeypatch,
):
    base = dict(width=96, height=64, offset_x=0, offset_y=0)
    directory = tmp_path / "camera"
    directory.mkdir()
    path = directory / "background.png"
    assert cv2.imwrite(str(path), np.full((64, 96), 185, np.uint8))
    original_bytes = path.read_bytes()
    cameras = []

    class CroppedCamera(FakeAcquisitionCamera):
        def read(self, timeout=0.2):
            frame = super().read(timeout)
            return replace(frame, image=frame.image[8:-8].copy()) if frame is not None else None

    def factory(config):
        camera = CroppedCamera(config)
        camera.diagnostics.update(
            actual={"Width": 96, "Height": 48, "OffsetX": 0, "OffsetY": 8},
            vertical_roi={"base": base, "top": 8, "bottom": 8},
        )
        cameras.append(camera)
        return camera

    monkeypatch.setattr(worker_module, "BaumerCamera", factory)
    from bibazu_vision.processing import ProcessingConfig
    from bibazu_vision.recording import CaptureConfig

    geometry = None
    for index in range(2):
        worker = worker_module.PipelineWorker(
            CameraConfig(crop_top=8, crop_bottom=8),
            ProcessingConfig(threshold=100, roi=[10, 10 if index == 0 else 2, 70, 40]),
            CaptureConfig(), directory, processing_geometry=geometry,
        )
        failures = []
        worker.failed.connect(failures.append)
        worker.start()
        try:
            wait_for(qt_app, lambda expected=index: len(cameras) > expected)
            cameras[-1].send_frame()
            wait_for(qt_app, lambda current=worker: current.snapshot() is not None)
            snapshot = worker.snapshot()
            assert snapshot["image"].shape == (48, 96)
            assert snapshot["has_background"]
            assert snapshot["config"]["roi"] == [10, 2, 70, 40]
            assert snapshot["detection"].angle_deg == pytest.approx(0)
            geometry = snapshot["camera_geometry"]
            assert geometry["offset_y"] == 8
            assert path.read_bytes() == original_bytes
            assert failures == []
        finally:
            worker.stop()
            assert worker.wait(3000)


def test_crop_preferences_survive_old_settings_saves_and_are_separate_per_camera(
    qt_app, tmp_path, monkeypatch,
):
    monkeypatch.setenv("BIBAZU_API_DISABLED", "1")
    window = MainWindow(tmp_path)
    try:
        window.crop_top.setValue(200)
        window.crop_bottom.setValue(200)
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("usb"))
        assert window.crop_top.value() == window.crop_bottom.value() == 0
        window.camera_selector.setCurrentIndex(window.camera_selector.findData("vlxt"))
        assert window.crop_top.value() == window.crop_bottom.value() == 200
        window.gap_ms.setValue(120)
        window.start_history()
        assert window.trace.measurements.max_gap_seconds == pytest.approx(0.12)
        window.set_language("en", persist=False)
        assert window.gap_ms.value() == 120
    finally:
        dispose(window, qt_app)
    # An already running older application may still replace settings.json.
    (tmp_path / "settings.json").write_text(json.dumps({}), encoding="utf-8")
    reopened = MainWindow(tmp_path)
    try:
        assert reopened.crop_top.value() == reopened.crop_bottom.value() == 200
    finally:
        dispose(reopened, qt_app)
