"""Deferred display work and startup thread tuning must preserve measurement data."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from test_processing import silhouette

from bibazu_vision import app as app_module
from bibazu_vision import performance
from bibazu_vision import processing as processing_module
from bibazu_vision.processing import Detection, FrameProcessor, ProcessingConfig, render_overlay
from bibazu_vision.recording import CaptureConfig, CaptureManager, load_clip


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25])
def test_mask_is_built_once_for_contour_only_and_retains_its_original_geometry(monkeypatch, scale):
    config = ProcessingConfig(roi=(60, 60, 260, 230), scale=scale, method="pca")
    processor = FrameProcessor(config)
    image = silhouette(17, irregular=True)
    detection = processor.process(image)
    original_pixels = image.copy()
    angle = detection.angle_deg
    assert angle is not None
    assert detection.mask is None
    source = detection._mask_source
    compact = source.binary.copy()
    calls = []
    get_region = processing_module._MaskSource.get_region

    def tracked(source):
        calls.append(source.image_shape)
        return get_region(source)

    monkeypatch.setattr(processing_module._MaskSource, "get_region", tracked)
    for obb, axis in ((True, False), (False, True), (True, True), (False, False)):
        render_overlay(
            image, detection, config, show_contour=False, show_obb=obb, show_axis=axis,
        )
        assert detection.mask is None
    assert calls == []
    # Later frames/configuration must not overwrite an older preview's small mask.
    preview_config = replace(config)
    config.roi = (0, 0, 32, 32)
    config.scale = 0.3
    processor.process(np.full_like(image, 230))
    np.testing.assert_array_equal(source.binary, compact)
    first = render_overlay(image, detection, preview_config, show_contour=True)
    second = render_overlay(image, detection, preview_config, show_contour=True)
    assert calls == [image.shape]
    assert detection.mask is None
    assert detection.get_mask().shape == image.shape
    assert not detection.get_mask()[:60].any()
    assert not detection.get_mask()[:, :60].any()
    assert detection.get_mask()[160, 180] == 255
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(image, original_pixels)
    assert detection.angle_deg == angle


def test_original_recording_never_materializes_a_display_mask(tmp_path, monkeypatch):
    def unexpected_mask(*args):
        raise AssertionError("original recording requested a display mask")

    monkeypatch.setattr(processing_module._MaskSource, "materialize", unexpected_mask)
    monkeypatch.setattr(processing_module._MaskSource, "get_region", unexpected_mask)
    processor = FrameProcessor(ProcessingConfig(scale=0.25, method="pca"))
    recorder = CaptureManager(CaptureConfig(), tmp_path)
    try:
        recorder.start_timed(1, now=10)
        originals, angles = [], []
        for index, angle in enumerate((0, 17, -30)):
            image = silhouette(angle, irregular=True)
            detection = processor.process(image)
            originals.append(image.copy())
            angles.append(detection.angle_deg)
            recorder.ingest(image, 10 + index / 10, index, detection)
            assert detection.mask is None
        recorder.poll(11)
        assert recorder.wait_until_saved(3)
        clip = load_clip(recorder.last_clip_path)
        assert [row["angle_deg"] for row in clip.measurements] == angles
        for actual, expected in zip(clip.frames, originals, strict=True):
            np.testing.assert_array_equal(actual, expected)
    finally:
        assert recorder.close(timeout=3)


def test_opencv_thread_configuration_reads_environment_and_allows_explicit_override(monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "setNumThreads", calls.append)
    monkeypatch.setattr(cv2, "getNumThreads", lambda: calls[-1])
    monkeypatch.delenv("BIBAZU_OPENCV_THREADS", raising=False)
    assert performance.configure_opencv_threads() == performance.DEFAULT_OPENCV_THREADS
    monkeypatch.setenv("BIBAZU_OPENCV_THREADS", "2")
    assert performance.configure_opencv_threads() == 2
    assert performance.configure_opencv_threads(4) == 4
    assert calls == [performance.DEFAULT_OPENCV_THREADS, 2, 4]


def test_native_tint_preserves_original_truncation_for_all_channel_values():
    values = np.tile(np.arange(256, dtype=np.uint8), (80, 1))
    image = np.stack((values, 255 - values, np.roll(values, 37, axis=1)), axis=2)
    mask = np.zeros(values.shape, np.uint8)
    mask[:, ::2] = 255
    expected = image.copy()
    selected = mask > 0
    expected[selected] = (image[selected] * 0.82 + np.array([30, 220, 30]) * 0.18).astype(
        np.uint8
    )
    actual = render_overlay(
        image, Detection(mask=mask), ProcessingConfig(), show_obb=False, show_axis=False,
    )
    # Exclude the status text at the top; all tinted and untouched pixels below
    # it must exactly match the former renderer, including all BGR channel values.
    np.testing.assert_array_equal(actual[40:], expected[40:])


def test_invalid_environment_setting_is_rejected_before_changing_opencv(monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "setNumThreads", calls.append)
    monkeypatch.setenv("BIBAZU_OPENCV_THREADS", "not-an-integer")
    with pytest.raises(ValueError, match="positive ganze Zahl"):
        performance.configure_opencv_threads()
    assert calls == []


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_thread_count_does_not_change_opencv(monkeypatch, value):
    calls = []
    monkeypatch.setattr(cv2, "setNumThreads", calls.append)
    with pytest.raises(ValueError, match="positive ganze Zahl"):
        performance.configure_opencv_threads(value)
    assert calls == []


def test_main_sets_thread_budget_before_starting_qt_or_worker_threads(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        sys, "argv", ["bibazu-vision", "--opencv-threads", "2", "--data-dir", str(tmp_path)],
    )
    monkeypatch.setattr(
        app_module, "configure_opencv_threads", lambda count: calls.append(("threads", count)) or 2,
    )

    def application(argv):
        calls.append("application")
        return SimpleNamespace(exec=lambda: 0)

    def window(*args, **kwargs):
        calls.append("window")
        return SimpleNamespace(note=lambda text: calls.append(text), show=lambda: None)

    monkeypatch.setattr(app_module, "QApplication", application)
    monkeypatch.setattr(app_module, "MainWindow", window)
    assert app_module.main() == 0
    assert calls == [("threads", 2), "application", "window", "OpenCV: 2 Rechenthread(s)."]


def test_thread_counts_preserve_angles_geometry_motion_and_masks_in_isolated_process():
    # Do not change the process-wide OpenCV setting while Qt test workers may run.
    script = """
import cv2, hashlib, json
import numpy as np
from bibazu_vision.performance import configure_opencv_threads
from bibazu_vision.processing import FrameProcessor, ProcessingConfig
from bibazu_vision.recording import _measurement
results = []
for threads in (1, 2, 4):
    configure_opencv_threads(threads)
    config = ProcessingConfig(method='pca', roi=(100,70,600,360), scale=0.25)
    processor = FrameProcessor(config)
    background = np.full((480,800),180,np.uint8)
    processor.set_background(background)
    rows = []
    for index, angle in enumerate((0, 17, -30)):
        image = background.copy()
        box = cv2.boxPoints(((350+index*5,240),(220,60),angle))
        cv2.fillPoly(image,[np.rint(box).astype(np.int32)],15)
        detection = processor.process(image)
        assert detection.mask is None
        row = _measurement(detection)
        row.pop('processing_ms')
        row['mask_sha256'] = hashlib.sha256(detection.get_mask().tobytes()).hexdigest()
        rows.append(row)
    results.append(rows)
assert results[0] == results[1] == results[2]
print(json.dumps({'threads': [1,2,4], 'identical': True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20, check=True,
    )
    assert json.loads(result.stdout) == {"threads": [1, 2, 4], "identical": True}
