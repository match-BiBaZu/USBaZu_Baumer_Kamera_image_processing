"""Behavioral checks for arbitrary silhouettes, angle calibration and isolation."""

import math

import cv2
import numpy as np
import pytest

from bibazu_vision.processing import (
    FrameProcessor,
    ProcessingConfig,
    render_overlay,
    wrap_axis_angle,
)


def silhouette(angle=0, *, irregular=False, center=(180, 160), background=230, foreground=15):
    image = np.full((320, 360), background, np.uint8)
    if irregular:
        polygon = np.array(
            [[-78, -24], [40, -24], [40, -12], [78, -12], [78, 24], [-78, 24]], np.float32
        )
    else:
        polygon = np.array([[-78, -24], [78, -24], [78, 24], [-78, 24]], np.float32)
    radians = math.radians(angle)
    rotation = np.array(
        [[math.cos(radians), math.sin(radians)], [-math.sin(radians), math.cos(radians)]]
    )
    points = polygon @ rotation.T + center
    cv2.fillPoly(image, [np.rint(points).astype(np.int32)], foreground)
    return image


@pytest.mark.parametrize("method", ["obb", "pca"])
@pytest.mark.parametrize("angle", [-77, -33, 0, 24, 69, 88])
def test_rotated_part_angle_uses_math_convention(method, angle):
    processor = FrameProcessor(ProcessingConfig(method=method))
    result = processor.process(silhouette(angle))
    assert result.angle_deg is not None
    assert abs(wrap_axis_angle(result.angle_deg - angle)) < 0.8
    assert result.confidence > 0.65
    assert result.center == pytest.approx((180, 160), abs=0.8)
    assert result.area > 7000


@pytest.mark.parametrize("method", ["obb", "pca"])
@pytest.mark.parametrize("angle", [-58, 37, 82])
def test_irregular_outline_changes_need_resting_zero_not_shape_training(method, angle):
    config = ProcessingConfig(method=method, chute_angle_deg=12)
    processor = FrameProcessor(config)
    at_rest = processor.process(silhouette(12, irregular=True))
    assert at_rest.angle_deg is not None
    config.zero_offset_deg = at_rest.angle_deg
    assert processor.process(silhouette(12, irregular=True)).angle_deg == pytest.approx(0, abs=1e-6)
    rotated = processor.process(silhouette(12 + angle, irregular=True))
    assert rotated.angle_deg is not None
    assert abs(wrap_axis_angle(rotated.angle_deg - angle)) < 1.0


def test_background_difference_detects_both_dark_and_light_without_adapting():
    background = np.full((320, 360), 120, np.uint8)
    processor = FrameProcessor(ProcessingConfig(threshold=25))
    processor.set_background(background)
    background[:] = 0  # Stored background is an owned snapshot.
    assert processor.has_background
    for color in (15, 220):
        image = silhouette(27, background=120, foreground=color)
        for _ in range(4):
            result = processor.process(image)
            assert result.angle_deg == pytest.approx(27, abs=0.8)
    result = processor.process(np.full((320, 360), 120, np.uint8))
    assert result.contour is None
    assert not result.mask.any()
    processor.clear_background()
    assert not processor.has_background


def test_absolute_light_segmentation_accepts_bgr_image():
    processor = FrameProcessor(ProcessingConfig(threshold=180, foreground="light"))
    image = cv2.cvtColor(silhouette(-22, background=15, foreground=230), cv2.COLOR_GRAY2BGR)
    result = processor.process(image)
    assert result.angle_deg == pytest.approx(-22, abs=0.8)
    overlay = render_overlay(image, result, processor.config)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, image)


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.3])
def test_roi_excludes_distractor_and_returns_original_coordinates(scale):
    image = silhouette(24, center=(220, 190))
    cv2.rectangle(image, (3, 2), (125, 110), 0, -1)  # Larger, unrelated foreground.
    processor = FrameProcessor(ProcessingConfig(roi=(130, 105, 190, 185), scale=scale))
    result = processor.process(image)
    assert result.angle_deg == pytest.approx(24, abs=1.5)
    assert result.center == pytest.approx((220, 190), abs=3)
    assert result.mask.shape == image.shape
    assert not result.mask[:105].any()
    assert not result.mask[:, :130].any()
    # At small scales, interpolated edge intensities and contour pixel centers
    # can reduce area; it must still be in original, not downsampled, pixel units.
    assert result.area == pytest.approx(156 * 48, rel=0.2)
    assert not result.touching_border


@pytest.mark.parametrize("method", ["obb", "pca"])
def test_square_is_visible_but_has_no_reliable_axis(method):
    image = np.full((200, 200), 230, np.uint8)
    square = cv2.boxPoints(((100, 100), (76, 76), 26))
    cv2.fillPoly(image, [np.rint(square).astype(np.int32)], 0)
    result = FrameProcessor(ProcessingConfig(method=method)).process(image)
    assert result.contour is not None
    assert result.box is not None
    assert result.angle_deg is None
    assert result.axis_deg is None
    assert result.confidence < 0.08


def test_no_part_tiny_noise_and_border_touch_are_distinguished():
    processor = FrameProcessor(ProcessingConfig(min_area=150))
    blank = np.full((320, 360), 230, np.uint8)
    blank[80:84, 60:64] = 0
    assert processor.process(blank).contour is None
    clipped = processor.process(silhouette(0, center=(34, 160)))
    assert clipped.contour is not None
    assert clipped.touching_border
    assert clipped.confidence < 0.5
    assert clipped.axis_deg is not None  # Geometric axis remains useful for the overlay.
    assert clipped.angle_deg is None  # A clipped silhouette must not claim a tilt measurement.


def test_motion_counts_change_in_roi_then_stops_for_stationary_object():
    config = ProcessingConfig(threshold=150, roi=(100, 70, 220, 230))
    processor = FrameProcessor(config)
    first = processor.process(silhouette(center=(180, 160), foreground=80))
    assert first.motion_fraction == 0
    shifted = processor.process(silhouette(center=(195, 160), foreground=80))
    assert shifted.motion_fraction > 0.01
    still = processor.process(silhouette(center=(195, 160), foreground=80))
    assert still.motion_fraction == 0
    outside = silhouette(center=(195, 160), foreground=80)
    outside[10:50, 10:50] = 0
    assert processor.process(outside).motion_fraction == 0
    config.roi = None
    assert processor.process(outside).motion_fraction == 0  # Geometry changes reset baseline.


@pytest.mark.parametrize(
    "angle, expected", [(90, -90), (-90, -90), (181, 1), (-181, -1), (270, -90)]
)
def test_angle_wrap_is_180_degree_periodic(angle, expected):
    assert wrap_axis_angle(angle) == expected


def test_zero_and_chute_offsets_cross_the_wrap_boundary():
    config = ProcessingConfig(chute_angle_deg=80, zero_offset_deg=10)
    result = FrameProcessor(config).process(silhouette(-85))
    assert result.axis_deg == pytest.approx(-85, abs=0.8)
    assert result.angle_deg == pytest.approx(5, abs=0.8)


@pytest.mark.parametrize(
    "changes",
    [
        dict(scale=0),
        dict(threshold=300),
        dict(method="bad"),
        dict(min_area=-1),
        dict(roi=(0, 0, 0, 20)),
    ],
)
def test_mutated_invalid_controls_raise_actionable_error(changes):
    config = ProcessingConfig()
    processor = FrameProcessor(config)
    for key, value in changes.items():
        setattr(config, key, value)
    with pytest.raises(ValueError):
        processor.process(silhouette())


def test_camera_resolution_change_requires_new_background():
    processor = FrameProcessor()
    processor.set_background(np.zeros((100, 100), np.uint8))
    with pytest.raises(ValueError, match="capture a new background"):
        processor.process(silhouette())
