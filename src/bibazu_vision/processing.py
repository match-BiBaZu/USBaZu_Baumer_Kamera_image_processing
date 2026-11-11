"""Fast, training-free segmentation and in-plane orientation measurement.

Angles use mathematical coordinates (image y points down, positive angles are
counterclockwise). They describe an *axis*, so 180 degree rotations are identical.
The major axis need not be parallel to the support surface of an irregular part:
calibrate ``zero_offset_deg`` from that part while it rests on the chute. This is
an image-plane measurement, not a measurement of 3D tilt or height.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# Match the previous float64 tint followed by uint8 truncation, using a native
# per-channel lookup instead of allocating/indexing floating-point pixel arrays.
_CONTOUR_TINT = (
    np.arange(256, dtype=np.float64)[:, None, None] * 0.82
    + np.array([30, 220, 30], dtype=np.float64)[None, None, :] * 0.18
).astype(np.uint8)


@dataclass
class ProcessingConfig:
    """Mutable controls; ROI and minimum area are in original image pixels.

    With a background, ``threshold`` is the absolute background difference.
    Without one, it is the absolute intensity cutoff for dark/light foreground.
    ``scale`` reduces computation only; all returned geometry uses original pixels.
    ``zero_offset_deg`` is the measured resting angle *relative to the chute*.
    ``ignore_border_touch`` allows a valid axis to be used even when the contour
    reaches the image border; useful when the part is intentionally close to the
    ROI edge and a rough measurement is preferred over rejecting the frame.
    """

    threshold: float = 25
    min_area: float = 150
    roi: tuple[int, int, int, int] | list[int] | None = None
    chute_angle_deg: float = 0
    zero_offset_deg: float = 0
    method: str = "obb"
    foreground: str = "dark"
    scale: float = 1.0
    ignore_border_touch: bool = False


@dataclass(frozen=True)
class _MaskSource:
    """Owned segmentation at processing resolution, with frozen image geometry."""

    binary: np.ndarray
    image_shape: tuple[int, int]
    bounds: tuple[int, int, int, int]

    def get_region(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds
        return (
            self.binary if self.binary.shape == (y1 - y0, x1 - x0)
            else cv2.resize(self.binary, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        )

    def materialize(self, region: np.ndarray | None = None) -> np.ndarray:
        height, width = self.image_shape
        x0, y0, x1, y1 = self.bounds
        if region is None:
            region = self.get_region()
        if self.bounds == (0, 0, width, height):
            return region
        mask = np.zeros(self.image_shape, np.uint8)
        mask[y0:y1, x0:x1] = region
        return mask


@dataclass
class Detection:
    # Full-sized tint mask is optional. Measurements and recordings never need it.
    mask: np.ndarray | None
    contour: np.ndarray | None = None
    box: np.ndarray | None = None
    center: tuple[float, float] | None = None
    axis_deg: float | None = None
    angle_deg: float | None = None
    confidence: float = 0.0
    area: float = 0.0
    touching_border: bool = False
    motion_fraction: float = 0.0
    processing_ms: float = 0.0
    _mask_source: _MaskSource | None = field(default=None, repr=False)
    _mask_region: np.ndarray | None = field(default=None, repr=False)

    def get_mask(self) -> np.ndarray:
        """Explicit full-sized mask access for analysis clients; previews use the ROI."""
        if self.mask is None:
            if self._mask_source is None:
                raise ValueError("Detection has no segmentation mask.")
            self.mask = self._mask_source.materialize(self._mask_region)
        return self.mask

    def get_mask_region(self, image_shape):
        """Lazily enlarge only the ROI needed for a displayed segmentation tint."""
        if self.mask is not None:
            if self.mask.shape != image_shape:
                raise ValueError("Detection mask size differs from image.")
            height, width = image_shape
            return (0, 0, width, height), self.mask
        if self._mask_source is None:
            raise ValueError("Detection has no segmentation mask.")
        if self._mask_source.image_shape != image_shape:
            raise ValueError("Detection mask size differs from image.")
        if self._mask_region is None:
            self._mask_region = self._mask_source.get_region()
        return self._mask_source.bounds, self._mask_region


def wrap_axis_angle(angle: float) -> float:
    """Normalize an undirected axis angle to [-90, 90)."""
    return float((angle + 90.0) % 180.0 - 90.0)


def _gray(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Camera images must be uint8 NumPy arrays.")
    if image.size == 0:
        raise ValueError("Camera image is empty.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(image, code)
    raise ValueError("Expected a mono, BGR, or BGRA image.")


def _validate(config: ProcessingConfig) -> None:
    if not math.isfinite(config.threshold) or not 0 <= config.threshold <= 255:
        raise ValueError("threshold must be between 0 and 255.")
    if not math.isfinite(config.min_area) or config.min_area < 0:
        raise ValueError("min_area must be nonnegative.")
    if not math.isfinite(config.scale) or not 0 < config.scale <= 1:
        raise ValueError("scale must be greater than 0 and at most 1.")
    if not math.isfinite(config.chute_angle_deg) or not math.isfinite(config.zero_offset_deg):
        raise ValueError("Reference angles must be finite.")
    if config.method not in ("obb", "pca"):
        raise ValueError("method must be 'obb' or 'pca'.")
    if config.foreground not in ("dark", "light"):
        raise ValueError("foreground must be 'dark' or 'light'.")


def _roi_bounds(config: ProcessingConfig, width: int, height: int) -> tuple[int, int, int, int]:
    if config.roi is None:
        return 0, 0, width, height
    if len(config.roi) != 4:
        raise ValueError("ROI must be (x, y, width, height).")
    if any(not math.isfinite(v) or int(v) != v for v in config.roi):
        raise ValueError("ROI coordinates must be finite integers.")
    x, y, w, h = map(int, config.roi)
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive.")
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("ROI does not intersect the camera image.")
    return x0, y0, x1, y1


class FrameProcessor:
    """Stateful processor; use from one worker thread in frame order.

    The stored background never adapts to moving or stationary foreground.
    Inter-frame motion is the fraction of ROI pixels whose intensity changes by
    more than a noise floor (5..25 gray levels). Its first sample is zero.
    """

    def __init__(self, config: ProcessingConfig | None = None) -> None:
        self.config = config if config is not None else ProcessingConfig()
        _validate(self.config)
        self._background: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._previous_key: tuple[int, ...] | None = None
        self._kernel = np.ones((3, 3), np.uint8)

    @property
    def has_background(self) -> bool:
        return self._background is not None

    def set_background(self, image: np.ndarray) -> None:
        self._background = _gray(image).copy()
        self._previous = None
        self._previous_key = None

    def clear_background(self) -> None:
        self._background = None
        self._previous = None
        self._previous_key = None

    def process(self, image: np.ndarray) -> Detection:
        started = time.perf_counter()
        _validate(self.config)
        gray = _gray(image)
        height, width = gray.shape
        x0, y0, x1, y1 = _roi_bounds(self.config, width, height)
        crop = gray[y0:y1, x0:x1]
        source_height, source_width = crop.shape
        target_size = (
            max(1, round(source_width * self.config.scale)),
            max(1, round(source_height * self.config.scale)),
        )
        if target_size != (source_width, source_height):
            working = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        else:
            working = crop
        working_height, working_width = working.shape
        # Actual x/y scales can differ slightly when resize dimensions are rounded.
        factor_x, factor_y = source_width / working_width, source_height / working_height

        key = (height, width, x0, y0, x1, y1, working_width, working_height)
        motion = 0.0
        if self._previous is not None and self._previous_key == key:
            change = cv2.absdiff(working, self._previous)
            motion_threshold = max(5.0, min(25.0, self.config.threshold))
            motion = float(np.count_nonzero(change > motion_threshold) / change.size)
        self._previous = working.copy()
        self._previous_key = key

        if self._background is not None:
            if self._background.shape != gray.shape:
                raise ValueError(
                    "Background size differs from camera image; capture a new background."
                )
            background = self._background[y0:y1, x0:x1]
            if target_size != (source_width, source_height):
                background = cv2.resize(background, target_size, interpolation=cv2.INTER_AREA)
            difference = cv2.absdiff(working, background)
            _, binary = cv2.threshold(difference, self.config.threshold, 255, cv2.THRESH_BINARY)
        else:
            mode = cv2.THRESH_BINARY_INV if self.config.foreground == "dark" else cv2.THRESH_BINARY
            _, binary = cv2.threshold(working, self.config.threshold, 255, mode)

        # Two small morphology passes suppress sensor speckles and close tiny gaps.
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel)
        detection = Detection(
            mask=None, motion_fraction=motion,
            _mask_source=_MaskSource(binary, (height, width), (x0, y0, x1, y1)),
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            detection.processing_ms = (time.perf_counter() - started) * 1000
            return detection
        local_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(local_contour) * factor_x * factor_y)
        if area < self.config.min_area or area <= 0:
            detection.processing_ms = (time.perf_counter() - started) * 1000
            return detection

        points = local_contour.reshape(-1, 2)
        touching = bool(
            np.any(points[:, 0] <= 0)
            or np.any(points[:, 1] <= 0)
            or np.any(points[:, 0] >= working_width - 1)
            or np.any(points[:, 1] >= working_height - 1)
        )
        contour = local_contour.astype(np.float32)
        contour[:, 0, 0] = (contour[:, 0, 0] + 0.5) * factor_x - 0.5 + x0
        contour[:, 0, 1] = (contour[:, 0, 1] + 0.5) * factor_y - 0.5 + y0
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        edges = np.roll(box, -1, axis=0) - box
        lengths = np.linalg.norm(edges, axis=1)
        major_edge = edges[int(np.argmax(lengths))]
        long_side, short_side = float(max(lengths)), float(min(lengths))

        if self.config.method == "obb":
            axis = wrap_axis_angle(-math.degrees(math.atan2(major_edge[1], major_edge[0])))
            anisotropy = 1.0 - short_side / max(long_side, 1e-12)
            center = tuple(map(float, rect[0]))
        else:
            # Area moments yield PCA of the filled silhouette. PCA of the
            # CHAIN_APPROX_SIMPLE vertices would overweight corners and notches.
            moments = cv2.moments(contour)
            mass = moments["m00"]
            covariance = np.array(
                [
                    [moments["mu20"] / mass, moments["mu11"] / mass],
                    [moments["mu11"] / mass, moments["mu02"] / mass],
                ]
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            major = eigenvectors[:, -1]
            axis = wrap_axis_angle(-math.degrees(math.atan2(major[1], major[0])))
            anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-12))
            center = (float(moments["m10"] / mass), float(moments["m01"] / mass))

        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = min(1.0, area / max(hull_area, 1e-12))
        confidence = min(1.0, max(0.0, anisotropy)) * math.sqrt(solidity)
        if touching:
            confidence *= 0.5
        if area / (source_width * source_height) > 0.8:
            confidence *= 0.1

        detection.contour = contour
        detection.box = box
        detection.center = center
        detection.area = area
        detection.touching_border = touching
        detection.confidence = float(confidence)
        # A near square or isotropic shape has no reliable major axis. Keep its
        # outline visible while declining to display a spurious angle.
        if anisotropy >= 0.08:
            detection.axis_deg = axis
            # Cropping can rotate the major axis by 90 degrees as a part leaves
            # the ROI. Keep the geometric overlay, but do not publish a tilt unless
            # the user explicitly allows border-touch measurements.
            if not touching or self.config.ignore_border_touch:
                detection.angle_deg = wrap_axis_angle(
                    axis - self.config.chute_angle_deg - self.config.zero_offset_deg
                )
        detection.processing_ms = (time.perf_counter() - started) * 1000
        return detection


def render_overlay(
    image: np.ndarray,
    detection: Detection,
    config: ProcessingConfig,
    *,
    show_contour: bool = True,
    show_obb: bool = True,
    show_axis: bool = True,
) -> np.ndarray:
    """Draw independent display layers on a BGR copy, without changing measurements.

    Contour includes the segmentation tint; axis includes its center marker and
    the chute reference. ROI and status remain visible while any layer is on.
    """
    gray = _gray(image)
    if image.ndim == 3 and image.shape[2] == 3:
        output = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        output = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if not (show_contour or show_obb or show_axis):
        return output
    if show_contour:
        (mx0, my0, mx1, my1), mask = detection.get_mask_region(gray.shape)
        region = output[my0:my1, mx0:mx1]
        cv2.copyTo(cv2.LUT(region, _CONTOUR_TINT), mask, region)
    height, width = gray.shape
    x0, y0, x1, y1 = _roi_bounds(config, width, height)
    if config.roi is not None:
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (180, 180, 180), 1)
    if show_contour and detection.contour is not None:
        cv2.drawContours(
            output, [np.rint(detection.contour).astype(np.int32)], -1, (70, 240, 70), 2
        )
    if show_obb and detection.box is not None:
        cv2.polylines(output, [np.rint(detection.box).astype(np.int32)], True, (255, 200, 20), 2)
    if show_axis and detection.center is not None:
        center = np.array(detection.center)
        cv2.circle(output, tuple(np.rint(center).astype(int)), 4, (255, 255, 255), -1)
        if detection.axis_deg is not None:
            angle = math.radians(detection.axis_deg)
            direction = np.array([math.cos(angle), -math.sin(angle)])
            extent = 50.0
            if detection.box is not None:
                extent = float(
                    np.linalg.norm(np.roll(detection.box, -1, axis=0) - detection.box, axis=1).max()
                    / 2
                )
            start = tuple(np.rint(center - direction * extent).astype(int))
            end = tuple(np.rint(center + direction * extent).astype(int))
            cv2.line(output, start, end, (230, 70, 230), 2, cv2.LINE_AA)
    if show_axis:
        # A short reference line near the ROI bottom marks the calibrated chute angle.
        anchor = np.array([x0 + 15.0, y1 - 20.0])
        reference_length = min(140, max(15, (x1 - x0) // 3))
        chute_radians = math.radians(config.chute_angle_deg)
        reference_end = (
            anchor
            + np.array([math.cos(chute_radians), -math.sin(chute_radians)]) * reference_length
        )
        cv2.line(
            output,
            tuple(np.rint(anchor).astype(int)),
            tuple(np.rint(reference_end).astype(int)),
            (30, 180, 255),
            2,
        )
    angle_text = "ambiguous" if detection.contour is not None else "no part"
    if detection.touching_border and not config.ignore_border_touch:
        angle_text = "clipped"
    if detection.angle_deg is not None:
        angle_text = f"{detection.angle_deg:+.1f} deg"
    text = f"{config.method.upper()}  {angle_text}  quality {detection.confidence:.2f}"
    if detection.touching_border and not config.ignore_border_touch:
        text += "  ROI BORDER"
    cv2.putText(output, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        output, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
    )
    return output
