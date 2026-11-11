"""Map sensor-coordinate backgrounds and software ROIs across camera crops."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def frame_geometry(diagnostics, image):
    actual = diagnostics.get("actual", {})
    height, width = image.shape[:2]
    return {
        "width": width, "height": height,
        "offset_x": int(actual.get("OffsetX") or 0),
        "offset_y": int(actual.get("OffsetY") or 0),
    }


def translated_roi(roi, source, target):
    if roi is None or source is None:
        return roi
    x, y, width, height = roi
    x += source["offset_x"] - target["offset_x"]
    y += source["offset_y"] - target["offset_y"]
    x1, y1 = min(target["width"], x + width), min(target["height"], y + height)
    x, y = max(0, x), max(0, y)
    result = [x, y, x1 - x, y1 - y] if x1 > x and y1 > y else None
    return roi if result == list(roi) else result


def load_background(path: Path):
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Hintergrundbild konnte nicht gelesen werden.")
    metadata = path.with_suffix(".json")
    geometry = None
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["sha256"] != hashlib.sha256(data).hexdigest():
            raise ValueError("Hintergrundbild und Bildausschnitt-Metadaten passen nicht zusammen.")
        geometry = saved["geometry"]
        valid_geometry = isinstance(geometry, dict) and all(
            type(geometry.get(key)) is int
            and geometry[key] >= (1 if key in {"width", "height"} else 0)
            for key in ("width", "height", "offset_x", "offset_y")
        )
        if not valid_geometry or (geometry["height"], geometry["width"]) != image.shape[:2]:
            raise ValueError("Hintergrundbild und Bildausschnitt-Metadaten passen nicht zusammen.")
    return image, geometry


def aligned_background(image, saved_geometry, base_geometry, target):
    # Legacy PNGs have no origin metadata. Only infer it when the image covers
    # the original pre-crop camera frame, never merely from an equal cropped size.
    source = saved_geometry
    if source is None:
        if image.shape[:2] != (base_geometry["height"], base_geometry["width"]):
            return None
        source = base_geometry
    dx = target["offset_x"] - source["offset_x"]
    dy = target["offset_y"] - source["offset_y"]
    width, height = target["width"], target["height"]
    if dx < 0 or dy < 0 or dx + width > source["width"] or dy + height > source["height"]:
        return None
    return image[dy:dy + height, dx:dx + width]


def save_background(path: Path, image, geometry):
    temporary_image = path.with_name("background.tmp.png")
    temporary_metadata = path.with_suffix(".json.tmp")
    if not cv2.imwrite(str(temporary_image), image):
        raise OSError("Hintergrund konnte nicht gespeichert werden.")
    metadata = {
        "geometry": geometry,
        "sha256": hashlib.sha256(temporary_image.read_bytes()).hexdigest(),
    }
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # A hash prevents an old metadata file from silently identifying a new PNG
    # as the old origin if publication is interrupted between these two replaces.
    temporary_image.replace(path)
    temporary_metadata.replace(path.with_suffix(".json"))
