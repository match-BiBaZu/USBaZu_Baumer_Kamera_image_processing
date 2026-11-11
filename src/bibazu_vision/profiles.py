"""Camera-specific preferences and migration of the original USB preferences."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .processing import ProcessingConfig

USB_PROFILE = "usb"
VLXT_PROFILE = "vlxt"
VLXT_SERIAL = "700006383255"
CAMERA_LABELS = {
    USB_PROFILE: "Kleine USB-Kamera (VCXU-02C)",
    VLXT_PROFILE: f"VLXT-28M.I · SN {VLXT_SERIAL}",
}
CAMERA_FIELDS = (
    "cti", "serial", "exposure", "fps", "camera", "processing", "capture", "processing_geometry",
)


def default_profile(camera_id: str) -> dict:
    producer = "gige" if camera_id == VLXT_PROFILE else "usb"
    return {
        "cti": rf"C:\Program Files\Baumer Camera Explorer\bgapi2_{producer}.cti",
        "serial": VLXT_SERIAL if camera_id == VLXT_PROFILE else "",
        "exposure": 1000,
        "fps": 200,
        "camera": {
            "mode": "live", "trigger_line": "Line0", "trigger_activation": "RisingEdge",
        },
        "processing": asdict(ProcessingConfig()),
        "processing_geometry": None,
        "capture": {
            "start_mode": "hardware", "quiet_seconds": 0.15, "max_seconds": 5,
            "max_memory_mb": 256, "motion_threshold": 0.001,
        },
    }


def migrate_settings(saved: dict) -> dict:
    """Read legacy data without writing it (in particular while running a demo)."""
    stored = saved.get("camera_profiles")
    if not isinstance(stored, dict):
        stored = {USB_PROFILE: {key: saved[key] for key in CAMERA_FIELDS if key in saved}}
    profiles = {}
    for camera_id in CAMERA_LABELS:
        profile = default_profile(camera_id)
        for key, value in deepcopy(stored.get(camera_id, {})).items():
            if key in {"camera", "processing", "capture"}:
                profile[key].update(value)
            elif key in CAMERA_FIELDS:
                profile[key] = value
        profiles[camera_id] = profile
    selected = saved.get("selected_camera", VLXT_PROFILE)
    return {
        "version": 2,
        "selected_camera": selected if selected in CAMERA_LABELS else VLXT_PROFILE,
        "camera_profiles": profiles,
        "language": saved.get("language", "de"),
        "preview": deepcopy(saved.get("preview", {})),
        **({"output_dir": saved["output_dir"]} if "output_dir" in saved else {}),
    }


def camera_data_dir(data_dir: Path, camera_id: str) -> Path:
    return data_dir / ("camera" if camera_id == USB_PROFILE else f"camera_vlxt_{VLXT_SERIAL}")
