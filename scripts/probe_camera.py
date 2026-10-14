"""Inspect the connected camera and save a persistent snapshot, then restore it."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bibazu_vision.camera import BaumerCamera, CameraConfig, find_cti_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Baumer USB-Kamera prüfen und ein Bild speichern.")
    parser.add_argument(
        "--cti", help="GenTL-Producer (.cti); automatische Baumer USB-Suche wenn leer"
    )
    parser.add_argument("--serial")
    parser.add_argument("--exposure-us", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--line", default="Line0")
    parser.add_argument(
        "--input-monitor", choices=["auto", "counter", "line_status"], default="auto"
    )
    parser.add_argument("--output", type=Path, default=ROOT / ".diagnostics" / "current_camera.png")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("CTI:", args.cti or find_cti_paths())
    camera = BaumerCamera(
        CameraConfig(
            cti_path=args.cti,
            serial=args.serial,
            exposure_us=args.exposure_us,
            fps=args.fps,
            trigger_line=args.line,
            input_monitor=args.input_monitor,
        )
    )
    result = 0
    try:
        camera.open()
        deadline = time.perf_counter() + args.timeout
        frame = None
        while frame is None and time.perf_counter() < deadline:
            frame = camera.read(timeout=min(0.5, max(0.01, deadline - time.perf_counter())))
        if frame is None:
            raise RuntimeError("Kein Bild innerhalb des Zeitlimits empfangen.")
        if not cv2.imwrite(str(args.output), frame.image):
            raise RuntimeError(f"Bild konnte nicht gespeichert werden: {args.output}")
        camera.diagnostics["snapshot"] = {
            "path": str(args.output),
            "shape": list(frame.image.shape),
            "frame_id": frame.frame_id,
        }
        print("Bild:", args.output, frame.image.shape, "Frame", frame.frame_id)
    except Exception as exc:
        camera.diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        print(camera.diagnostics["error"], file=sys.stderr)
        result = 1
    finally:
        camera.close()
        target = args.output.parent / "camera_diagnostics.json"
        target.write_text(
            json.dumps(camera.diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Diagnose:", target)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
