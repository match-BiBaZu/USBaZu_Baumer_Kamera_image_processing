"""Compare CPU processing/overlay cost on frozen images, without opening a camera."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bibazu_vision import processing  # noqa: E402
from bibazu_vision.recording import _measurement  # noqa: E402


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("processing_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def processor(module, config, background):
    result = module.FrameProcessor(module.ProcessingConfig(**config))
    if background is not None:
        result.set_background(background)
    return result


def check_equivalence(baseline, config, background, images):
    """Verify angles, motion, geometry, full masks and all display layers before timing."""
    old = processor(baseline, config, background)
    new = processor(processing, config, background)
    for image in images:
        expected, actual = old.process(image), new.process(image)
        old_values, new_values = _measurement(expected), _measurement(actual)
        old_values.pop("processing_ms")
        new_values.pop("processing_ms")
        assert old_values == new_values, (old_values, new_values)
        if expected.contour is not None:
            np.testing.assert_array_equal(actual.contour, expected.contour)
        for flags in ((True, True, True), (False, True, True), (False, False, False)):
            options = dict(zip(("show_contour", "show_obb", "show_axis"), flags, strict=True))
            np.testing.assert_array_equal(
                processing.render_overlay(image, actual, new.config, **options),
                baseline.render_overlay(image, expected, old.config, **options),
            )
        np.testing.assert_array_equal(actual.get_mask(), expected.mask)


def measure(module, config, background, images, iterations, warmup, preview_every):
    engine = processor(module, config, background)
    totals, analysis, built_masks = [], [], 0
    for index in range(warmup + iterations):
        image = images[index % len(images)]
        started = time.perf_counter()
        detection = engine.process(image)
        measured = time.perf_counter()
        if preview_every and index % preview_every == 0:
            module.render_overlay(image, detection, engine.config)
        finished = time.perf_counter()
        if index >= warmup:
            totals.append((finished - started) * 1000)
            analysis.append((measured - started) * 1000)
            built_masks += detection.mask is not None
    return {
        "measurement_median_ms": float(np.median(analysis)),
        "total_median_ms": float(np.median(totals)),
        "total_mean_ms": float(np.mean(totals)),
        "total_p95_ms": float(np.percentile(totals, 95)),
        "full_masks_built": int(built_masks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--config", type=Path, help="JSON containing ProcessingConfig fields")
    parser.add_argument(
        "--baseline", type=Path, help="Previous standalone processing.py to compare",
    )
    parser.add_argument("--threads", nargs="+", type=int)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_threads = cv2.getNumThreads()
    counts = list(dict.fromkeys(args.threads or [1, 2, 4, original_threads]))
    if any(count < 1 for count in counts) or args.iterations < 1 or args.warmup < 0:
        parser.error("Thread counts and iterations must be positive; warmup must be nonnegative.")
    if args.preview_every < 1:
        parser.error("preview-every must be positive.")
    config = asdict(processing.ProcessingConfig())
    if args.config:
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    images = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in args.images]
    background = cv2.imread(str(args.background), cv2.IMREAD_UNCHANGED) if args.background else None
    if any(image is None for image in images) or (args.background and background is None):
        parser.error("Could not read an input image.")
    modules = {"current": processing}
    if args.baseline:
        modules = {"baseline": load_baseline(args.baseline), **modules}
    report = {
        "opencv_version": cv2.__version__, "default_opencv_threads": original_threads,
        "images": [str(path) for path in args.images],
        "shapes": [list(image.shape) for image in images],
        "processing": config, "iterations": args.iterations,
        "scope": (
            "CPU analysis and optional overlay only; excludes acquisition, Qt, JPEG and disk IO"
        ),
        "equivalence_verified": False, "results": [],
    }
    try:
        print("Implementation  Threads  Preview every  Analysis median ms  Total mean ms  P95 ms")
        for count in counts:
            cv2.setNumThreads(count)
            if "baseline" in modules:
                check_equivalence(modules["baseline"], config, background, images)
                report["equivalence_verified"] = True
            for name, module in modules.items():
                for every in (0, args.preview_every):
                    row = {
                        "implementation": name, "threads": cv2.getNumThreads(),
                        "preview_every": every,
                        **measure(
                            module, config, background, images,
                            args.iterations, args.warmup, every,
                        ),
                    }
                    report["results"].append(row)
                    print(
                        f"{name:15s} {count:7d} {every:14d} "
                        f"{row['measurement_median_ms']:19.3f} "
                        f"{row['total_mean_ms']:14.3f} {row['total_p95_ms']:7.3f}"
                    )
    finally:
        cv2.setNumThreads(original_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", args.output)


if __name__ == "__main__":
    main()
