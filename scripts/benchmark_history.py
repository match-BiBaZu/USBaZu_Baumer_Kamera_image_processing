"""Compare history bookkeeping at fixed sizes; excludes JPEG encoding and disk I/O."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bibazu_vision.history import HistorySample, _PreviewFiles  # noqa: E402


class FileWithoutIO:
    def unlink(self, *, missing_ok):
        pass


def measure(count, iterations, mode, optimized):
    """Advance a full buffer by one frame, keeping its size constant."""
    path = FileWithoutIO()
    index = _PreviewFiles(count)
    files = {}
    for key in range(count):
        if optimized:
            index.add(HistorySample(key, key / 60, key, 0), path, 1)
        else:
            files[key] = (key / 60, path, 1)
    used_bytes = count
    times = []
    for step in range(iterations):
        cutoff = (step + 0.5) / 60 if mode == "age" else float("-inf")
        key = count + step
        started = time.perf_counter_ns()
        if optimized:
            index.prune(cutoff, required_bytes=1)
            index.add(HistorySample(key, key / 60, key, 0), path, 1)
        else:
            # Previous archive implementation: scan every entry, then find min key.
            for old_key, (timestamp, old_path, size) in list(files.items()):
                if timestamp >= cutoff:
                    continue
                old_path.unlink(missing_ok=True)
                del files[old_key]
                used_bytes -= size
            while files and used_bytes + 1 > count:
                old_key = min(files)
                _, oldest, size = files[old_key]
                oldest.unlink(missing_ok=True)
                del files[old_key]
                used_bytes -= size
            files[key] = (key / 60, path, 1)
            used_bytes += 1
        times.append((time.perf_counter_ns() - started) / 1000)
    if optimized:
        assert index.used_bytes == count and len(index.files) == count
        assert min(index.files) == iterations
    else:
        assert used_bytes == count and len(files) == count
        assert min(files) == iterations
    return {
        "median_us": statistics.median(times),
        "p95_us": sorted(times)[int((len(times) - 1) * 0.95)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 6000, 36000])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and iterations must be positive")
    results = []
    for count in args.sizes:
        for mode in ("age", "quota"):
            before = measure(count, args.iterations, mode, False)
            after = measure(count, args.iterations, mode, True)
            result = {"entries": count, "mode": mode, "before": before, "after": after}
            results.append(result)
            print(
                f"{count:6d} {mode:5s}: median {before['median_us']:.1f} -> "
                f"{after['median_us']:.1f} us; p95 {before['p95_us']:.1f} -> "
                f"{after['p95_us']:.1f} us"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
