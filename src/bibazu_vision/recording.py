"""Bounded, nonblocking passage capture and lossless clip replay.

Camera timestamps must use time.perf_counter(), a high-resolution monotonic
clock. The camera worker owns public methods; a background writer owns disk IO.
Image arrays are copied once on ingestion, then ownership is transferred to the
writer. The memory limit includes pre-roll, current capture and pending writes.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaptureConfig:
    pre_roll_seconds: float = 0.15
    quiet_seconds: float = 0.15
    wait_seconds: float = 2.0
    max_seconds: float = 5.0
    motion_threshold: float = 0.003
    max_memory_mb: float = 256.0
    start_mode: str = "motion"
    writer_queue_depth: int = 3

    def __post_init__(self) -> None:
        if self.start_mode not in {"motion", "hardware", "manual"}:
            raise ValueError("start_mode must be motion, hardware or manual")
        for name in ("quiet_seconds", "wait_seconds", "max_seconds", "max_memory_mb"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.pre_roll_seconds) or self.pre_roll_seconds < 0:
            raise ValueError("pre_roll_seconds must be nonnegative and finite")
        if not math.isfinite(self.motion_threshold) or not 0 <= self.motion_threshold <= 1:
            raise ValueError("motion_threshold must be between 0 and 1")
        if self.writer_queue_depth < 1:
            raise ValueError("writer_queue_depth must be positive")


@dataclass(frozen=True)
class CaptureStatus:
    state: str
    reason: str | None
    buffered_frames: int
    buffered_bytes: int
    dropped_frames: int
    writer_pending: int
    error: str | None = None
    rejected_triggers: int = 0


@dataclass(frozen=True)
class ClipSaveResult:
    path: Path | None
    reason: str
    frame_count: int
    dropped_frames: int
    error: str | None = None


@dataclass
class Clip:
    path: Path
    frames: list[np.ndarray]
    timestamps: list[float]
    frame_ids: list[int | str | None]
    measurements: list[dict[str, Any]]
    manifest: dict[str, Any]

    def export_csv(self, path: str | Path) -> Path:
        """Export measurements using the same columns as automatic clip export."""
        rows = [
            {"timestamp": ts, "frame_id": frame_id, "measurement": measurement}
            for ts, frame_id, measurement in zip(
                self.timestamps, self.frame_ids, self.measurements, strict=True
            )
        ]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(target, rows)
        return target


@dataclass
class _Frame:
    image: np.ndarray
    timestamp: float
    frame_id: int | str | None
    measurement: dict[str, Any]
    memory_bytes: int


@dataclass
class _Capture:
    start: float
    trigger_source: str
    created_utc: str
    capture_config: dict[str, Any]
    metadata: dict[str, Any]
    frames: list[_Frame] = field(default_factory=list)
    motion_seen: bool = False
    last_motion: float | None = None
    last_present: bool = False
    dropped_frames: int = 0
    pre_roll_truncated_frames: int = 0
    end: float | None = None
    reason: str = ""
    aborted: bool = False


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _scalar(value: Any) -> float | int | bool | str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _measurement(detection: Any) -> dict[str, Any]:
    """Keep only small overlay/measurement fields; never retain mask/contour."""
    fields = (
        "axis_deg",
        "angle_deg",
        "confidence",
        "area",
        "touching_border",
        "processing_ms",
        "motion_fraction",
        "motion_ratio",
        "motion_score",
    )
    result = {name: _scalar(_get(detection, name)) for name in fields}
    for name, shape in (("center", (2,)), ("box", (4, 2))):
        value = _get(detection, name)
        if value is not None:
            try:
                array = np.asarray(value, dtype=float)
                if array.shape == shape and np.all(np.isfinite(array)):
                    result[name] = array.tolist()
            except (TypeError, ValueError):
                pass
    explicit_presence = next(
        (
            _get(detection, name)
            for name in ("present", "found", "detected")
            if _get(detection, name) is not None
        ),
        None,
    )
    contour = _get(detection, "contour")
    if explicit_presence is not None:
        present = bool(explicit_presence)
    else:
        contour_present = contour is not None and np.size(contour) > 0
        area = result.get("area")
        present = (
            contour_present
            or result.get("center") is not None
            or (isinstance(area, (int, float)) and area > 0)
        )
    result["present"] = bool(present)
    return result


class CaptureManager:
    """Capture one passage at a time without performing disk IO in ingest.

    `arm()` enables automatic starts. `trigger()` starts a manual/hardware wait
    (and arms if needed). A passage finishes only after a part has moved, then
    disappeared and motion has stayed quiet. A stationary visible part therefore
    ends at the explicit duration cap, never as a successful passage.
    """

    _FRAME_OVERHEAD = 512

    def __init__(self, config: CaptureConfig, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.metadata: dict[str, Any] = {}
        self.last_clip_path: Path | None = None
        self.completed_clips: deque[ClipSaveResult] = deque(maxlen=64)
        self._lock = threading.RLock()
        self._saved = threading.Condition(self._lock)
        self._jobs: queue.Queue[_Capture] = queue.Queue(config.writer_queue_depth)
        self._closing = threading.Event()
        self._closed = False
        self._armed = False
        self._active: _Capture | None = None
        self._pre_roll: deque[_Frame] = deque()
        self._bytes = 0
        self._frames = 0
        self._pending = 0
        self._dropped_frames = 0
        self._rejected_triggers = 0
        self._pre_roll_truncated = 0
        self._last_timestamp: float | None = None
        self._last_shape: tuple[int, ...] | None = None
        self._reason: str | None = None
        self._error: str | None = None
        self._thread = threading.Thread(target=self._writer, name="clip-writer", daemon=True)
        self._thread.start()

    @property
    def status(self) -> CaptureStatus:
        with self._lock:
            return self._status()

    @property
    def last_clip(self) -> Path | None:
        return self.last_clip_path

    def _status(self) -> CaptureStatus:
        if self._active is not None:
            state = "moving" if self._active.motion_seen else "waiting"
        elif self._pending:
            state = "saving"
        else:
            state = "armed" if self._armed else "idle"
        return CaptureStatus(
            state,
            self._reason,
            self._frames,
            self._bytes,
            self._dropped_frames,
            self._pending,
            self._error,
            self._rejected_triggers,
        )

    def arm(self, now: float | None = None) -> CaptureStatus:
        with self._lock:
            self._ensure_open()
            self._armed = True
            self._reason = None
            self._error = None
            return self._status()

    def configure(self, config: CaptureConfig) -> CaptureStatus:
        """Apply next-capture settings only when no capture or write is pending."""
        with self._lock:
            self._ensure_open()
            if self._active is not None or self._pending:
                raise RuntimeError("configuration requires capture and writer to be idle")
            if config.writer_queue_depth != self.config.writer_queue_depth:
                raise ValueError("writer_queue_depth cannot change after construction")
            self._clear_pre_roll()
            self._pre_roll_truncated = 0
            self.config = config
            self._reason = None
            return self._status()

    def disarm(self) -> CaptureStatus:
        with self._lock:
            self._armed = False
            if self._active is not None:
                self._finish("manual_stop", self._last_timestamp or self._active.start)
            self._clear_pre_roll()
            return self._status()

    def trigger(self, now: float | None = None) -> CaptureStatus:
        with self._lock:
            self._ensure_open()
            self._armed = True
            stamp = self._clock(now)
            self._trim_pre_roll(stamp)
            self._begin(stamp, "manual" if self.config.start_mode == "manual" else "hardware")
            return self._status()

    def ingest(
        self,
        image: np.ndarray,
        timestamp: float,
        frame_id: int | str | None,
        detection: Any,
        triggered: bool = False,
    ) -> CaptureStatus:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        image = np.asarray(image)
        if image.ndim not in (2, 3) or image.size == 0 or image.dtype.hasobject:
            raise ValueError("image must be a nonempty numeric 2D or 3D array")
        measurement = _measurement(detection)
        motion = next(
            (
                measurement[name]
                for name in ("motion_fraction", "motion_ratio", "motion_score")
                if isinstance(measurement.get(name), (int, float))
            ),
            0.0,
        )
        moving = motion >= self.config.motion_threshold and motion > 0
        present = measurement["present"]
        with self._lock:
            self._ensure_open()
            if self._last_timestamp is not None and timestamp < self._last_timestamp:
                if self._active is not None:
                    self._finish("timestamp_reset", self._last_timestamp, aborted=True)
                self._clear_pre_roll()
            if self._last_shape is not None and image.shape != self._last_shape:
                if self._active is not None:
                    self._finish("image_shape_changed", timestamp, aborted=True)
                self._clear_pre_roll()
            self._last_timestamp = timestamp
            self._last_shape = image.shape
            self._trim_pre_roll(timestamp)
            if not self._armed:
                return self._status()
            if triggered and self.config.start_mode == "hardware":
                self._begin(timestamp, "hardware")
            elif self.config.start_mode == "motion" and moving and present and self._active is None:
                self._begin(timestamp, "motion")
            if self._active is not None:
                if not self._append_frame(image, timestamp, frame_id, measurement, active=True):
                    self._active.dropped_frames += 1
                    self._dropped_frames += 1
                    self._finish("memory_limit", timestamp, aborted=True)
                    return self._status()
                capture = self._active
                capture.last_present = present
                if moving:
                    capture.last_motion = timestamp
                    if present:
                        capture.motion_seen = True
                self._check_end(timestamp)
            elif self.config.pre_roll_seconds > 0:
                self._append_frame(image, timestamp, frame_id, measurement, active=False)
            return self._status()

    def poll(self, now: float | None = None) -> CaptureStatus:
        with self._lock:
            if self._active is not None:
                self._check_end(self._clock(now))
            return self._status()

    def wait_until_saved(self, timeout: float | None = None) -> bool:
        """Wait for queued writes, intended for shutdown/tests, never acquisition."""
        with self._saved:
            return self._saved.wait_for(lambda: self._pending == 0, timeout)

    def close(self, wait: bool = True, timeout: float | None = 10.0) -> bool:
        with self._lock:
            if not self._closed:
                self._armed = False
                if self._active is not None:
                    self._finish("closed", self._last_timestamp or self._active.start)
                self._clear_pre_roll()
                self._closed = True
                self._closing.set()
        if wait:
            self._thread.join(timeout)
        return not self._thread.is_alive()

    shutdown = close

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("capture manager is closed")

    def _clock(self, now: float | None) -> float:
        if now is not None:
            return float(now)
        return time.perf_counter()

    def _begin(self, stamp: float, source: str) -> bool:
        if self._active is not None:
            return False
        if self._pending >= self.config.writer_queue_depth:
            self._rejected_triggers += 1
            self._reason = "writer_busy"
            return False
        self._active = _Capture(
            stamp,
            source,
            datetime.now(timezone.utc).isoformat(),
            dict(vars(self.config)),
            json.loads(json.dumps(self.metadata, default=_scalar, allow_nan=False)),
            frames=list(self._pre_roll),
            pre_roll_truncated_frames=self._pre_roll_truncated,
        )
        self._pre_roll.clear()
        self._pre_roll_truncated = 0
        self._reason = None
        return True

    def _append_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        frame_id: Any,
        measurement: dict[str, Any],
        *,
        active: bool,
    ) -> bool:
        required = image.nbytes + self._FRAME_OVERHEAD
        limit = int(self.config.max_memory_mb * 1024 * 1024)
        if not active:
            while self._pre_roll and self._bytes + required > limit:
                self._release(self._pre_roll.popleft())
                self._pre_roll_truncated += 1
        if self._bytes + required > limit:
            if not active:
                self._pre_roll_truncated += 1
            return False
        snapshot = np.array(image, copy=True, order="C")
        snapshot.flags.writeable = False
        frame = _Frame(snapshot, timestamp, _scalar(frame_id), measurement, required)
        self._bytes += required
        self._frames += 1
        if active:
            assert self._active is not None
            self._active.frames.append(frame)
        else:
            self._pre_roll.append(frame)
        return True

    def _release(self, frame: _Frame) -> None:
        self._bytes -= frame.memory_bytes
        self._frames -= 1

    def _clear_pre_roll(self) -> None:
        while self._pre_roll:
            self._release(self._pre_roll.popleft())

    def _trim_pre_roll(self, now: float) -> None:
        cutoff = now - self.config.pre_roll_seconds
        while self._pre_roll and self._pre_roll[0].timestamp < cutoff:
            self._release(self._pre_roll.popleft())

    def _check_end(self, now: float) -> None:
        capture = self._active
        if capture is None:
            return
        if now - capture.start >= self.config.max_seconds:
            self._finish("max_duration", now)
        elif not capture.motion_seen and now - capture.start >= self.config.wait_seconds:
            self._finish("no_motion_timeout", now)
        elif (
            capture.motion_seen
            and not capture.last_present
            and capture.last_motion is not None
            and now - capture.last_motion >= self.config.quiet_seconds
        ):
            self._finish("part_left", now)

    def _finish(self, reason: str, stamp: float, *, aborted: bool = False) -> None:
        capture = self._active
        assert capture is not None
        capture.end = stamp
        capture.reason = reason
        capture.aborted = aborted
        self._active = None
        self._reason = reason
        self._pending += 1
        # _begin reserves a queue slot, including the currently executing writer.
        self._jobs.put_nowait(capture)

    def _writer(self) -> None:
        while True:
            try:
                capture = self._jobs.get(timeout=0.05)
            except queue.Empty:
                if self._closing.is_set():
                    return
                continue
            path: Path | None = None
            error = None
            try:
                path = self._persist(capture)
            except Exception as exc:  # Surface filesystem errors to the camera worker.
                error = f"{type(exc).__name__}: {exc}"
            with self._saved:
                if path is not None:
                    self.last_clip_path = path
                if error is not None:
                    self._error = error
                self.completed_clips.append(
                    ClipSaveResult(
                        path,
                        capture.reason,
                        len(capture.frames),
                        capture.dropped_frames,
                        error,
                    )
                )
                for frame in capture.frames:
                    self._release(frame)
                if capture.frames:
                    del frame
                capture.frames.clear()
                self._pending -= 1
                self._saved.notify_all()
            self._jobs.task_done()

    def _persist(self, capture: _Capture) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        utc_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        name = f"session_{utc_stamp}_{uuid.uuid4().hex[:8]}"
        temporary = self.output_dir / ("." + name + ".tmp")
        final = self.output_dir / name
        temporary.mkdir()
        frames_dir = temporary / "frames"
        frames_dir.mkdir()
        rows = []
        frame_id_gaps = 0
        previous_id = None
        for index, frame in enumerate(capture.frames):
            filename = f"frames/{index:06d}.npy"
            with (temporary / filename).open("wb") as stream:
                np.save(stream, frame.image, allow_pickle=False)
            rows.append(
                {
                    "file": filename,
                    "timestamp": frame.timestamp,
                    "frame_id": frame.frame_id,
                    "measurement": frame.measurement,
                }
            )
            if isinstance(frame.frame_id, int) and not isinstance(frame.frame_id, bool):
                if previous_id is not None and frame.frame_id > previous_id + 1:
                    frame_id_gaps += frame.frame_id - previous_id - 1
                previous_id = frame.frame_id
        manifest = {
            "schema_version": 1,
            "created_utc": capture.created_utc,
            "timebase": "monotonic_seconds",
            "trigger_timestamp": capture.start,
            "trigger_source": capture.trigger_source,
            "end_timestamp": capture.end,
            "stop_reason": capture.reason,
            "aborted": capture.aborted,
            "motion_seen": capture.motion_seen,
            "frame_count": len(rows),
            "missing_angle_frames": sum(
                row["measurement"].get("angle_deg") is None for row in rows
            ),
            "dropped_frames": capture.dropped_frames,
            "frame_id_gaps": frame_id_gaps,
            "pre_roll_truncated_frames": capture.pre_roll_truncated_frames,
            "capture_config": capture.capture_config,
            "metadata": capture.metadata,
            "frames": rows,
        }
        with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
        _write_csv(temporary / "measurements.csv", rows)
        os.replace(temporary, final)
        return final


_CSV_FIELDS = (
    "timestamp",
    "relative_seconds",
    "frame_id",
    "present",
    "angle_deg",
    "axis_deg",
    "confidence",
    "area",
    "motion_fraction",
    "touching_border",
    "processing_ms",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    start = rows[0]["timestamp"] if rows else 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            measurement = row.get("measurement", {})
            output = {name: measurement.get(name) for name in _CSV_FIELDS}
            output.update(
                timestamp=row["timestamp"],
                relative_seconds=row["timestamp"] - start,
                frame_id=row.get("frame_id"),
            )
            writer.writerow(output)


def load_clip(path: str | Path) -> Clip:
    """Load lossless originals as read-only memory maps; overlays stay editable."""
    source = Path(path)
    directory = source.parent if source.name == "manifest.json" else source
    directory = directory.resolve()
    with (directory / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported clip schema version")
    frames, timestamps, frame_ids, measurements = [], [], [], []
    for row in manifest.get("frames", []):
        frame_path = (directory / row["file"]).resolve()
        if not frame_path.is_relative_to(directory):
            raise ValueError("clip frame path escapes session directory")
        frames.append(np.load(frame_path, mmap_mode="r", allow_pickle=False))
        timestamps.append(float(row["timestamp"]))
        frame_ids.append(row.get("frame_id"))
        measurements.append(row.get("measurement", {}))
    if len(frames) != manifest.get("frame_count"):
        raise ValueError("clip frame count does not match manifest")
    return Clip(directory, frames, timestamps, frame_ids, measurements, manifest)
