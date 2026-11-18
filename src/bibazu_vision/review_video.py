"""Compact, streamed review media. Measurements always retain original coordinates/times."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import cv2

RESERVE_BYTES = 1024**3


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Media path escapes capture directory")
    return path


def validate_target(directory, run_id, trial_id):
    original = Path(directory)
    if not original.is_absolute() or original.is_symlink():
        raise ValueError("Capture destination must be an absolute reserved directory")
    target = original.resolve(strict=True)
    marker = json.loads((target / "capture-request.json").read_text("utf-8"))
    if marker != {"format": "review_video_v1", "run_id": run_id, "trial_id": trial_id}:
        raise ValueError("Capture destination does not belong to this trial")
    if (target / "clip").exists() or (target / "clip.tmp").exists():
        raise ValueError("Capture destination already contains a recording")
    if shutil.disk_usage(target).free < RESERVE_BYTES:
        raise OSError("Weniger als 1 GiB freier Speicher am Aufnahmeziel")
    return target


def publish(temporary, final):
    """Bounded retries after every encoder/file handle is closed; never overwrite data."""
    temporary, final = Path(temporary), Path(final)
    for delay in (0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0):
        if delay:
            time.sleep(delay)
        if final.exists():
            raise FileExistsError("Recording destination already exists: " + str(final))
        try:
            temporary.rename(final)
            return final
        except PermissionError:
            pass
    # A complete manifest remains readable and can be published by a later GET.
    return temporary


def finalize_pending(path):
    path = Path(path)
    if path.name != "clip.tmp":
        return path
    final = path.with_name("clip")
    if not path.exists() and final.exists():
        return final
    if path.exists() and (path / "manifest.json").is_file():
        return publish(path, final)
    return path


def recover_pending(directory):
    """Rebuild only an explicitly reserved v2 recording from its flushed journal."""
    root = Path(directory).resolve()
    pending = root / "clip.tmp"
    try:
        marker = json.loads((root / "capture-request.json").read_text("utf-8"))
        # A crash may occur after publication but before the client saves trial.json.
        published = root / "clip"
        complete = pending if pending.is_dir() else published
        if (complete / "manifest.json").is_file():
            manifest = json.loads((complete / "manifest.json").read_text("utf-8"))
            metadata = manifest["metadata"]
            if manifest.get("schema_version") != 2 or marker != {
                "format": "review_video_v1",
                "run_id": metadata["run_id"],
                "trial_id": metadata["trial_id"],
            }:
                return None
            return finalize_pending(complete)
        if not pending.is_dir():
            return None
        manifest = json.loads((pending / "capture.json").read_text("utf-8"))
        metadata = manifest["metadata"]
        if marker != {
            "format": "review_video_v1",
            "run_id": metadata["run_id"],
            "trial_id": metadata["trial_id"],
        }:
            return None
        rows = []
        with (pending / "measurements.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    break  # An interrupted final write is never treated as a measurement.
                if row.get("video_frame_index") != len(rows):
                    break
                rows.append(row)
        manifest.update(
            schema_version=2,
            stop_reason="write_error",
            aborted=True,
            end_timestamp=rows[-1]["timestamp"] if rows else manifest["trigger_timestamp"],
            frames=rows,
            frame_count=len(rows),
            dropped_frames=0,
            media={
                "state": "failed",
                "file": "review.mp4",
                "format": "mp4v",
                "error": "Unterbrochene Aufnahme aus Messjournal wiederhergestellt",
            },
        )
        with (pending / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, allow_nan=False, indent=2)
        return finalize_pending(pending)
    except (OSError, ValueError, KeyError):
        return None


class VideoFrames:
    """Bounded lazy decoding without holding Windows file handles between UI calls."""

    def __init__(self, directory, manifest):
        self.directory, self.media = Path(directory), manifest["media"]
        self.count = (
            manifest["frame_count"]
            if self.media.get("state") in {"available", "deleting"}
            and (contained(self.directory, self.media["file"]).is_file())
            else 0
        )
        self.cache = {}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError(index)
        if index not in self.cache:
            capture = cv2.VideoCapture(str(contained(self.directory, self.media["file"])))
            try:
                if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise ValueError("Prüfvideo kann nicht geöffnet werden")
                self.cache.clear()
                for position in range(index, min(index + 8, self.count)):
                    ok, pixels = capture.read()
                    if not ok:
                        raise ValueError("Prüfvideo enthält fehlende Bilder")
                    pixels.flags.writeable = False
                    self.cache[position] = pixels
            finally:
                capture.release()
        return self.cache[index]

    def close(self):
        self.cache.clear()


def persist_video(manager, capture):
    """Drain the bounded capture queue while recording; never write NPY fallbacks."""
    root = capture.review_target
    temporary, final = root / "clip.tmp", root / "clip"
    temporary.mkdir(exist_ok=False)
    rows = []
    writer = None
    media = {"format": "mp4v", "state": "failed", "file": "review.mp4", "error": None}
    previous_id = None
    frame_gaps = 0
    journal = (temporary / "measurements.jsonl").open("w", encoding="utf-8")
    try:
        with (temporary / "capture.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 2,
                    "created_utc": capture.created_utc,
                    "trigger_timestamp": capture.start,
                    "trigger_source": capture.trigger_source,
                    "timebase": "monotonic_seconds",
                    "capture_config": capture.capture_config,
                    "metadata": capture.metadata,
                },
                stream,
                allow_nan=False,
            )
        while True:
            with manager._saved:
                manager._saved.wait_for(lambda: capture.frames or capture.end is not None)
                if not capture.frames:
                    break
                frame = capture.frames.popleft()
            try:
                row = {
                    "video_frame_index": len(rows),
                    "timestamp": frame.timestamp,
                    "frame_id": frame.frame_id,
                    "measurement": frame.measurement,
                }
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                rows.append(row)
                if isinstance(frame.frame_id, int):
                    if previous_id is not None and frame.frame_id > previous_id + 1:
                        frame_gaps += frame.frame_id - previous_id - 1
                    previous_id = frame.frame_id
                if media["error"] is None:
                    try:
                        if writer is None:
                            height, width = frame.image.shape[:2]
                            roi = capture.metadata.get("processing", {}).get("roi")
                            x, y, w, h = roi if roi else (0, 0, width, height)
                            x, y = max(0, int(x)), max(0, int(y))
                            w, h = min(int(w), width - x), min(int(h), height - y)
                            if min(w, h) <= 0:
                                raise ValueError("Invalid review ROI")
                            scale = min(1.0, 960 / max(w, h))
                            sw, sh = max(2, round(w * scale)), max(2, round(h * scale))
                            size = (sw + sw % 2, sh + sh % 2)
                            fps = capture.metadata.get("capture_fps", 0)
                            fps = float(fps) if isinstance(fps, (int, float)) else 0
                            if not math.isfinite(fps) or fps <= 0:
                                fps = 30.0
                            media.update(
                                crop_xywh=[x, y, w, h],
                                image_size=list(size),
                                content_size=[sw, sh],
                                scale_xy=[sw / w, sh / h],
                                fps=fps,
                            )
                            writer = cv2.VideoWriter(
                                str(temporary / "review.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps,
                                size,
                            )
                            if not writer.isOpened():
                                raise RuntimeError("MP4-Encoder nicht verfügbar")
                        x, y, w, h = media["crop_xywh"]
                        image = cv2.resize(
                            frame.image[y : y + h, x : x + w],
                            tuple(media["content_size"]),
                            interpolation=cv2.INTER_AREA,
                        )
                        if image.ndim == 2 or image.shape[2] == 1:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        elif image.shape[2] == 4:
                            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                        sw, sh = media["content_size"]
                        width, height = media["image_size"]
                        image = cv2.copyMakeBorder(
                            image, 0, height - sh, 0, width - sw, cv2.BORDER_REPLICATE
                        )
                        writer.write(image)
                    except Exception as exc:
                        media.update(state="failed", error=str(exc))
                capture.written_frames += 1
            finally:
                with manager._saved:
                    manager._release(frame)
                del frame
    finally:
        journal.close()
        if writer is not None:
            writer.release()
    if media["error"] is None and rows:
        video = cv2.VideoCapture(str(temporary / "review.mp4"))
        try:
            if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
                media["error"] = "Prüfvideo ist unvollständig"
            else:
                media["state"] = "available"
        finally:
            video.release()
    else:
        media["error"] = media["error"] or "Keine Videobilder"
    manifest = {
        "schema_version": 2,
        "created_utc": capture.created_utc,
        "timebase": "monotonic_seconds",
        "trigger_timestamp": capture.start,
        "trigger_source": capture.trigger_source,
        "end_timestamp": capture.end,
        "stop_reason": capture.reason,
        "aborted": capture.aborted,
        "motion_seen": capture.motion_seen,
        "frame_count": len(rows),
        "missing_angle_frames": sum(r["measurement"].get("angle_deg") is None for r in rows),
        "dropped_frames": capture.dropped_frames,
        "frame_id_gaps": frame_gaps,
        "pre_roll_truncated_frames": capture.pre_roll_truncated_frames,
        "capture_config": capture.capture_config,
        "metadata": capture.metadata,
        "frames": rows,
        "media": media,
    }
    # The small journal is retained for interruption recovery, even if encoding fails.
    with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, allow_nan=False, indent=2)
    return publish(temporary, final)
