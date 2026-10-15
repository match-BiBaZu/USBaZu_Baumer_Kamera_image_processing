"""Worker-owned external trials and a bounded, thread-safe telemetry mailbox."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import deque

from .local_api_runtime import ApiError
from .recording import CaptureConfig, _measurement

TERMINAL = {"completed", "failed", "aborted"}


class ApiCapture:
    def __init__(self, worker, capacity=8192):
        self.worker = worker
        self.lock = threading.RLock()
        self.trials = {}
        self.active_id = None
        self.samples = deque(maxlen=capacity)
        self.sequence = 0
        self.lease = None
        self.token = None
        self.started = 0.0
        self.seen_result = None
        self.cancel_reason = None

    def busy(self):
        with self.lock:
            return self.active_id is not None

    def create(self, request, lease, token):
        with self.lock:
            lease.require(token)
            if getattr(self.worker, "camera_settings_updating", False):
                raise ApiError(
                    409, "Kameraeinstellungen werden angewendet; Versuch später starten."
                )
            trial_id = request["trial_id"]
            if trial_id in self.trials:
                raise ApiError(409, "Versuchskennung wurde bereits verwendet.")
            if self.active_id:
                raise ApiError(409, "Ein Versuch läuft bereits.")
            record = {**request, "status": "pending", "error": None, "clip_path": None}
            self.trials[trial_id] = record
            if len(self.trials) > 256:
                self.trials.pop(next(iter(self.trials)))
            self.active_id = trial_id
            self.lease, self.token = lease, token
            self.cancel_reason = None
            if not self.worker.command("api_start", request):
                record.update(status="failed", error="Worker-Warteschlange voll")
                self.active_id = None
            return copy.deepcopy(record)

    def get(self, trial_id):
        with self.lock:
            if trial_id not in self.trials:
                raise ApiError(404, "Versuch nicht gefunden.")
            result = copy.deepcopy(self.trials[trial_id])
        if result["status"] in TERMINAL and result["clip_path"]:
            # Completed folders are published atomically by the existing writer.
            from pathlib import Path

            manifest = json.loads((Path(result["clip_path"]) / "manifest.json").read_text("utf-8"))
            result["manifest"] = manifest
            result["measurements"] = [
                {
                    "timestamp": row["timestamp"],
                    "frame_id": row["frame_id"],
                    "trial_id": trial_id,
                    "valid": row["measurement"].get("angle_deg") is not None,
                    **row["measurement"],
                }
                for row in manifest["frames"]
            ]
        return result

    def measurements(self, after, limit):
        with self.lock:
            first = self.samples[0]["sequence"] if self.samples else self.sequence + 1
            rows = [copy.deepcopy(s) for s in self.samples if s["sequence"] > after][:limit]
            return {
                "measurements": rows,
                "next_cursor": rows[-1]["sequence"] if rows else after,
                "overflow": after < first - 1,
                "oldest_cursor": first,
                "latest_cursor": self.sequence,
            }

    def start(self, capture, request):
        with self.lock:
            if not self.active_id or request["trial_id"] != self.active_id:
                return
            record = self.trials[self.active_id]
            try:
                self.lease.require(self.token)
                if self.cancel_reason:
                    raise ApiError(409, self.cancel_reason)
                if capture.status.state != "idle" or capture.status.writer_pending:
                    raise ApiError(
                        409, "Manuelle Aufnahme zuerst entschärfen und Speicherung abwarten."
                    )
                self.started = time.monotonic()
                self.seen_result = capture.completed_clips[-1] if capture.completed_clips else None
                if request.get("capture_config"):
                    capture.configure(CaptureConfig(**request["capture_config"]))
                    self.worker.capture_config = capture.config
                capture.arm(single_shot=True)
                record.update(status="armed", start_mode=capture.config.start_mode)
            except Exception as exc:
                record.update(status="failed", error=str(exc))
                self.active_id = None

    def metadata(self):
        with self.lock:
            if not self.active_id:
                return {}
            record = self.trials[self.active_id]
            return {key: copy.deepcopy(record[key]) for key in ("run_id", "trial_id", "context")}

    def cancel(self, reason="external_abort"):
        with self.lock:
            if self.active_id:
                self.cancel_reason = reason

    def trigger(self, capture):
        with self.lock:
            if not self.active_id or self.cancel_reason:
                return
            self.lease.require(self.token)
            if capture.config.start_mode != "manual" or capture.status.state != "armed":
                raise ApiError(409, "Manueller Trigger nur im scharfen manuellen Modus möglich.")
            capture.trigger()

    def tick(self, capture, frame=None, detection=None):
        """Called every acquisition iteration, including read timeouts."""
        with self.lock:
            if frame is not None:
                self.sequence += 1
                measurement = _measurement(detection)
                self.samples.append(
                    {
                        "sequence": self.sequence,
                        "timestamp": frame.timestamp,
                        "frame_id": frame.frame_id,
                        "trial_id": self.active_id,
                        "valid": measurement["angle_deg"] is not None,
                        "capture_state": capture.status.state,
                        **measurement,
                    }
                )
            if not self.active_id:
                return
            record = self.trials[self.active_id]
            if record["status"] == "pending":
                return
            try:
                self.lease.require(self.token)
            except ApiError:
                self.cancel_reason = "session_expired"
            if time.monotonic() - self.started > record["timeout_seconds"]:
                self.cancel_reason = "trial_timeout"
            if self.cancel_reason:
                capture.abort(self.cancel_reason)
            status = capture.status
            state = status.state
            if state in {"waiting", "moving", "saving"}:
                record["status"] = "saving" if state == "saving" else "recording"
            completed = capture.completed_clips[-1] if capture.completed_clips else None
            if (
                completed is not None
                and completed is not self.seen_result
                and not status.writer_pending
            ):
                record.update(
                    status="aborted"
                    if self.cancel_reason
                    else (
                        "failed"
                        if completed.error or completed.reason != "part_left"
                        else "completed"
                    ),
                    error=self.cancel_reason or completed.error,
                    stop_reason=completed.reason,
                    clip_path=str(completed.path) if completed.path else None,
                )
                self.active_id = None
            elif self.cancel_reason and state == "idle" and not status.writer_pending:
                record.update(status="aborted", error=self.cancel_reason)
                self.active_id = None

    def shutdown(self, reason):
        with self.lock:
            if self.active_id:
                self.trials[self.active_id].update(status="aborted", error=reason)
                self.active_id = None
