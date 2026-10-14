"""Local HTTP facade for the connected GUI pipeline."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import Header, Query
from pydantic import BaseModel, ConfigDict, Field
from PyQt6.QtCore import QObject

from .local_api_runtime import (
    ApiError,
    LocalApiServer,
    QtExecutor,
    SessionLease,
    StrictModel,
    create_app,
)


class TrialRequest(StrictModel):
    run_id: Annotated[str, Field(min_length=1, max_length=128)]
    trial_id: Annotated[str, Field(min_length=1, max_length=128)]
    context: dict = Field(default_factory=dict)
    timeout_seconds: Annotated[float, Field(ge=1, le=3600)] = 60.0


class Measurement(BaseModel):
    model_config = ConfigDict(extra="allow")
    timestamp: float
    frame_id: int | str | None
    angle_deg: float | None = None
    confidence: float | None = None
    valid: bool
    trial_id: str | None = None
    sequence: int | None = None


class MeasurementBatch(BaseModel):
    measurements: list[Measurement]
    next_cursor: int
    oldest_cursor: int
    latest_cursor: int
    overflow: bool


class TrialResult(BaseModel):
    run_id: str
    trial_id: str
    context: dict
    timeout_seconds: float
    capture_config: dict
    status: Literal["pending", "armed", "recording", "saving", "completed", "failed", "aborted"]
    start_mode: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    clip_path: str | None = None
    manifest: dict | None = None
    measurements: list[Measurement] | None = None


class VisionApi(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.executor = QtExecutor(self)
        self.lease = SessionLease()
        self.server = None
        self.trial_sources = {}
        self.app = create_app(self, "BiBaZu Camera API")
        self.routes()

    def invoke(self, callback):
        return self.executor.call(callback)

    def pipeline(self):
        worker = self.window.worker
        if worker is None or not worker.isRunning() or worker.snapshot() is None:
            raise ApiError(503, "Zuerst Kamera oder Demo verbinden.")
        return worker

    def health(self):
        worker = self.window.worker
        return {"api_version": 1, "connected": worker is not None and worker.isRunning()}

    def heartbeat(self, token):
        self.pipeline()
        return self.lease.heartbeat(token)

    def capabilities(self):
        return {
            "api_version": 1,
            "single_shot": True,
            "measurement_capacity": 8192,
            "start_modes": ["motion", "hardware", "manual"],
            "angle_semantics": "image_plane_axis_modulo_180_degrees",
        }

    def state(self):
        worker = self.pipeline()
        snapshot = worker.snapshot()
        return {
            **self.health(),
            "session_active": self.lease.active(),
            "capture": asdict(snapshot["status"]),
            "processing": snapshot["config"],
            "capture_config": asdict(self.window.capture_config())
            if hasattr(self.window, "capture_config")
            else asdict(worker.capture_config),
            "has_background": snapshot["has_background"],
            "stale_seconds": snapshot["stale_seconds"],
            "demo": worker.demo,
            "trial_active": worker.api_capture.busy(),
        }

    def acquire(self, client):
        self.expire()
        worker = self.pipeline()
        if worker.api_capture.busy() or worker.snapshot()["status"].state != "idle":
            raise ApiError(409, "Kameraaufnahme zuerst beenden und entschärfen.")
        return self.lease.acquire()

    def expire(self):
        worker = self.window.worker
        if worker and not self.lease.active():
            worker.api_capture.cancel("session_expired")

    def routes(self):
        app, invoke = self.app, self.app.state.invoke

        @app.post("/api/v1/trials", status_code=202, response_model=TrialResult)
        def trial(body: TrialRequest, x_session_id: str = Header()):
            def submit():
                worker = self.pipeline()
                request = body.model_dump()
                request["capture_config"] = (
                    asdict(self.window.capture_config())
                    if hasattr(self.window, "capture_config")
                    else asdict(worker.capture_config)
                )
                result = worker.api_capture.create(request, self.lease, x_session_id)
                self.trial_sources[body.trial_id] = worker.api_capture
                if len(self.trial_sources) > 256:
                    self.trial_sources.pop(next(iter(self.trial_sources)))
                return result

            return invoke(submit)

        @app.get("/api/v1/trials/{trial_id}", response_model=TrialResult)
        def get_trial(trial_id: str):
            # Read manifests on the HTTP thread, never the acquisition or Qt thread.
            def source():
                if trial_id not in self.trial_sources:
                    raise ApiError(404, "Versuch nicht gefunden.")
                return self.trial_sources[trial_id]

            capture = invoke(source)
            try:
                return capture.get(trial_id)
            except ApiError as exc:
                from fastapi import HTTPException

                raise HTTPException(exc.status, str(exc)) from exc

        @app.post("/api/v1/trials/{trial_id}/trigger", status_code=202)
        def trigger(trial_id: str, x_session_id: str = Header()):
            def submit():
                self.lease.require(x_session_id)
                worker = self.pipeline()
                if worker.api_capture.active_id != trial_id:
                    raise ApiError(409, "Versuch ist nicht aktiv.")
                trial = worker.api_capture.trials[trial_id]
                if trial["capture_config"]["start_mode"] != "manual":
                    raise ApiError(409, "Aufnahme ist nicht im manuellen Modus.")
                if not worker.command("api_trigger"):
                    raise ApiError(409, "Worker-Warteschlange voll.")
                return {"accepted": True}

            return invoke(submit)

        @app.post("/api/v1/trials/{trial_id}/abort", status_code=202)
        def abort(trial_id: str, x_session_id: str = Header()):
            def submit():
                self.lease.require(x_session_id)
                worker = self.pipeline()
                if worker.api_capture.active_id != trial_id:
                    raise ApiError(409, "Versuch ist nicht aktiv.")
                worker.api_capture.cancel()
                return {"accepted": True}

            return invoke(submit)

        @app.get("/api/v1/measurements", response_model=MeasurementBatch)
        def measurements(after: int = Query(0, ge=0), limit: int = Query(2048, ge=1, le=8192)):
            return invoke(lambda: self.pipeline().api_capture.measurements(after, limit))

    def start(self):
        if os.environ.get("BIBAZU_API_DISABLED") == "1":
            return
        try:
            self.server = LocalApiServer(
                self.app, int(os.environ.get("BIBAZU_VISION_API_PORT", 8766))
            )
            self.server.start()
            self.window.note(f"Kamera-API: http://127.0.0.1:{self.server.port}/docs")
        except (OSError, ValueError) as exc:
            self.window.note(f"Kamera-API konnte nicht starten: {exc}")

    def stop(self):
        self.lease.invalidate()
        self.expire()
        if self.server:
            self.server.stop()
