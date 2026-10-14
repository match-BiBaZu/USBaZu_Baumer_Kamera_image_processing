"""Small local API host. Keep in sync with bibazu_vision/local_api_runtime.py."""

from __future__ import annotations

import asyncio
import secrets
import socket
import threading
import time
from concurrent.futures import Future, TimeoutError
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict
from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class SessionRequest(StrictModel):
    client: str = "external"


class Health(BaseModel):
    api_version: int = 1
    connected: bool


class SessionResponse(BaseModel):
    session_id: str
    expires_in: float
    snapshot_operation_id: str | None = None


class SessionLease:
    """Thread-safe, expiring ownership; queued commands recheck at execution."""

    def __init__(self, ttl=15.0, clock=time.monotonic):
        self.ttl, self.clock = ttl, clock
        self.lock = threading.RLock()
        self.token = None
        self.deadline = 0.0

    def active(self):
        with self.lock:
            return self.token is not None and self.clock() < self.deadline

    def acquire(self):
        with self.lock:
            if self.active():
                raise ApiError(409, "Eine externe Sitzung ist bereits aktiv.")
            self.token = secrets.token_urlsafe(32)
            self.deadline = self.clock() + self.ttl
            return {"session_id": self.token, "expires_in": self.ttl}

    def require(self, token):
        with self.lock:
            if not self.active() or not secrets.compare_digest(token or "", self.token):
                raise ApiError(409, "Sitzung fehlt oder ist abgelaufen.")

    def heartbeat(self, token):
        with self.lock:
            self.require(token)
            self.deadline = self.clock() + self.ttl
            return {"expires_in": self.ttl}

    def release(self, token):
        with self.lock:
            self.require(token)
            self.token = None

    def invalidate(self):
        with self.lock:
            self.token = None


class QtExecutor(QObject):
    requested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.requested.connect(self.execute, Qt.ConnectionType.QueuedConnection)

    @pyqtSlot(object)
    def execute(self, request):
        future, callback = request
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(callback())
        except Exception as exc:
            future.set_exception(exc)

    def call(self, callback, timeout=5):
        future = Future()
        self.requested.emit((future, callback))
        try:
            return future.result(timeout)
        except TimeoutError as exc:
            future.cancel()
            raise ApiError(504, "Die Anwendung antwortet nicht rechtzeitig.") from exc


def create_app(adapter, title, state_model=None):
    @asynccontextmanager
    async def lifespan(app):
        async def watch():
            while True:
                await asyncio.sleep(0.5)
                try:
                    await asyncio.to_thread(adapter.invoke, adapter.expire)
                except ApiError:
                    pass

        task = asyncio.create_task(watch())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title=title, version="1.0.0", lifespan=lifespan)

    def invoke(callback):
        try:
            return adapter.invoke(callback)
        except ApiError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    app.state.invoke = invoke

    @app.middleware("http")
    async def local_clients_only(request, call_next):
        # Native clients have no Origin. Do not permit browser-originated writes
        # or DNS-rebinding against a service capable of changing machine settings.
        host = request.headers.get("host", "").split(":")[0]
        if host not in {"127.0.0.1", "localhost", "testserver"} or (
            request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("origin")
        ):
            from starlette.responses import JSONResponse

            return JSONResponse({"detail": "Nur lokale native Clients erlaubt."}, status_code=403)
        return await call_next(request)

    @app.get("/api/v1/health", response_model=Health)
    def health():
        return invoke(adapter.health)

    @app.get("/api/v1/capabilities")
    def capabilities():
        return invoke(adapter.capabilities)

    @app.get("/api/v1/state", response_model=state_model)
    def state():
        return invoke(adapter.state)

    @app.post("/api/v1/sessions", status_code=201, response_model=SessionResponse)
    def acquire(body: SessionRequest):
        return invoke(lambda: adapter.acquire(body.client))

    @app.post("/api/v1/sessions/heartbeat")
    def heartbeat(x_session_id: str = Header()):
        return invoke(lambda: adapter.heartbeat(x_session_id))

    @app.delete("/api/v1/sessions/current")
    def release(x_session_id: str = Header()):
        def do_release():
            adapter.lease.release(x_session_id)
            adapter.expire()
            return {"released": True}

        return invoke(do_release)

    return app


class LocalApiServer:
    """One ASGI server thread, prebound loopback socket, no reload or processes."""

    def __init__(self, app, port):
        self.app, self.port = app, port
        self.server = None
        self.thread = None
        self.sock = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            self.sock.bind(("127.0.0.1", self.port))
            self.sock.listen(128)
            self.port = self.sock.getsockname()[1]
        except OSError:
            self.sock.close()
            raise
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                access_log=False,
                lifespan="on",
                timeout_graceful_shutdown=1,
            )
        )
        self.thread = threading.Thread(
            target=self.server.run,
            kwargs={"sockets": [self.sock]},
            daemon=True,
            name=f"local-api-{self.port}",
        )
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(2)
        if self.sock:
            self.sock.close()
