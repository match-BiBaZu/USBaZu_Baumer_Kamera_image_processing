"""Acquisition and analysis off the UI thread; latest-only preview mailbox."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from .api_capture import ApiCapture
from .camera import BaumerCamera, CameraConfig, CameraSettingsError
from .demo import DemoCamera
from .local_api_runtime import ApiError
from .processing import FrameProcessor, ProcessingConfig
from .recording import CaptureConfig, CaptureManager


class PipelineWorker(QThread):
    message = pyqtSignal(str)
    failed = pyqtSignal(str)
    recording_failed = pyqtSignal(str)
    camera_settings_finished = pyqtSignal(bool, dict, str)

    def __init__(
        self,
        camera_config: CameraConfig,
        processing_config: ProcessingConfig,
        capture_config: CaptureConfig,
        data_dir: Path,
        demo=False,
        output_dir=None,
    ):
        super().__init__()
        self.camera_config = camera_config
        self.processing_config = processing_config
        self.capture_config = capture_config
        self.data_dir = Path(data_dir)
        self.demo = demo
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "recordings"
        self.commands = queue.Queue(maxsize=100)
        self.lock = threading.Lock()
        self.latest = None
        self.stopping = threading.Event()
        self.camera_idle = threading.Event()
        self.camera_settings_updating = False
        self.api_capture = ApiCapture(self)

    def camera_settings_available(self):
        # Separate from the frame mailbox: hardware-triggered cameras may not
        # deliver their first frame until long after connecting.
        return (
            self.camera_idle.is_set() and not self.demo
            and not self.stopping.is_set() and not self.api_capture.busy()
        )

    def apply_camera_settings(self, camera, capture, value):
        # Reserve against API trials without holding their telemetry lock across
        # slow GenTL operations (the GUI also reads that lock).
        with self.api_capture.lock:
            if (
                self.demo or self.stopping.is_set() or self.api_capture.busy()
                or capture.status.state != "idle"
            ):
                self.camera_settings_finished.emit(
                    False, {}, "Kameraeinstellungen nur bei verbundener Kamera und "
                    "beendeter Aufzeichnung ohne AutoTune-Versuch anwenden."
                )
                return False
            self.camera_settings_updating = True
            self.camera_idle.clear()
        warning_count = len(camera.diagnostics.get("warnings", []))
        try:
            applied = camera.apply_settings(**value)
        except Exception as exc:
            self.camera_settings_finished.emit(False, {}, str(exc))
            if not isinstance(exc, CameraSettingsError):
                raise
            return False
        finally:
            with self.api_capture.lock:
                self.camera_settings_updating = False
        for warning in camera.diagnostics.get("warnings", [])[warning_count:]:
            self.message.emit(warning)
        self.camera_settings_finished.emit(True, applied, "")
        return True

    def command(self, name, value=None):
        try:
            self.commands.put_nowait((name, value))
            return True
        except queue.Full:
            self.message.emit("Zu viele Einstellungen: kurz warten und erneut anwenden.")
            return False

    def snapshot(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.stopping.set()

    def run(self):
        camera = None
        capture = None
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            processor = FrameProcessor(self.processing_config)
            background_path = self.data_dir / "background.png"
            if self.demo:
                processor.set_background(DemoCamera.background())
            elif background_path.exists():
                image = cv2.imread(str(background_path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    processor.set_background(image)
            capture = CaptureManager(self.capture_config, self.output_dir)
            camera = (
                DemoCamera(self.camera_config) if self.demo else BaumerCamera(self.camera_config)
            )
            camera.open()
            if capture.status.state == "idle":
                self.camera_idle.set()
            devices = camera.diagnostics.get("devices", [])
            model = (
                devices[0].get("model", "Baumer")
                if devices
                else camera.diagnostics.get("model", "Baumer")
            )
            monitor = camera.diagnostics.get("hardware_trigger_monitor", "Demo")
            self.message.emit(f"Verbunden: {model} · Triggersignal: {monitor}")
            for warning in camera.diagnostics.get("warnings", []):
                self.message.emit(warning)
            frame = None
            detection = None
            frames = 0
            rate_started = time.perf_counter()
            fps = 0.0
            frame_gaps = 0
            previous_id = None
            previous_recording_error = None

            def report_recording_error(status):
                nonlocal previous_recording_error
                if status.error and status.error != previous_recording_error:
                    self.recording_failed.emit(status.error)
                    self.message.emit(f"Aufzeichnung fehlgeschlagen: {status.error}")
                previous_recording_error = status.error

            while not self.stopping.is_set():
                while True:
                    try:
                        name, value = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if name == "camera_settings":
                        if self.apply_camera_settings(camera, capture, value):
                            previous_id = None
                            frames = 0
                            fps = 0.0
                            rate_started = time.perf_counter()
                        continue
                    if name == "api_start":
                        self.api_capture.start(capture, value)
                        if self.api_capture.busy():
                            capture.metadata = {
                                "processing": asdict(processor.config),
                                "camera": dict(camera.diagnostics), "demo": self.demo,
                                **self.api_capture.metadata(),
                            }
                            if self.camera_config.mode == "acquisition_trigger" and not self.demo:
                                camera.rearm_acquisition()
                        continue
                    if name == "api_trigger":
                        try:
                            self.api_capture.trigger(capture)
                        except ApiError as exc:
                            self.message.emit(str(exc))
                        continue
                    if self.api_capture.busy():
                        if name == "disarm":
                            self.api_capture.cancel("manual_stop")
                        self.message.emit("AutoTune-Versuch aktiv; Einstellungen sind gesperrt.")
                        continue
                    if name in {
                        "config",
                        "background",
                        "clear_background",
                        "zero",
                    } and capture.status.state in {"waiting", "moving"}:
                        self.message.emit("Messreferenz erst nach dem laufenden Durchlauf ändern.")
                        continue
                    if name == "config":
                        for key, setting in value.items():
                            setattr(processor.config, key, setting)
                    elif name == "background" and frame is not None:
                        processor.set_background(frame.image)
                        if not self.demo:
                            if not cv2.imwrite(str(background_path), frame.image):
                                raise OSError("Hintergrund konnte nicht gespeichert werden.")
                        self.message.emit(
                            "Hintergrund aufgenommen. Nur bei leerer Rutsche verwenden."
                        )
                    elif name == "clear_background":
                        processor.clear_background()
                        if background_path.exists() and not self.demo:
                            background_path.unlink()
                        self.message.emit(
                            "Hintergrund entfernt; absolute Helligkeitsschwelle aktiv."
                        )
                    elif name == "zero" and detection is not None:
                        if detection.axis_deg is not None and detection.angle_deg is not None:
                            processor.config.zero_offset_deg = (
                                detection.axis_deg - processor.config.chute_angle_deg
                            )
                            self.message.emit(
                                "Liegendes Bauteil als 0° gesetzt; gilt für diese Geometrie."
                            )
                        else:
                            self.message.emit(
                                "Keine eindeutige Bauteilachse: 0° konnte nicht gesetzt werden."
                            )
                    elif name == "arm":
                        capture.metadata = {
                            "processing": asdict(processor.config),
                            "camera": dict(camera.diagnostics),
                            "demo": self.demo,
                        }
                        was_active = capture.status.state in {"waiting", "moving"}
                        capture.arm()
                        if (
                            self.camera_config.mode == "acquisition_trigger"
                            and not self.demo
                            and not was_active
                        ):
                            camera.rearm_acquisition()
                    elif name == "disarm":
                        was_active = capture.status.state in {"waiting", "moving"}
                        capture.disarm()
                        if (
                            self.camera_config.mode == "acquisition_trigger"
                            and not self.demo
                            and was_active
                        ):
                            camera.rearm_acquisition()
                    elif name == "trigger":
                        capture.metadata = {
                            "processing": asdict(processor.config),
                            "camera": dict(camera.diagnostics),
                            "demo": self.demo,
                        }
                        capture.trigger()
                    elif name == "capture_config":
                        try:
                            capture.configure(CaptureConfig(**value))
                            self.capture_config = capture.config
                        except (ValueError, RuntimeError) as exc:
                            self.message.emit(f"Aufzeichnungseinstellungen: {exc}")
                if capture.status.state == "idle":
                    self.camera_idle.set()
                else:
                    self.camera_idle.clear()
                current = camera.read(timeout=0.2)
                if current is None:
                    previous_state = capture.status.state
                    status = capture.poll(time.perf_counter())
                    report_recording_error(status)
                    self.api_capture.tick(capture)
                    if (
                        self.camera_config.mode == "acquisition_trigger"
                        and not self.demo
                        and previous_state in {"waiting", "moving"}
                        and status.state not in {"waiting", "moving"}
                    ):
                        camera.rearm_acquisition()
                    with self.lock:
                        if self.latest is not None:
                            self.latest = {
                                **self.latest,
                                "status": status,
                                "last_clip": capture.last_clip_path,
                                "stale_seconds": time.perf_counter() - frame.timestamp,
                            }
                    continue
                frame = current
                if previous_id is not None and frame.frame_id > previous_id + 1:
                    frame_gaps += frame.frame_id - previous_id - 1
                previous_id = frame.frame_id
                detection = processor.process(frame.image)
                capture.metadata = {
                    "processing": asdict(processor.config),
                    "camera": dict(camera.diagnostics),
                    "demo": self.demo,
                    **self.api_capture.metadata(),
                }
                previous_state = capture.status.state
                status = capture.ingest(
                    frame.image,
                    frame.timestamp,
                    frame.frame_id,
                    detection,
                    triggered=frame.triggered,
                )
                report_recording_error(status)
                self.api_capture.tick(capture, frame, detection)
                if (
                    self.camera_config.mode == "acquisition_trigger"
                    and not self.demo
                    and (
                        previous_state in {"waiting", "moving"}
                        or (frame.triggered and capture.config.start_mode == "hardware")
                    )
                    and status.state not in {"waiting", "moving"}
                ):
                    camera.rearm_acquisition()
                frames += 1
                now = time.perf_counter()
                if now - rate_started >= 0.5:
                    fps = frames / (now - rate_started)
                    frames = 0
                    rate_started = now
                with self.lock:
                    self.latest = {
                        "image": frame.image,
                        "detection": detection,
                        "config": asdict(processor.config),
                        "has_background": processor.has_background,
                        "status": status,
                        "fps": fps,
                        "frame_id": frame.frame_id,
                        "frame_gaps": frame_gaps,
                        "last_clip": capture.last_clip_path,
                        "stale_seconds": 0.0,
                    }
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.camera_idle.clear()
            if camera is not None:
                try:
                    camera.close()
                except Exception as exc:
                    self.message.emit(f"Kamera schließen: {exc}")
            if capture is not None:
                try:
                    if self.api_capture.busy():
                        self.api_capture.cancel("camera_disconnected")
                        capture.abort("camera_disconnected")
                    # Keep the worker alive until its writer has finished; the UI
                    # close handler can then safely wait or let the user retry.
                    capture.close(timeout=None)
                    self.api_capture.tick(capture)
                    if capture.status.error:
                        self.recording_failed.emit(capture.status.error)
                except Exception as exc:
                    self.message.emit(f"Aufzeichnung abschließen: {exc}")
            self.api_capture.shutdown("camera_disconnected")
