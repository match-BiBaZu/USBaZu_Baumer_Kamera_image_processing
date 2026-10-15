"""Baumer GenTL capture independent of Qt; all returned pixels own their memory."""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class CameraConfig:
    cti_path: str | None = None
    serial: str | None = None
    exposure_us: float | None = None
    fps: float | None = None
    mode: str = "live"
    trigger_line: str = "Line0"
    trigger_activation: str = "RisingEdge"
    input_monitor: str = "auto"


@dataclass(slots=True)
class CameraFrame:
    image: np.ndarray
    timestamp: float
    frame_id: int
    triggered: bool = False


class CameraSettingsError(RuntimeError):
    """Settings were rejected, but acquisition is still usable."""


def find_cti_paths() -> list[str]:
    """Prefer Baumer USB producers; do not mistake a GigE producer for USB."""
    paths: list[Path] = []
    for name in ("GENICAM_GENTL64_PATH", "GENICAM_GENTL32_PATH"):
        for directory in os.environ.get(name, "").split(os.pathsep):
            if directory:
                paths.extend(Path(directory).glob("*.cti"))
    for root in (
        Path(os.environ.get("ProgramFiles", "C:/Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")),
    ):
        for directory in root.glob("Baumer*"):
            paths.extend(directory.rglob("*.cti"))
    unique = {str(path.resolve()) for path in paths if path.is_file()}
    return sorted(
        unique, key=lambda path: ("usb" not in path.lower(), "baumer" not in path.lower(), path)
    )


def _node(nm: Any, name: str) -> Any:
    try:
        return getattr(nm, name)
    except Exception:
        return None


def _value(nm: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(nm, name).value
    except Exception:
        return default


def _entries(nm: Any, name: str) -> list[str]:
    try:
        from genicam.genapi import is_available

        return [entry.symbolic for entry in getattr(nm, name).entries if is_available(entry)]
    except Exception:
        return []


def _writable(node: Any) -> bool:
    if node is None:
        return False
    try:
        from genicam.genapi import is_writable

        return bool(is_writable(node))
    except Exception:
        return False


def convert_component(component: Any) -> np.ndarray:
    """Copy before the GenTL buffer is requeued, keeping Mono8 inexpensive."""
    width, height = int(component.width), int(component.height)
    fmt = str(component.data_format)
    lower = fmt.lower()
    raw = np.asarray(component.data)
    if "packed" in lower or lower.endswith("p"):
        raise ValueError(
            f"Gepacktes Pixelformat {fmt} wird nicht unterstützt; Mono8/Bayer8 auswählen."
        )
    if lower.startswith(("mono", "bayer")):
        if raw.size != width * height:
            raise ValueError(f"Unerwartete Puffergröße für {fmt}: {raw.size}.")
        image = raw.reshape(height, width)
        if image.dtype != np.uint8:
            bits = re.search(r"(10|12|14|16)", lower)
            bit_count = int(bits.group(1)) if bits else image.dtype.itemsize * 8
            image = np.clip(
                image.astype(np.float32) * (255.0 / ((1 << bit_count) - 1)), 0, 255
            ).astype(np.uint8)
        else:
            image = np.array(image, order="C", copy=True)
        if lower.startswith("mono"):
            return image
        codes = {
            "bayerrg": cv2.COLOR_BayerRGGB2BGR,
            "bayerbg": cv2.COLOR_BayerBGGR2BGR,
            "bayergr": cv2.COLOR_BayerGRBG2BGR,
            "bayergb": cv2.COLOR_BayerGBRG2BGR,
        }
        for prefix, code in codes.items():
            if lower.startswith(prefix):
                return cv2.cvtColor(image, code)
    if lower.startswith(("rgb8", "bgr8")):
        image = np.array(raw.reshape(height, width, 3), dtype=np.uint8, order="C", copy=True)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if lower.startswith("rgb") else image
    raise ValueError(f"Nicht unterstütztes Pixelformat: {fmt}; Mono8 oder Bayer8 auswählen.")


class BaumerCamera:
    """Use on one capture thread. Configuration is restored when closed.

    ``live`` streams continuously and uses an unused hardware counter when
    available; its fallback samples LineStatus and may miss short pulses.
    ``frame_trigger`` delivers
    one externally triggered frame per pulse. ``acquisition_trigger`` requires a
    real AcquisitionStart selector and begins a continuous stream on one pulse.
    Call rearm_acquisition() between passes to await the next acquisition pulse.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._harvester: Any = None
        self._acquirer: Any = None
        self._nm: Any = None
        self._restore: list[tuple[str, Any]] = []
        self._triggers: list[dict[str, Any]] = []
        self._sequence = 0
        self._first_acquisition_frame = True
        self._last_line_state: bool | None = None
        self._line_node: Any = None
        self._counter_node: Any = None
        self._last_counter_value: int | None = None
        self._counter_restore: tuple[str, dict[str, Any]] | None = None
        self._opened = False
        self.diagnostics: dict[str, Any] = {"warnings": []}

    @property
    def is_open(self) -> bool:
        return self._opened

    def _set(self, name: str, value: Any, *, required: bool = True) -> bool:
        node = _node(self._nm, name)
        if not _writable(node):
            if required:
                raise RuntimeError(f"Kameramerkmal {name} ist nicht schreibbar.")
            return False
        old_value = _value(self._nm, name)
        if old_value is not None and not any(saved_name == name for saved_name, _ in self._restore):
            self._restore.append((name, old_value))
        node.value = value
        return True

    def _inspect(self) -> None:
        nm = self._nm
        names = (
            "DeviceModelName",
            "DeviceSerialNumber",
            "Width",
            "Height",
            "PixelFormat",
            "ExposureTime",
            "ExposureTimeAbs",
            "ExposureAuto",
            "GainAuto",
            "Gain",
            "BalanceWhiteAuto",
            "ColorTransformationAuto",
            "AcquisitionFrameRate",
            "AcquisitionFrameRateEnable",
            "ResultingFrameRate",
            "AcquisitionMode",
            "TriggerSelector",
            "LineSelector",
            "EventSelector",
            "ChunkSelector",
            "CounterSelector",
        )
        self.diagnostics["nodes"] = {
            name: {"value": _value(nm, name), "entries": _entries(nm, name)} for name in names
        }
        selector_node = _node(nm, "TriggerSelector")
        old_selector = _value(nm, "TriggerSelector")
        if old_selector is not None:
            self._restore.append(("TriggerSelector", old_selector))
        for selector in _entries(nm, "TriggerSelector"):
            if not _writable(selector_node):
                break
            selector_node.value = selector
            record = {
                "selector": selector,
                "mode": _value(nm, "TriggerMode"),
                "source": _value(nm, "TriggerSource"),
                "activation": _value(nm, "TriggerActivation"),
                "sources": _entries(nm, "TriggerSource"),
                "activations": _entries(nm, "TriggerActivation"),
            }
            self._triggers.append(record)
        if old_selector is not None and _writable(selector_node):
            selector_node.value = old_selector
        self.diagnostics["triggers"] = list(self._triggers)

    def _configure_settings(self, exposure_us, fps, rollback=None) -> None:
        def set_value(name, value, *, required=True):
            if rollback is not None and _writable(_node(self._nm, name)):
                old = _value(self._nm, name)
                if old is None:
                    raise RuntimeError(f"Kameramerkmal {name} ist nicht lesbar.")
                rollback.append((name, old))
            return self._set(name, value, required=required)

        if exposure_us is not None:
            set_value("ExposureAuto", "Off", required=False)
            # Preserve gain/color coefficients while fixing image brightness.
            for name in ("GainAuto", "BalanceWhiteAuto", "ColorTransformationAuto"):
                current = _value(self._nm, name)
                if current is None:
                    continue
                if not set_value(name, "Off", required=False) and current != "Off":
                    warning = (
                        f"{name} ist weiterhin {current}; "
                        "Helligkeit/Farbe können die Konturerkennung verändern."
                    )
                    if warning not in self.diagnostics["warnings"]:
                        self.diagnostics["warnings"].append(warning)
            name = (
                "ExposureTime" if _node(self._nm, "ExposureTime") is not None
                else "ExposureTimeAbs"
            )
            set_value(name, float(exposure_us))
        if fps is not None:
            set_value("AcquisitionFrameRateEnable", True, required=False)
            set_value("AcquisitionFrameRate", float(fps))

    def _refresh_actual(self) -> None:
        self.diagnostics["actual"] = {
            name: _value(self._nm, name)
            for name in (
                "Width", "Height", "PixelFormat", "ExposureTime", "ExposureTimeAbs",
                "ExposureAuto", "GainAuto", "Gain", "BalanceWhiteAuto",
                "ColorTransformationAuto", "AcquisitionFrameRateEnable",
                "AcquisitionFrameRate", "ResultingFrameRate",
            )
        }

    def _restart_after_settings(self) -> None:
        self._first_acquisition_frame = True
        # Ignore edges occurring during the preview pause.
        if self._counter_node is not None:
            self._last_counter_value = int(self._counter_node.value)
        if self._line_node is not None:
            self._last_line_state = bool(self._line_node.value)
        self._acquirer.start()

    def apply_settings(self, exposure_us: float, fps: float) -> dict[str, float]:
        """Pause acquisition on this thread, update, and resume the same device.

        A CameraSettingsError means the previous values were recovered. Other
        errors require the caller to close the camera through its failure path.
        """
        if not self._opened:
            raise RuntimeError("Kamera ist nicht geöffnet.")
        if not all(math.isfinite(v) and v > 0 for v in (exposure_us, fps)):
            raise CameraSettingsError("Belichtung und Bildrate müssen positiv und endlich sein.")
        rollback = []
        self._acquirer.stop()
        try:
            self._configure_settings(exposure_us, fps, rollback)
            name = (
                "ExposureTime" if _node(self._nm, "ExposureTime") is not None
                else "ExposureTimeAbs"
            )
            applied = {
                "exposure_us": float(getattr(self._nm, name).value),
                "fps": float(self._nm.AcquisitionFrameRate.value),
            }
            self._restart_after_settings()
        except Exception as exc:
            try:
                self._acquirer.stop()
                for name, previous in reversed(rollback):
                    if _value(self._nm, name) != previous:
                        getattr(self._nm, name).value = previous
                self._restart_after_settings()
                self._refresh_actual()
            except Exception as recovery:
                raise RuntimeError(
                    f"Kameraeinstellungen fehlgeschlagen: {exc}; "
                    f"Wiederherstellung fehlgeschlagen: {recovery}"
                ) from recovery
            raise CameraSettingsError(str(exc)) from exc
        self.config.exposure_us = applied["exposure_us"]
        self.config.fps = applied["fps"]
        self._refresh_actual()
        return applied

    def open(self) -> None:
        if self._opened:
            return
        if self.config.mode not in {"live", "acquisition_trigger", "frame_trigger"}:
            raise ValueError(f"Unbekannter Kameramodus: {self.config.mode}")
        if self.config.input_monitor not in {"auto", "counter", "line_status"}:
            raise ValueError(f"Unbekannte Eingangsüberwachung: {self.config.input_monitor}")
        from harvesters.core import Harvester

        paths = [self.config.cti_path] if self.config.cti_path else find_cti_paths()
        if not paths:
            raise RuntimeError(
                "Kein Baumer GenTL-Producer (.cti) gefunden. "
                "Baumer Camera Explorer/SDK installieren."
            )
        self.diagnostics = {"warnings": [], "cti_path": paths[0], "mode": self.config.mode}
        self._restore.clear()
        self._triggers.clear()
        self._first_acquisition_frame = True
        self._last_line_state = None
        self._last_counter_value = None
        self._counter_node = None
        self._counter_restore = None
        try:
            self._harvester = Harvester()
            self._harvester.add_file(paths[0], check_existence=True, check_validity=True)
            self._harvester.update()
            devices = list(self._harvester.device_info_list)
            self.diagnostics["devices"] = [
                {"model": device.model, "serial": device.serial_number} for device in devices
            ]
            matches = [
                device
                for device in devices
                if not self.config.serial or device.serial_number == self.config.serial
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Keine eindeutige USB-Kamera gefunden. CTI-Pfad und Seriennummer prüfen."
                )
            self._acquirer = self._harvester.create(matches[0])
            self._nm = self._acquirer.remote_device.node_map
            self._inspect()
            for saved in self._triggers:
                self._nm.TriggerSelector.value = saved["selector"]
                if _writable(_node(self._nm, "TriggerMode")):
                    self._nm.TriggerMode.value = "Off"
            self._set("AcquisitionMode", "Continuous")
            self._configure_settings(self.config.exposure_us, self.config.fps)
            if self.config.mode != "live":
                requested = (
                    "AcquisitionStart"
                    if self.config.mode == "acquisition_trigger"
                    else "FrameStart"
                )
                selected = next(
                    (saved for saved in self._triggers if saved["selector"] == requested), None
                )
                if selected is None:
                    supported = ", ".join(saved["selector"] for saved in self._triggers) or "keine"
                    raise RuntimeError(
                        f"{requested} wird nicht unterstützt (verfügbar: {supported}). "
                        "FrameStart erzeugt nur ein Bild je Puls; "
                        "für eine Folge Live + Lichtschranken-Erfassung verwenden."
                    )
                if self.config.trigger_line not in selected["sources"]:
                    raise RuntimeError(
                        f"Trigger-Eingang {self.config.trigger_line} nicht verfügbar: "
                        f"{selected['sources']}."
                    )
                self._nm.TriggerSelector.value = requested
                self._nm.TriggerSource.value = self.config.trigger_line
                if _writable(_node(self._nm, "TriggerActivation")):
                    self._nm.TriggerActivation.value = self.config.trigger_activation
                self._nm.TriggerMode.value = "On"
                self.diagnostics["hardware_trigger_monitor"] = requested
            else:
                use_counter = self.config.input_monitor != "line_status"
                if not use_counter or not self._configure_counter_monitor():
                    if self.config.input_monitor == "counter":
                        raise RuntimeError("Kein sicher nutzbarer Hardware-Zähler für den Eingang.")
                    self._configure_line_monitor()
            self._acquirer.num_buffers = max(self._acquirer.min_num_buffers, 16)
            self._acquirer.start()
            self._sequence = 0
            self._opened = True
            self._refresh_actual()
        except Exception as exc:
            self.close()
            message = str(exc) or type(exc).__name__
            if any(
                word in f"{type(exc).__name__} {message}".lower() for word in ("access", "resource")
            ):
                message += (
                    " Baumer Camera Explorer oder andere Kameraprogramme schließen "
                    "und erneut verbinden."
                )
            raise RuntimeError(message) from exc

    def _configure_counter_monitor(self) -> bool:
        """Reserve an unused counter and self-reset at its maximum, without reset commands."""
        nm = self._nm
        selector = _node(nm, "CounterSelector")
        old_selector = _value(nm, "CounterSelector")
        if old_selector is None or not _writable(selector):
            return False
        names = (
            "CounterEventSource",
            "CounterEventActivation",
            "CounterTriggerSource",
            "CounterResetSource",
            "CounterResetActivation",
            "CounterDuration",
            "CounterValue",
        )
        self.diagnostics["counters"] = []
        selected = False
        try:
            for entry in _entries(nm, "CounterSelector"):
                selector.value = entry
                saved = {name: _value(nm, name) for name in names}
                self.diagnostics["counters"].append({"selector": entry, **saved})
                trigger_node = _node(nm, "CounterTriggerSource")
                # Absent CounterTriggerSource means this model has no gated counter start.
                trigger_off = trigger_node is None or saved["CounterTriggerSource"] == "Off"
                if saved["CounterEventSource"] != "Off" or not trigger_off:
                    continue
                if saved["CounterResetSource"] != "Off":
                    continue
                changed_names = (
                    "CounterEventSource",
                    "CounterEventActivation",
                    "CounterResetSource",
                    "CounterResetActivation",
                    "CounterDuration",
                )
                if not all(_writable(_node(nm, name)) for name in changed_names):
                    continue
                activation = self.config.trigger_activation
                if activation == "BothEdges":
                    activation = "AnyEdge"
                if self.config.trigger_line not in _entries(nm, "CounterEventSource"):
                    continue
                if activation not in _entries(nm, "CounterEventActivation"):
                    continue
                reset_source = entry + "End"
                if reset_source not in _entries(nm, "CounterResetSource"):
                    continue
                if "RisingEdge" not in _entries(nm, "CounterResetActivation"):
                    continue
                try:
                    maximum = int(_node(nm, "CounterDuration").max)
                    baseline = int(saved["CounterValue"])
                except (TypeError, ValueError, AttributeError):
                    continue
                if maximum < 2 or baseline < 0 or baseline >= maximum:
                    continue
                self._counter_restore = (entry, saved)
                if not any(name == "CounterSelector" for name, _ in self._restore):
                    self._restore.append(("CounterSelector", old_selector))
                # Self-resetting CounterEnd avoids permanent saturation after 65535 pulses.
                nm.CounterDuration.value = maximum
                nm.CounterResetActivation.value = "RisingEdge"
                nm.CounterResetSource.value = reset_source
                nm.CounterEventActivation.value = activation
                nm.CounterEventSource.value = self.config.trigger_line
                self._counter_node = _node(nm, "CounterValue")
                self._last_counter_value = int(self._counter_node.value)
                self.diagnostics["hardware_trigger_monitor"] = "hardware_counter"
                self.diagnostics["trigger_line"] = self.config.trigger_line
                self.diagnostics["counter"] = {
                    "selector": entry,
                    "duration": maximum,
                    "reset_source": reset_source,
                    "activation": activation,
                    "initial_value": self._last_counter_value,
                    "observed_increments": 0,
                }
                self.diagnostics["warnings"].append(
                    "Hardware-Zähler erfasst Eingangsflanken zwischen Frames. "
                    "Verdrahtung, Polarität und reale kurze Lichtschranken-Pulse prüfen."
                )
                selected = True
                return True
        except Exception as exc:
            self._restore_counter()
            self.diagnostics["warnings"].append(f"Hardware-Zähler nicht nutzbar: {exc}")
        finally:
            if not selected and _writable(selector):
                selector.value = old_selector
        return False

    def _restore_counter(self) -> None:
        if self._counter_restore is None:
            return
        selector, saved = self._counter_restore
        try:
            self._nm.CounterSelector.value = selector
            self._nm.CounterEventSource.value = "Off"
            for name in (
                "CounterResetSource",
                "CounterResetActivation",
                "CounterDuration",
                "CounterEventActivation",
                "CounterValue",
                "CounterEventSource",
            ):
                if saved[name] is not None and _writable(_node(self._nm, name)):
                    getattr(self._nm, name).value = saved[name]
        except Exception as exc:
            self.diagnostics["warnings"].append(f"Zähler konnte nicht restauriert werden: {exc}")
        self._counter_restore = None
        self._counter_node = None

    def _configure_line_monitor(self) -> None:
        lines = _entries(self._nm, "LineSelector")
        if self.config.trigger_line in lines:
            self._set("LineSelector", self.config.trigger_line)
            self._line_node = _node(self._nm, "LineStatus")
            state = _value(self._nm, "LineStatus")
            if state is not None:
                self._last_line_state = bool(state)
                self.diagnostics["hardware_trigger_monitor"] = "line_status_poll"
                self.diagnostics["trigger_line"] = self.config.trigger_line
                self.diagnostics["warnings"].append(
                    "LineStatus wird zwischen Frames abgefragt. "
                    "Sehr kurze Lichtschranken-Pulse können fehlen; "
                    "Aufnahmen auf realen Pulsen prüfen."
                )
                return
        self._line_node = None
        self.diagnostics["hardware_trigger_monitor"] = "unavailable"
        self.diagnostics["warnings"].append(
            f"Lichtschranken-Eingang {self.config.trigger_line} nicht lesbar; "
            f"verfügbare Linien: {lines}."
        )

    def read(self, timeout: float = 0.5) -> CameraFrame | None:
        if not self._opened:
            raise RuntimeError("Kamera ist nicht geöffnet.")
        try:
            with self._acquirer.fetch(timeout=max(0.001, float(timeout))) as buffer:
                image = convert_component(buffer.payload.components[0])
                timestamp = time.perf_counter()
                self._sequence += 1
                try:
                    frame_id = int(buffer.frame_id)
                except Exception:
                    frame_id = self._sequence
                triggered = self.config.mode == "frame_trigger"
                if self.config.mode == "acquisition_trigger":
                    triggered = self._first_acquisition_frame
                    self._first_acquisition_frame = False
                elif self._counter_node is not None:
                    count = int(self._counter_node.value)
                    previous = self._last_counter_value
                    self._last_counter_value = count
                    triggered = previous is not None and count != previous
                    if triggered:
                        counter_info = self.diagnostics["counter"]
                        increments = (count - previous) % counter_info["duration"]
                        counter_info["observed_increments"] += increments
                elif self._line_node is not None:
                    state = bool(self._line_node.value)
                    previous = self._last_line_state
                    self._last_line_state = state
                    if previous is not None:
                        if self.config.trigger_activation == "RisingEdge":
                            triggered = state and not previous
                        elif self.config.trigger_activation == "FallingEdge":
                            triggered = previous and not state
                        elif self.config.trigger_activation in {"AnyEdge", "BothEdges"}:
                            triggered = state != previous
            return CameraFrame(image, timestamp, frame_id, triggered)
        except Exception as exc:
            if "timeout" in f"{type(exc).__name__} {exc}".lower():
                return None
            raise

    def rearm_acquisition(self) -> None:
        if self.config.mode != "acquisition_trigger" or not self._opened:
            return
        self._acquirer.stop()
        self._first_acquisition_frame = True
        self._acquirer.start()

    def close(self) -> None:
        self._opened = False
        if self._acquirer is not None:
            try:
                self._acquirer.stop()
            except Exception:
                pass
            self._restore_counter()
            for saved in self._triggers:
                try:
                    self._nm.TriggerSelector.value = saved["selector"]
                    self._nm.TriggerMode.value = "Off"
                    for name, key in (
                        ("TriggerSource", "source"),
                        ("TriggerActivation", "activation"),
                    ):
                        if saved[key] is not None and _writable(_node(self._nm, name)):
                            getattr(self._nm, name).value = saved[key]
                    if saved["mode"] is not None:
                        self._nm.TriggerMode.value = saved["mode"]
                except Exception as exc:
                    self.diagnostics["warnings"].append(
                        f"Trigger-Einstellungen konnten nicht wiederhergestellt werden: {exc}"
                    )
            for name, value in reversed(self._restore):
                try:
                    if _writable(_node(self._nm, name)):
                        getattr(self._nm, name).value = value
                except Exception as exc:
                    self.diagnostics["warnings"].append(
                        f"{name} konnte nicht wiederhergestellt werden: {exc}"
                    )
            try:
                self._acquirer.destroy()
            except Exception:
                pass
            self._acquirer = None
        if self._harvester is not None:
            try:
                self._harvester.reset()
            except Exception:
                pass
            self._harvester = None
        self._nm = None
        self._line_node = None
        self._counter_node = None

    def __enter__(self) -> BaumerCamera:
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
