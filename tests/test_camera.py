"""Checks for persistent GenTL pixels and honest hardware trigger semantics."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from bibazu_vision.camera import (
    BaumerCamera,
    CameraConfig,
    CameraSettingsError,
    convert_component,
)


def test_mono_frame_owns_pixels_after_gen_tl_buffer_reuse():
    source = np.arange(24, dtype=np.uint8)
    image = convert_component(SimpleNamespace(width=6, height=4, data_format="Mono8", data=source))
    source[:] = 0
    assert image.shape == (4, 6)
    assert image[3, 5] == 23
    assert not np.shares_memory(image, source)


def test_bayer_rg_is_rgb_sensor_pattern_and_returns_bgr():
    source = np.zeros((8, 8), dtype=np.uint8)
    source[::2, ::2] = 200  # PFNC BayerRG is red at the upper left.
    source[::2, 1::2] = 100
    source[1::2, ::2] = 100
    source[1::2, 1::2] = 50
    image = convert_component(
        SimpleNamespace(width=8, height=8, data_format="BayerRG8", data=source)
    )
    assert image[4, 4].tolist() == [50, 100, 200]


def test_packed_format_does_not_silently_produce_wrong_pixels():
    with pytest.raises(ValueError, match="Gepacktes"):
        convert_component(
            SimpleNamespace(
                width=8, height=8, data_format="Mono12p", data=np.zeros(96, dtype=np.uint8)
            )
        )


class Node:
    def __init__(self, value, choices=()):
        self.value = value
        self.choices = list(choices)


class FakeAcquirer:
    min_num_buffers = 3

    def __init__(self):
        self.node_map = SimpleNamespace(
            TriggerSelector=Node("FrameStart", ["FrameStart"]),
            TriggerMode=Node("On", ["On", "Off"]),
            TriggerSource=Node("Line0", ["Line0", "Software"]),
            TriggerActivation=Node("FallingEdge", ["FallingEdge", "RisingEdge"]),
            AcquisitionMode=Node("SingleFrame", ["SingleFrame", "Continuous"]),
            LineSelector=Node("Line0", ["Line0"]),
            LineStatus=Node(False),
        )
        self.remote_device = SimpleNamespace(node_map=self.node_map)
        self.started = False
        self.destroyed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def destroy(self):
        self.destroyed = True


class FakeHarvester:
    def __init__(self, acquirer):
        self.acquirer = acquirer
        self.device_info_list = [SimpleNamespace(model="VCXU-test", serial_number="123")]

    def add_file(self, *args, **kwargs):
        pass

    def update(self):
        pass

    def create(self, device):
        return self.acquirer

    def reset(self):
        pass


def patches(acquirer):
    return (
        patch("harvesters.core.Harvester", return_value=FakeHarvester(acquirer)),
        patch("bibazu_vision.camera._writable", side_effect=lambda node: node is not None),
        patch(
            "bibazu_vision.camera._entries",
            side_effect=lambda nm, name: getattr(nm, name, SimpleNamespace(choices=[])).choices,
        ),
    )


def test_missing_acquisition_start_fails_and_restores_original_camera_state():
    acquirer = FakeAcquirer()
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti", mode="acquisition_trigger"))
    with p1, p2, p3:
        with pytest.raises(RuntimeError, match="AcquisitionStart wird nicht unterstützt"):
            camera.open()
    assert acquirer.node_map.AcquisitionMode.value == "SingleFrame"
    assert acquirer.node_map.TriggerMode.value == "On"
    assert acquirer.node_map.TriggerSource.value == "Line0"
    assert acquirer.node_map.TriggerActivation.value == "FallingEdge"
    assert acquirer.destroyed and not camera.is_open


def test_live_close_restores_mode_and_trigger_settings():
    acquirer = FakeAcquirer()
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti"))
    with p1, p2, p3:
        camera.open()
        assert acquirer.node_map.AcquisitionMode.value == "Continuous"
        assert acquirer.node_map.TriggerMode.value == "Off"
        assert camera.diagnostics["hardware_trigger_monitor"] == "line_status_poll"
        camera.close()
    assert acquirer.node_map.AcquisitionMode.value == "SingleFrame"
    assert acquirer.node_map.TriggerMode.value == "On"
    assert acquirer.node_map.TriggerActivation.value == "FallingEdge"


class RecycledBuffer:
    def __init__(self, frame_id):
        self.data = np.full(12, 37, dtype=np.uint8)
        self.frame_id = frame_id
        self.payload = SimpleNamespace(
            components=[SimpleNamespace(width=4, height=3, data_format="Mono8", data=self.data)]
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Mimic the producer taking ownership and overwriting the queued buffer.
        self.data[:] = 99


def test_read_preserves_pixels_and_only_marks_a_live_input_edge_once():
    camera = BaumerCamera(CameraConfig())
    camera._opened = True
    camera._last_line_state = False
    camera._line_node = Node(True)
    camera._acquirer = SimpleNamespace(fetch=lambda **kwargs: RecycledBuffer(42))
    first = camera.read()
    second = camera.read()
    assert first.triggered and not second.triggered
    assert first.frame_id == 42
    assert np.all(first.image == 37)
    assert second.timestamp >= first.timestamp


def test_acquisition_trigger_marks_only_first_frame_until_rearmed():
    camera = BaumerCamera(CameraConfig(mode="acquisition_trigger"))
    camera._opened = True
    camera._acquirer = SimpleNamespace(
        fetch=lambda **kwargs: RecycledBuffer(42), stop=lambda: None, start=lambda: None
    )
    assert camera.read().triggered
    assert not camera.read().triggered
    camera.rearm_acquisition()
    assert camera.read().triggered


class CounterMap:
    """Counter fields depend on CounterSelector, as they do in a GenICam map."""

    def __init__(self, basic):
        self.__dict__.update(vars(basic))
        self.CounterSelector = Node("Counter2", ["Counter1", "Counter2"])
        self.counters = {}
        for name in ("Counter1", "Counter2"):
            duration = Node(4)
            duration.max = 16
            self.counters[name] = {
                "CounterEventSource": Node("Off", ["Off", "Line0", "ExposureActive"]),
                "CounterEventActivation": Node("FallingEdge", ["RisingEdge", "FallingEdge"]),
                "CounterResetSource": Node("Off", ["Off", "Counter1End", "Counter2End"]),
                "CounterResetActivation": Node("FallingEdge", ["RisingEdge", "FallingEdge"]),
                "CounterDuration": duration,
                "CounterValue": Node(0),
            }
        self.counters["Counter2"]["CounterEventSource"].value = "ExposureActive"

    def __getattr__(self, name):
        counter = self.counters[self.CounterSelector.value]
        if name in counter:
            return counter[name]
        raise AttributeError(name)


def counter_acquirer():
    acquirer = FakeAcquirer()
    acquirer.node_map = CounterMap(acquirer.node_map)
    acquirer.remote_device.node_map = acquirer.node_map
    acquirer.fetch = lambda **kwargs: RecycledBuffer(42)
    return acquirer


def test_hardware_counter_latches_short_pulse_even_when_line_is_already_false():
    acquirer = counter_acquirer()
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti"))
    with p1, p2, p3:
        camera.open()
        assert camera.diagnostics["hardware_trigger_monitor"] == "hardware_counter"
        assert acquirer.node_map.CounterSelector.value == "Counter1"
        assert not acquirer.node_map.LineStatus.value
        acquirer.node_map.CounterValue.value += 1
        assert camera.read().triggered
        assert not camera.read().triggered
        camera.close()
    # Restore the same counter context before restoring the user's selector.
    first = acquirer.node_map.counters["Counter1"]
    assert first["CounterEventSource"].value == "Off"
    assert first["CounterEventActivation"].value == "FallingEdge"
    assert first["CounterResetSource"].value == "Off"
    assert first["CounterResetActivation"].value == "FallingEdge"
    assert first["CounterDuration"].value == 4
    assert first["CounterValue"].value == 0
    assert acquirer.node_map.CounterSelector.value == "Counter2"
    assert acquirer.node_map.CounterEventSource.value == "ExposureActive"


def test_counter_rollover_is_detected_and_automatic_reset_prevents_saturation():
    acquirer = counter_acquirer()
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti"))
    with p1, p2, p3:
        camera.open()
        assert acquirer.node_map.CounterDuration.value == 16
        assert acquirer.node_map.CounterResetSource.value == "Counter1End"
        acquirer.node_map.CounterValue.value = 15
        assert camera.read().triggered
        acquirer.node_map.CounterValue.value = 0
        assert camera.read().triggered
        assert camera.diagnostics["counter"]["observed_increments"] == 16
        assert not camera.read().triggered
        camera.close()


def test_busy_counters_are_left_untouched_and_line_status_is_used_in_auto_mode():
    acquirer = counter_acquirer()
    acquirer.node_map.counters["Counter1"]["CounterEventSource"].value = "ExposureActive"
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti"))
    with p1, p2, p3:
        camera.open()
        assert camera.diagnostics["hardware_trigger_monitor"] == "line_status_poll"
        camera.close()
    for counter in acquirer.node_map.counters.values():
        assert counter["CounterEventSource"].value == "ExposureActive"
        assert counter["CounterDuration"].value == 4
    assert acquirer.node_map.CounterSelector.value == "Counter2"


def test_explicit_counter_mode_rejects_saturated_counter_without_resetting_it():
    acquirer = counter_acquirer()
    acquirer.node_map.counters["Counter1"]["CounterValue"].value = 16
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti", input_monitor="counter"))
    with p1, p2, p3:
        with pytest.raises(RuntimeError, match="Kein sicher nutzbarer Hardware-Zähler"):
            camera.open()
    assert acquirer.node_map.counters["Counter1"]["CounterValue"].value == 16
    assert acquirer.node_map.CounterSelector.value == "Counter2"


def autogain_acquirer():
    acquirer = FakeAcquirer()
    nm = acquirer.node_map
    nm.ExposureAuto = Node("Continuous", ["Off", "Continuous"])
    nm.ExposureTime = Node(20331.0)
    nm.GainAuto = Node("Continuous", ["Off", "Continuous"])
    nm.Gain = Node(1.0)
    nm.BalanceWhiteAuto = Node("Continuous", ["Off", "Continuous"])
    nm.ColorTransformationAuto = Node("Once", ["Off", "Once", "Continuous"])
    nm.Gamma = Node(1.2)
    return acquirer


def test_manual_exposure_freezes_all_auto_controls_without_changing_gain_or_gamma():
    acquirer = autogain_acquirer()
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti", exposure_us=20331))
    with p1, p2, p3:
        camera.open()
        for name in ("ExposureAuto", "GainAuto", "BalanceWhiteAuto", "ColorTransformationAuto"):
            assert getattr(acquirer.node_map, name).value == "Off"
        assert acquirer.node_map.Gain.value == 1.0
        assert acquirer.node_map.Gamma.value == 1.2
        camera.close()
    for name in ("ExposureAuto", "GainAuto", "BalanceWhiteAuto"):
        assert getattr(acquirer.node_map, name).value == "Continuous"
    assert acquirer.node_map.ColorTransformationAuto.value == "Once"
    assert acquirer.node_map.ExposureTime.value == 20331.0
    assert acquirer.node_map.Gain.value == 1.0
    assert acquirer.node_map.Gamma.value == 1.2


def test_automatic_exposure_probe_preserves_gain_and_color_auto_controls():
    acquirer = autogain_acquirer()
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti"))
    with p1, p2, p3:
        camera.open()
        for name in ("ExposureAuto", "GainAuto", "BalanceWhiteAuto"):
            assert getattr(acquirer.node_map, name).value == "Continuous"
        assert acquirer.node_map.ColorTransformationAuto.value == "Once"
        camera.close()


def settings_acquirer(exposure_name="ExposureTime"):
    acquirer = autogain_acquirer()
    if exposure_name == "ExposureTimeAbs":
        acquirer.node_map.ExposureTimeAbs = acquirer.node_map.ExposureTime
        del acquirer.node_map.ExposureTime
    acquirer.node_map.AcquisitionFrameRate = Node(50.0)
    acquirer.node_map.AcquisitionFrameRateEnable = Node(False)
    return acquirer


@pytest.mark.parametrize("exposure_name", ["ExposureTime", "ExposureTimeAbs"])
@pytest.mark.parametrize("mode", ["live", "frame_trigger", "acquisition_trigger"])
def test_live_settings_reuse_device_refresh_diagnostics_and_restore_originals(exposure_name, mode):
    acquirer = settings_acquirer(exposure_name)
    nm = acquirer.node_map
    nm.TriggerSelector.choices.append("AcquisitionStart")
    original = {name: node.value for name, node in vars(nm).items()}
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti", mode=mode, exposure_us=1000, fps=100))
    with p1, p2, p3:
        camera.open()
        trigger = (nm.TriggerSelector.value, nm.TriggerMode.value, nm.TriggerActivation.value)
        with patch.object(acquirer, "start", wraps=acquirer.start) as start:
            for exposure, fps in [(5000, 80), (10000, 60)]:
                result = camera.apply_settings(exposure, fps)
                assert result == {"exposure_us": exposure, "fps": fps}
                assert camera.config.exposure_us == exposure
                assert camera.config.fps == fps
                assert camera.diagnostics["actual"][exposure_name] == exposure
                assert camera.diagnostics["actual"]["AcquisitionFrameRate"] == fps
                assert camera._acquirer is acquirer and acquirer.started
                assert not acquirer.destroyed
                assert (nm.TriggerSelector.value, nm.TriggerMode.value,
                        nm.TriggerActivation.value) == trigger
                assert camera._first_acquisition_frame
            assert start.call_count == 2
        camera.close()
    assert {name: node.value for name, node in vars(nm).items()} == original


class RejectingNode(Node):
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value == 999:
            raise ValueError("outside camera range")
        self._value = value


def test_live_settings_roll_back_partial_update_and_auto_controls_without_disconnect():
    acquirer = settings_acquirer()
    nm = acquirer.node_map
    nm.AcquisitionFrameRate = RejectingNode(50)
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti"))
    with p1, p2, p3:
        camera.open()
        previous = dict(camera.diagnostics["actual"])
        with pytest.raises(CameraSettingsError, match="outside camera range"):
            camera.apply_settings(7000, 999)
        assert acquirer.started and not acquirer.destroyed and camera.is_open
        assert camera.diagnostics["actual"] == previous
        assert camera.config.exposure_us is None and camera.config.fps is None
        # A later valid update still succeeds on this connection.
        camera.apply_settings(5000, 80)
        camera.close()
    assert nm.ExposureTime.value == 20331
    assert nm.AcquisitionFrameRate.value == 50
    assert nm.ExposureAuto.value == "Continuous"


@pytest.mark.parametrize("recoverable", [True, False])
def test_settings_restart_failure_recovers_or_requires_camera_shutdown(recoverable):
    acquirer = settings_acquirer()
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti"))
    with p1, p2, p3:
        camera.open()
        previous = dict(camera.diagnostics["actual"])
        failures = [OSError("restart failed"), None if recoverable else OSError("USB lost")]
        with patch.object(acquirer, "start", side_effect=failures):
            with pytest.raises(RuntimeError) as error:
                camera.apply_settings(5000, 80)
        assert isinstance(error.value, CameraSettingsError) is recoverable
        if recoverable:
            assert camera.diagnostics["actual"] == previous
        else:
            assert "Wiederherstellung fehlgeschlagen" in str(error.value)
        camera.close()
    assert acquirer.destroyed


@pytest.mark.parametrize("exposure,fps", [(float("nan"), 50), (5000, 0)])
def test_invalid_settings_do_not_pause_acquisition(exposure, fps):
    acquirer = settings_acquirer()
    p1, p2, p3 = patches(acquirer)
    camera = BaumerCamera(CameraConfig(cti_path="test.cti"))
    with p1, p2, p3:
        camera.open()
        with patch.object(acquirer, "stop") as stop:
            with pytest.raises(CameraSettingsError):
                camera.apply_settings(exposure, fps)
            stop.assert_not_called()
        camera.close()
