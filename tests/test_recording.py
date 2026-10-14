from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import numpy as np
import pytest

from bibazu_vision.recording import CaptureConfig, CaptureManager, load_clip


@pytest.fixture
def managers(tmp_path):
    created = []

    def make(**overrides):
        defaults = dict(pre_roll_seconds=0.1, quiet_seconds=0.1, wait_seconds=0.5, max_seconds=2)
        defaults.update(overrides)
        manager = CaptureManager(CaptureConfig(**defaults), tmp_path / "captures")
        created.append(manager)
        return manager

    yield make
    for manager in created:
        assert manager.close(timeout=5)


def frame(value=0):
    return np.full((8, 10), value, dtype=np.uint8)


def detection(present=True, motion=0.02, angle=0.0):
    return {"present": present, "motion_fraction": motion, "angle_deg": angle, "confidence": 0.9}


def saved(manager):
    assert manager.wait_until_saved(5)
    assert manager.last_clip_path is not None
    return load_clip(manager.last_clip_path)


def test_hardware_requires_trigger_and_waits_for_visible_motion(managers):
    manager = managers(start_mode="hardware")
    manager.arm()
    assert manager.ingest(frame(), 0.0, 1, detection()).state == "armed"
    assert manager.ingest(frame(), 0.02, 2, detection(motion=0), triggered=True).state == "waiting"
    assert manager.ingest(frame(), 0.1, 3, detection()).state == "moving"
    manager.ingest(frame(), 0.21, 4, detection(present=False, motion=0))
    clip = saved(manager)
    assert clip.manifest["stop_reason"] == "part_left"
    assert clip.manifest["trigger_source"] == "hardware"
    assert clip.manifest["motion_seen"] is True
    assert clip.frame_ids == [1, 2, 3, 4]
    assert manager.poll(0.22).state == "armed"


def test_wait_timeout_without_frames_is_saved_and_never_claims_a_passage(managers):
    manager = managers(start_mode="hardware")
    assert manager.trigger(1.0).state == "waiting"
    assert manager.poll(1.6).reason == "no_motion_timeout"
    clip = saved(manager)
    assert clip.manifest["motion_seen"] is False
    assert clip.manifest["stop_reason"] == "no_motion_timeout"
    assert clip.frames == []


def test_implicit_trigger_and_poll_use_current_high_resolution_clock(managers, monkeypatch):
    clock = [1.0]
    monkeypatch.setattr("bibazu_vision.recording.time.perf_counter", lambda: clock[0])
    manager = managers(start_mode="hardware")
    manager.arm()
    manager.ingest(frame(), 0.8, 1, detection(motion=0))
    assert manager.trigger().state == "waiting"
    clock[0] = 1.6
    assert manager.poll().reason == "no_motion_timeout"
    clip = saved(manager)
    assert clip.manifest["trigger_timestamp"] == 1.0
    assert clip.manifest["end_timestamp"] == 1.6
    assert clip.frame_ids == []


def test_motion_needs_presence_and_lingering_part_is_not_finished(managers):
    manager = managers(start_mode="motion", pre_roll_seconds=0)
    manager.arm()
    assert manager.ingest(frame(), 0.0, 1, detection(present=False)).state == "armed"
    assert manager.ingest(frame(), 0.1, 2, detection()).state == "moving"
    assert manager.ingest(frame(), 0.8, 3, detection(motion=0)).state == "moving"
    assert manager.poll(2.2).reason == "max_duration"
    clip = saved(manager)
    assert clip.manifest["stop_reason"] == "max_duration"
    assert clip.frame_ids == [2, 3]


def test_exit_motion_must_settle_before_part_left(managers):
    manager = managers(pre_roll_seconds=0)
    manager.arm()
    manager.ingest(frame(), 0.0, 1, detection())
    assert manager.ingest(frame(), 0.05, 2, detection(present=False)).state == "moving"
    assert manager.ingest(frame(), 0.1, 3, detection(present=False, motion=0)).state == "moving"
    manager.ingest(frame(), 0.16, 4, detection(present=False, motion=0))
    assert saved(manager).manifest["stop_reason"] == "part_left"


def test_snapshot_immutable_originals_and_overlay_metadata_round_trip(managers, tmp_path):
    @dataclass
    class Detection:
        contour: np.ndarray
        center: tuple[float, float] = (5.0, 4.0)
        box: tuple = ((1, 1), (8, 1), (8, 6), (1, 6))
        area: float = 35.0
        angle_deg: float = 1.25
        axis_deg: float = 2.5
        confidence: float = 0.9
        touching_border: bool = False
        processing_ms: float = 0.7
        motion_fraction: float = 0.02

    manager = managers(pre_roll_seconds=0)
    manager.arm()
    image = frame(17)
    result = Detection(np.ones((4, 1, 2), dtype=np.int32))
    manager.ingest(image, 10.0, 101, result)
    image[:] = 200
    result.contour[:] = 999
    manager.disarm()
    clip = saved(manager)
    assert np.all(clip.frames[0] == 17)
    assert not clip.frames[0].flags.writeable
    assert clip.measurements[0]["present"] is True
    assert clip.measurements[0]["box"] == [[1.0, 1.0], [8.0, 1.0], [8.0, 6.0], [1.0, 6.0]]
    assert "contour" not in clip.measurements[0]
    assert clip.measurements[0]["angle_deg"] == 1.25
    assert clip.timestamps == [10.0]
    assert clip.frame_ids == [101]
    export = clip.export_csv(tmp_path / "export.csv")
    assert "1.25" in export.read_text(encoding="utf-8")
    assert (clip.path / "measurements.csv").exists()
    assert not list(clip.path.parent.glob("*.tmp"))


def test_memory_overflow_is_explicit_and_persists_retained_frames(managers):
    # Each tiny frame reserves its pixels and 512 bytes for bounded metadata.
    manager = managers(pre_roll_seconds=0, max_memory_mb=0.0012)
    manager.arm()
    manager.ingest(frame(1), 0.0, 1, detection())
    manager.ingest(frame(2), 0.01, 2, detection())
    status = manager.ingest(frame(3), 0.02, 3, detection())
    assert status.reason == "memory_limit"
    assert status.dropped_frames == 1
    assert status.buffered_bytes <= int(0.0012 * 1024 * 1024)
    clip = saved(manager)
    assert clip.manifest["aborted"] is True
    assert clip.manifest["stop_reason"] == "memory_limit"
    assert clip.manifest["dropped_frames"] == 1
    assert clip.frame_ids == [1, 2]
    assert manager.status.buffered_bytes == 0


def test_writer_is_nonblocking_and_its_frames_share_the_memory_budget(managers):
    manager = managers(
        start_mode="manual", pre_roll_seconds=0, max_memory_mb=0.002, writer_queue_depth=1
    )
    entered, release = threading.Event(), threading.Event()
    persist = manager._persist

    def slow_persist(capture):
        entered.set()
        assert release.wait(5)
        return persist(capture)

    manager._persist = slow_persist
    manager.trigger(0.0)
    manager.ingest(frame(), 0.01, 1, detection())
    manager.disarm()
    assert entered.wait(5)
    try:
        assert manager.status.buffered_bytes > 0
        assert manager.trigger(0.1).reason == "writer_busy"
        assert manager.status.rejected_triggers == 1
        assert manager.ingest(frame(), 0.11, 2, detection()).writer_pending == 1
        assert manager.status.buffered_bytes <= int(0.002 * 1024 * 1024)
    finally:
        release.set()
    saved(manager)
    assert manager.status.buffered_bytes == 0


def test_hardware_clip_contains_only_recent_pre_roll_and_detects_frame_id_gaps(managers):
    manager = managers(start_mode="hardware")
    manager.arm()
    manager.ingest(frame(), 0.0, 1, detection(motion=0))
    manager.ingest(frame(), 0.4, 8, detection(motion=0))
    manager.ingest(frame(), 0.45, 9, detection(), triggered=True)
    manager.ingest(frame(), 0.56, 12, detection(present=False, motion=0))
    clip = saved(manager)
    assert clip.frame_ids == [8, 9, 12]
    assert clip.manifest["frame_id_gaps"] == 2


def test_frame_dimensions_reset_aborts_previous_clip_safely(managers):
    manager = managers(pre_roll_seconds=0, start_mode="manual")
    manager.trigger(0)
    manager.ingest(frame(), 0.1, 1, detection())
    manager.ingest(np.zeros((12, 10), np.uint8), 0.2, 2, detection())
    clip = saved(manager)
    assert clip.manifest["stop_reason"] == "image_shape_changed"
    assert clip.manifest["aborted"] is True
    assert len(clip.frames) == 1


def test_close_flushes_active_clip_and_surfaces_disk_failure(managers):
    manager = managers(pre_roll_seconds=0)
    manager.arm()
    manager.ingest(frame(), 0.0, 1, detection())
    assert manager.close(timeout=5)
    assert load_clip(manager.last_clip_path).manifest["stop_reason"] == "closed"
    with pytest.raises(RuntimeError):
        manager.arm()

    failing = managers(pre_roll_seconds=0)
    failing._persist = lambda _capture: (_ for _ in ()).throw(OSError("disk full"))
    failing.trigger(0.0)
    failing.ingest(frame(), 0.0, 1, detection())
    failing.disarm()
    assert failing.wait_until_saved(5)
    assert "disk full" in failing.status.error
    assert failing.last_clip_path is None
    assert failing.completed_clips[-1].error == "OSError: disk full"
    assert failing.status.buffered_bytes == 0


def test_loading_rejects_external_frame_paths(managers, tmp_path):
    manager = managers(pre_roll_seconds=0)
    manager.trigger(0)
    manager.ingest(frame(), 0.0, 1, detection())
    manager.disarm()
    clip = saved(manager)
    external = tmp_path / "external.npy"
    np.save(external, frame())
    manifest = clip.manifest
    manifest["frames"][0]["file"] = str(external)
    (clip.path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_clip(clip.path)


def test_configuration_requires_idle_and_capture_metadata_is_a_snapshot(managers):
    manager = managers(start_mode="manual")
    manager.arm()
    manager.ingest(frame(), 0.0, 1, detection(motion=0))
    assert manager.status.buffered_frames == 1
    config = CaptureConfig(start_mode="manual", pre_roll_seconds=0, max_seconds=3, max_memory_mb=1)
    manager.configure(config)
    assert manager.status.buffered_frames == 0
    manager.metadata = {"processing_config": {"reference_angle_deg": 3.5}}
    manager.trigger(0.1)
    manager.ingest(frame(), 0.1, 2, detection())
    with pytest.raises(RuntimeError, match="idle"):
        manager.configure(config)
    manager.metadata["processing_config"]["reference_angle_deg"] = 99
    manager.disarm()
    clip = saved(manager)
    assert clip.manifest["capture_config"]["max_seconds"] == 3
    assert clip.manifest["metadata"]["processing_config"]["reference_angle_deg"] == 3.5
    with pytest.raises(ValueError, match="writer_queue_depth"):
        manager.configure(CaptureConfig(writer_queue_depth=2))


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_memory_mb": 0},
        {"wait_seconds": -1},
        {"start_mode": "unknown"},
        {"motion_threshold": float("nan")},
        {"writer_queue_depth": 0},
    ],
)
def test_invalid_config_rejected(arguments):
    with pytest.raises(ValueError):
        CaptureConfig(**arguments)
