"""Streaming compact media must preserve all original measurements and bounded RAM."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from bibazu_vision import review_video
from bibazu_vision.recording import CaptureConfig, CaptureManager, load_clip


@pytest.fixture
def manager(tmp_path):
    target = tmp_path / "trial"
    target.mkdir()
    (target / "capture-request.json").write_text(
        json.dumps(
            {
                "format": "review_video_v1",
                "run_id": "run",
                "trial_id": "trial",
            }
        )
    )
    capture = CaptureManager(
        CaptureConfig(start_mode="manual", pre_roll_seconds=0, max_memory_mb=4, max_seconds=10),
        tmp_path / "legacy",
    )
    capture.review_target = target
    capture.metadata = {
        "run_id": "run",
        "trial_id": "trial",
        "capture_fps": 68.0,
        "processing": {"roi": [10, 20, 1800, 900]},
    }
    yield capture
    capture.close(timeout=10)


def drain(manager):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if manager.status.buffered_bytes == 0:
            return
        time.sleep(0.002)
    raise AssertionError("Writer did not release captured images while recording")


def record(manager, count=32):
    manager.trigger(0)
    for index in range(count):
        image = np.full((964, 1920), index + 30, dtype=np.uint8)
        manager.ingest(
            image,
            index / 68,
            100 + index,
            {"present": True, "angle_deg": float(index), "motion_fraction": 0},
        )
        drain(manager)
        assert manager.status.buffered_bytes <= 4 * 1024**2
    manager.disarm()
    assert manager.wait_until_saved(10)
    assert manager.last_clip_path is not None, manager.status.error
    return load_clip(manager.last_clip_path)


def test_streams_beyond_memory_budget_without_npy_or_duplicate_output(manager):
    clip = record(manager)
    assert clip.manifest["schema_version"] == 2
    assert len(clip.frames) == len(clip.timestamps) == 32
    assert clip.manifest["dropped_frames"] == 0
    assert clip.frames[0].shape == (480, 960, 3)
    assert np.mean(clip.frames[31]) == pytest.approx(61, abs=5)
    assert np.mean(clip.frames[4]) == pytest.approx(34, abs=5)
    assert len(clip.frames.cache) <= 8
    assert clip.frame_ids == list(range(100, 132))
    assert clip.timestamps == pytest.approx([i / 68 for i in range(32)])
    assert [r["angle_deg"] for r in clip.measurements] == list(range(32))
    assert not list(clip.path.rglob("*.npy"))
    assert not manager.output_dir.exists()
    assert clip.path == manager.review_target / "clip"
    assert (clip.path / "review.mp4").stat().st_size < 32 * 964 * 1920 / 10
    assert manager.status.buffered_bytes == 0


def test_encoder_failure_keeps_measurements_and_never_falls_back_to_raw(manager, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("Encoder failed")

    monkeypatch.setattr(review_video.cv2, "VideoWriter", broken)
    clip = record(manager, 6)
    assert len(clip.frames) == 0
    assert len(clip.measurements) == 6
    assert clip.manifest["media"]["state"] == "failed"
    assert "Encoder failed" in clip.manifest["media"]["error"]
    assert manager.completed_clips[-1].error is None
    clip.export_csv(clip.path / "export.csv")
    assert len((clip.path / "export.csv").read_text().splitlines()) == 7
    assert not list(clip.path.rglob("*.npy"))


def test_recovery_finds_already_published_video_and_validates_reservation(manager):
    clip = record(manager, 2)
    assert review_video.recover_pending(manager.review_target) == clip.path
    (manager.review_target / "capture-request.json").write_text(
        json.dumps({"format": "review_video_v1", "run_id": "wrong", "trial_id": "trial"})
    )
    assert review_video.recover_pending(manager.review_target) is None


def test_publication_retries_without_overwriting_other_data(tmp_path, monkeypatch):
    pending, final = tmp_path / "clip.tmp", tmp_path / "clip"
    pending.mkdir()
    (pending / "manifest.json").write_text("{}")
    rename = Path.rename
    attempts = []

    def locked_once(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError("Temporarily locked")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", locked_once)
    monkeypatch.setattr(review_video.time, "sleep", lambda _: None)
    assert review_video.finalize_pending(pending) == final
    assert len(attempts) == 3
    assert review_video.finalize_pending(pending) == final
    pending.mkdir()
    with pytest.raises(FileExistsError):
        review_video.publish(pending, final)
    assert (final / "manifest.json").exists()


def test_permanent_lock_preserves_complete_pending_recording(tmp_path, monkeypatch):
    pending = tmp_path / "clip.tmp"
    pending.mkdir()
    (pending / "manifest.json").write_text("{}")

    def locked(*args):
        raise PermissionError("Locked")

    monkeypatch.setattr(Path, "rename", locked)
    monkeypatch.setattr(review_video.time, "sleep", lambda _: None)
    assert review_video.publish(pending, tmp_path / "clip") == pending
    assert (pending / "manifest.json").exists()


def test_capture_target_requires_matching_reservation_and_free_space(tmp_path, monkeypatch):
    marker = tmp_path / "capture-request.json"
    marker.write_text(json.dumps({"format": "review_video_v1", "run_id": "r", "trial_id": "t"}))
    assert review_video.validate_target(str(tmp_path), "r", "t") == tmp_path
    with pytest.raises(ValueError, match="belong"):
        review_video.validate_target(str(tmp_path), "wrong", "t")
    monkeypatch.setattr(review_video.shutil, "disk_usage", lambda _: type("D", (), {"free": 10})())
    with pytest.raises(OSError, match="1 GiB"):
        review_video.validate_target(str(tmp_path), "r", "t")


def test_journal_write_failure_recovers_previously_flushed_measurements(manager, monkeypatch):
    dumps = review_video.json.dumps

    def full(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("video_frame_index") == 3:
            raise OSError("Simulated disk full")
        return dumps(value, *args, **kwargs)

    monkeypatch.setattr(review_video.json, "dumps", full)
    clip = record(manager, 8)
    assert clip.manifest["aborted"]
    assert clip.manifest["stop_reason"] == "write_error"
    assert [m["angle_deg"] for m in clip.measurements] == [0, 1, 2]
    assert clip.manifest["media"]["state"] == "failed"
    assert manager.completed_clips[-1].error


def test_overlay_coordinates_follow_video_crop_and_scale(manager, monkeypatch):
    from types import SimpleNamespace

    from bibazu_vision.app import MainWindow

    clip = record(manager, 2)
    original = [[10, 20], [1810, 20], [1810, 920], [10, 920]]
    clip.measurements[0]["box"] = original
    polygons = []
    monkeypatch.setattr(
        review_video.cv2,
        "polylines",
        lambda image, points, *args: polygons.append(points[0].tolist()),
    )
    image = MainWindow.review_image(SimpleNamespace(clip=clip), 0)
    assert image.shape == (480, 960, 3)
    assert polygons == [[[0, 0], [960, 0], [960, 480], [0, 480]]]
    assert clip.measurements[0]["box"] == original
