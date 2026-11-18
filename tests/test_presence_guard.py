"""Qualify geometry over time without consuming single-shot trials on brief intrusions."""

import pytest
from test_recording import detection, frame, saved
from test_recording import managers as managers

from bibazu_vision.recording import CaptureConfig


def geometry(**changes):
    return {
        **detection(motion=0),
        "area": 20.0,
        "box": [[1, 1], [7, 1], [7, 4], [1, 4]],
        "touching_border": False,
        **changes,
    }


def test_short_intrusions_do_not_consume_trial_and_real_part_keeps_pre_roll(managers):
    manager = managers(start_mode="presence", presence_confirm_seconds=0.1, pre_roll_seconds=0)
    assert manager.config.pre_roll_seconds >= 0.15
    manager.arm(single_shot=True)
    for i in range(3):
        manager.ingest(frame(), i * 0.01, i, geometry())
    manager.ingest(frame(), 0.03, 3, geometry(present=False, angle_deg=None))
    assert manager.status.state == "armed"
    assert not manager.completed_clips and not manager.output_dir.exists()
    for i in range(4, 15):
        # Position changes are normal band motion and must not reset shape confirmation.
        box = [[x + i, y] for x, y in geometry()["box"]]
        manager.ingest(frame(), i * 0.01, i, geometry(box=box))
    assert manager.status.state in {"waiting", "moving"}
    manager.ingest(frame(), 0.15, 15, geometry(present=False, angle_deg=None))
    manager.ingest(frame(), 0.26, 16, geometry(present=False, angle_deg=None))
    clip = saved(manager)
    assert clip.manifest["stop_reason"] == "part_left"
    assert clip.timestamps[0] <= 0.04
    assert len(manager.completed_clips) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"angle_deg": None},
        {"confidence": 0.19},
        {"touching_border": True},
        {"box": None},
        {"area": 0},
        {"present": False},
    ],
)
def test_unmeasurable_geometry_never_qualifies(managers, change):
    manager = managers(start_mode="presence", presence_confirm_seconds=0.1)
    manager.arm()
    for i in range(30):
        manager.ingest(frame(), i * 0.01, i, geometry(**change))
    assert manager.status.state == "armed"
    assert not manager.status.writer_pending


@pytest.mark.parametrize("change", ["angle", "area", "box", "gap", "invalid", "rearm"])
def test_candidate_resets_after_interruption_or_changed_geometry(managers, change):
    manager = managers(start_mode="presence", presence_confirm_seconds=0.1)
    manager.arm()
    for i in range(9):
        manager.ingest(frame(), i * 0.01, i, geometry())
    stamp, altered = 0.09, geometry()
    if change == "angle":
        altered["angle_deg"] = 10
    elif change == "area":
        altered["area"] = 40
    elif change == "box":
        altered["box"] = [[1, 1], [15, 1], [15, 4], [1, 4]]
    elif change == "gap":
        stamp = 0.2
    elif change == "invalid":
        altered["angle_deg"] = None
    else:
        manager.disarm()
        manager.arm()
    manager.ingest(frame(), stamp, 9, altered)
    manager.ingest(frame(), stamp + 0.01, 10, geometry())
    assert manager.status.state == "armed"


def test_manual_start_does_not_require_geometry_confirmation(managers):
    manager = managers(start_mode="manual", presence_confirm_seconds=0.1)
    manager.trigger(0)
    manager.ingest(frame(), 0, 1, geometry(angle_deg=None))
    assert manager.status.state == "waiting"


@pytest.mark.parametrize("seconds", [-1, 0.01, 0.6, float("nan"), float("inf")])
def test_guard_duration_is_validated(seconds):
    with pytest.raises(ValueError):
        CaptureConfig(presence_confirm_seconds=seconds)
