"""Full-rate angle continuity, explicit gaps and peak-preserving plot reduction."""

import time
from types import SimpleNamespace

import pytest

from bibazu_vision.angle_history import MeasurementHistory, envelope_points, gap_spans


def rows(angles, *, start=100, sequence=1, frame_id=1, dt=1 / 142, epoch=0):
    return [
        {
            "sequence": sequence + i,
            "timestamp": start + i * dt,
            "frame_id": frame_id + i,
            "angle_deg": angle,
            "axis_deg": angle,
            "valid": angle is not None,
            "present": True,
            "touching_border": False,
            "epoch": epoch,
        }
        for i, angle in enumerate(angles)
    ]


@pytest.mark.parametrize(
    "angles, expected",
    [
        (
            [0, -45, -89, 89, 45, 0, -45, -89, 89, 45, 0],
            [0, -45, -89, -91, -135, -180, -225, -269, -271, -315, -360],
        ),
        ([0, 45, 89, -89, -45, 0], [0, 45, 89, 91, 135, 180]),
        ([0, 40, 80, -80, 80, 40, 0], [0, 40, 80, 100, 80, 40, 0]),
    ],
)
def test_continuity_crosses_both_boundaries_and_retains_originals(angles, expected):
    history = MeasurementHistory()
    batch = rows(angles)
    assert history.ingest(batch) == len(angles)
    actual = history.visible(100, 101)
    assert [row.raw_angle for row in actual] == angles
    assert [row.continuous_angle for row in actual] == expected
    assert len({row.segment for row in actual}) == 1
    assert history.ingest(batch) == 0
    assert history.total_count == len(angles)


@pytest.mark.parametrize(
    "edit, reason",
    [
        ({"timestamp": 100.051}, "time_gap"),
        ({"frame_id": 3}, "frame_gap"),
        ({"sequence": 3}, "sequence_gap"),
        ({"epoch": 1}, "configuration"),
        ({"angle_deg": -90}, "ambiguous_step"),
    ],
)
def test_discontinuities_start_at_actual_raw_angle(edit, reason):
    history = MeasurementHistory()
    batch = rows([0, 40])
    batch[1].update(edit)
    history.ingest(batch)
    assert history.last.break_reason == reason
    assert history.last.continuous_angle == batch[1]["angle_deg"]
    assert history.first.segment != history.last.segment
    assert gap_spans(history.visible(100, 101))[-1][2] == reason


@pytest.mark.parametrize(
    "edit, reason",
    [
        ({"present": False}, "no_contour"),
        ({"present": True, "axis_deg": None}, "ambiguous_axis"),
        ({"axis_deg": 40, "touching_border": True}, "border_contact"),
        ({"present": None}, "invalid"),
    ],
)
def test_invalid_samples_remain_gaps_and_do_not_propagate_phase(edit, reason):
    history = MeasurementHistory()
    batch = rows([80, None, -80])
    batch[1].update(edit)
    history.ingest(batch)
    invalid = history.visible(100, 101)[1]
    assert invalid.reason == reason and invalid.continuous_angle is None
    assert history.last.continuous_angle == -80
    assert history.last.break_reason == "after_invalid"
    assert gap_spans(history.visible(100, 101))[0][2] == reason


def test_overflow_is_counted_once_and_preview_lookup_requires_exact_identity():
    history = MeasurementHistory()
    history.ingest(rows([80]))
    batch = rows([-80], sequence=10, frame_id=10, start=100.01)
    history.ingest(batch, overflow=True)
    assert history.last.break_reason == "buffer_overflow"
    assert history.last.continuous_angle == -80
    history.ingest(batch, overflow=True)
    assert history.overflow_count == 1
    preview = SimpleNamespace(timestamp=100.01, frame_id=10, angle=-80)
    assert history.for_preview(preview) is history.last
    preview.timestamp += 0.0001
    assert history.for_preview(preview) is None


def test_ten_minute_history_preserves_single_frame_peaks_and_prunes_storage():
    history = MeasurementHistory()
    angles = [0] * (142 * 600 + 1)
    angles[42001], angles[42002] = 40, -40
    history.ingest(rows(angles))
    assert len(history) == 85201
    start = time.perf_counter()
    reduced = envelope_points(history.visible(100, 700), 100, 700, 1000)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"Ten-minute plot reduction took {elapsed:.3f}s"
    assert len(reduced) <= 4004
    assert max(row.continuous_angle for row in reduced) == 40
    assert min(row.continuous_angle for row in reduced) == -40
    assert [row.sequence for row in reduced] == sorted(row.sequence for row in reduced)
    assert history.nearest(100).timestamp == 100
    history.ingest(rows([0], start=701, sequence=85202, frame_id=85202))
    assert history.first.timestamp >= 101
    history.expire_before(10000)
    assert len(history) == 0 and not history._by_frame


def test_fifty_ms_is_allowed_and_low_quality_does_not_hide_a_valid_angle():
    history = MeasurementHistory()
    batch = rows([0, 40], dt=0.05)
    batch[1]["confidence"] = 0.01
    history.ingest(batch)
    assert history.last.continuous_angle == 40
    assert history.last.reason is None and history.last.break_reason is None
