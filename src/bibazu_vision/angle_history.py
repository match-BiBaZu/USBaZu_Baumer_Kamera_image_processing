"""Full-rate measurement history. Continuous angles are display-only estimates."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np

REASONS = {
    "no_contour": "Kein geeigneter Umriss",
    "ambiguous_axis": "Uneindeutige Achse",
    "border_contact": "Randkontakt",
    "invalid": "Ungültiger Messwert",
    "after_invalid": "Fortführung nach ungültigem Messwert neu angesetzt",
    "frame_gap": "Frame-ID-Unterbrechung",
    "buffer_overflow": "Messpuffer übergelaufen",
    "sequence_gap": "Messwerte fehlen",
    "time_gap": "Zeitlücke",
    "configuration": "Messkonfiguration geändert",
    "ambiguous_step": "Winkelschritt um 90° uneindeutig",
}


@dataclass(frozen=True, slots=True)
class AngleMeasurement:
    sequence: int
    timestamp: float
    frame_id: int | str | None
    raw_angle: float | None
    continuous_angle: float | None
    reason: str | None
    break_reason: str | None
    segment: int
    epoch: int
    gap_limit_ms: float = 50.0


class MeasurementHistory:
    """Append in host-time order; binary-search visible intervals and preview matches.

    Prefix compaction is amortized, so a ten-minute history is not copied on
    every new frame. At most 4095 expired records await the next compaction.
    """

    def __init__(self, max_gap_seconds=0.05):
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("time gap limit must be positive and finite")
        self.max_gap_seconds = float(max_gap_seconds)
        self._items = []
        self._times = []
        self._head = 0
        self._by_frame = {}
        self._previous = None
        self.cursor = 0
        self.total_count = 0
        self.overflow_count = 0

    def __len__(self):
        return len(self._items) - self._head

    @property
    def first(self):
        return self._items[self._head] if len(self) else None

    @property
    def last(self):
        return self._items[-1] if len(self) else None

    @staticmethod
    def invalid_reason(row, raw):
        if raw is not None and row.get("valid", True):
            return None
        if row.get("present") is False:
            return "no_contour"
        if row.get("present") and row.get("axis_deg") is None:
            return "ambiguous_axis"
        if row.get("touching_border"):
            return "border_contact"
        return "invalid"

    def ingest(self, rows, *, overflow=False):
        added = 0
        for row in rows:
            sequence = row["sequence"]
            if sequence <= self.cursor:
                continue
            timestamp = row["timestamp"]
            frame_id = row.get("frame_id")
            epoch = row.get("epoch", 0)
            raw = row.get("angle_deg")
            raw = float(raw) if raw is not None and math.isfinite(raw) else None
            reason = self.invalid_reason(row, raw)
            previous = self._previous
            boundary = None
            if overflow and added == 0:
                boundary = "buffer_overflow"
                self.overflow_count += 1
            elif previous is not None:
                if epoch != previous.epoch:
                    boundary = "configuration"
                elif sequence != previous.sequence + 1:
                    boundary = "sequence_gap"
                elif (
                    isinstance(frame_id, int)
                    and isinstance(previous.frame_id, int)
                    and frame_id != previous.frame_id + 1
                ):
                    boundary = "frame_gap"
                elif timestamp - previous.timestamp > self.max_gap_seconds + 1e-9:
                    boundary = "time_gap"
                elif previous.continuous_angle is None:
                    boundary = "after_invalid"
            segment = previous.segment if previous is not None else 0
            continuous = None
            if reason is None:
                continuous = raw
                if previous is not None and previous.continuous_angle is not None and not boundary:
                    delta = ((raw - previous.raw_angle + 90) % 180) - 90
                    if math.isclose(abs(delta), 90, abs_tol=1e-9):
                        boundary = "ambiguous_step"
                    else:
                        continuous = previous.continuous_angle + delta
            if boundary or (
                reason is not None
                and previous is not None
                and previous.continuous_angle is not None
            ):
                segment += 1
            measurement = AngleMeasurement(
                sequence, timestamp, frame_id, raw, continuous, reason, boundary, segment, epoch,
                self.max_gap_seconds * 1000,
            )
            self._items.append(measurement)
            self._times.append(timestamp)
            self._by_frame[timestamp, frame_id] = measurement
            self._previous = measurement
            self.cursor = sequence
            self.total_count += 1
            added += 1
        if self.last is not None:
            self.expire_before(self.last.timestamp - 600)
        return added

    def expire_before(self, timestamp):
        end = bisect_left(self._times, timestamp, lo=self._head)
        for i in range(self._head, end):
            row = self._items[i]
            self._by_frame.pop((row.timestamp, row.frame_id), None)
        self._head = end
        if self._head >= 4096 or self._head == len(self._items):
            self._items = self._items[self._head :]
            self._times = self._times[self._head :]
            self._head = 0

    def visible(self, start, end):
        left = bisect_left(self._times, start, lo=self._head)
        right = bisect_right(self._times, end, lo=left)
        return self._items[left:right]

    def nearest(self, timestamp):
        index = bisect_left(self._times, timestamp, lo=self._head)
        candidates = self._items[max(self._head, index - 1) : index + 1]
        return (
            min(candidates, key=lambda row: abs(row.timestamp - timestamp)) if candidates else None
        )

    def for_preview(self, sample):
        row = self._by_frame.get((sample.timestamp, sample.frame_id))
        return row if row is not None and row.raw_angle == sample.angle else None


def envelope_points(rows, start, end, pixels):
    """Keep actual first/min/max/last samples per pixel and segment, in time order."""
    if len(rows) > 4096:
        return _dense_envelope(rows, start, end, pixels)
    result = []
    group_key = None
    first = last = low = high = None

    def flush():
        if first is not None:
            unique = {row.sequence: row for row in (first, low, high, last)}
            result.extend(unique[key] for key in sorted(unique))

    factor = max(1, pixels) / max(end - start, 1e-12)
    for row in rows:
        if row.continuous_angle is None:
            continue
        key = (int((row.timestamp - start) * factor), row.segment)
        if key != group_key:
            flush()
            first = last = low = high = row
            group_key = key
        else:
            last = row
            if row.continuous_angle < low.continuous_angle:
                low = row
            if row.continuous_angle > high.continuous_angle:
                high = row
    flush()
    return result


def _dense_envelope(rows, start, end, pixels):
    # Vectorize full ten-minute views; avoid tens of thousands of Python
    # bucket operations on the same thread that presents the live image.
    count = len(rows)
    values = np.fromiter(
        (row.continuous_angle if row.continuous_angle is not None else math.nan for row in rows),
        dtype=float,
        count=count,
    )
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return []
    times = np.fromiter((row.timestamp for row in rows), dtype=float, count=count)[valid]
    segments = np.fromiter((row.segment for row in rows), dtype=np.int64, count=count)[valid]
    values = values[valid]
    columns = ((times - start) * max(1, pixels) / max(end - start, 1e-12)).astype(np.int64)
    starts = np.r_[
        0, 1 + np.flatnonzero((columns[1:] != columns[:-1]) | (segments[1:] != segments[:-1]))
    ]
    ends = np.r_[starts[1:], len(valid)]
    counts = ends - starts
    indices = np.arange(len(valid))
    minimum = np.minimum.reduceat(values, starts)
    maximum = np.maximum.reduceat(values, starts)
    low = np.minimum.reduceat(
        np.where(values == np.repeat(minimum, counts), indices, count), starts
    )
    high = np.minimum.reduceat(
        np.where(values == np.repeat(maximum, counts), indices, count), starts
    )
    chosen = np.unique(np.r_[starts, ends - 1, low, high])
    return [rows[valid[index]] for index in chosen]


def gap_spans(rows):
    """Group invalid runs; include the unobserved interval before a discontinuity."""
    spans = []
    previous = None
    for row in rows:
        if row.reason is not None:
            if spans and spans[-1][2] == row.reason and previous is not None and previous.reason:
                spans[-1] = (spans[-1][0], row.timestamp, row.reason)
            else:
                spans.append((row.timestamp, row.timestamp, row.reason))
        elif row.break_reason and previous is not None:
            spans.append((previous.timestamp, row.timestamp, row.break_reason))
        previous = row
    return spans
