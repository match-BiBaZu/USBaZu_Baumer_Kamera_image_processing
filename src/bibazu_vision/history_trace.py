"""Time-based angle navigation drawn with Qt, without a plotting dependency."""

from __future__ import annotations

import math
from collections import deque

from PyQt6.QtCore import QPointF, QRectF, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QScrollBar, QVBoxLayout, QWidget

from .angle_history import REASONS, MeasurementHistory, envelope_points, gap_spans
from .history import HistorySample
from .i18n import translate


class AngleTrace(QWidget):
    selected = pyqtSignal(object)
    selection_expired = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.samples = deque()
        self.measurements = MeasurementHistory()
        self.max_gap_seconds = 0.05
        self.preview_times = deque()
        self.preview_count = 0
        self.language = "de"
        self.origin = 0.0
        self.span = 10.0
        self.view_start = 0.0
        self.follow_live = True
        self.selection = None
        self._next_key = 0
        self._caption = "Fortlaufender Achswinkel · Lücken markiert"
        self.plot = TracePlot(self)
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.valueChanged.connect(self._scroll_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.plot)
        layout.addWidget(self.scrollbar)

    @property
    def caption(self):
        return self._caption

    @caption.setter
    def caption(self, value):
        self._caption = value
        self.plot.update()

    def reset(self, origin):
        self.samples.clear()
        self.measurements = MeasurementHistory(self.max_gap_seconds)
        self.preview_times.clear()
        self.preview_count = 0
        self.origin = origin
        self.selection = None
        self.follow_live = True
        self.view_start = 0.0
        self._next_key = 0
        self._sync()

    def add(self, timestamp, frame_id, angle):
        """Register a selectable preview frame, independently of the dense curve."""
        sample = HistorySample(self._next_key, timestamp, frame_id, angle)
        self._next_key += 1
        self.samples.append(sample)
        self.preview_times.append(timestamp)
        self.preview_count += 1
        self.expire_before(timestamp - 600)
        return sample

    def add_measurements(self, batch):
        added = self.measurements.ingest(batch["measurements"], overflow=batch["overflow"])
        if added:
            self._sync()
        return added

    def rates(self, now):
        while self.preview_times and self.preview_times[0] < now - 1:
            self.preview_times.popleft()
        rows = self.measurements.visible(now - 1, now)
        valid = sum(row.reason is None for row in rows)
        return len(rows), len(self.preview_times), valid / len(rows) if rows else 0

    def expire_before(self, cutoff):
        while self.samples and self.samples[0].timestamp < cutoff:
            self.samples.popleft()
        self.measurements.expire_before(cutoff)
        expired = self.selection is not None and self.selection.timestamp < cutoff
        if expired:
            self.selection = None
        self._sync()
        if expired:
            self.selection_expired.emit()

    def set_span(self, seconds):
        self.span = float(seconds)
        self._sync()

    def live(self):
        self.selection = None
        self.follow_live = True
        self._sync()

    def _sync(self):
        oldest = self.samples[0].timestamp - self.origin if self.samples else 0.0
        latest = self.samples[-1].timestamp - self.origin if self.samples else 0.0
        if self.measurements.first is not None:
            first = self.measurements.first.timestamp - self.origin
            last = self.measurements.last.timestamp - self.origin
            oldest = min(oldest, first) if self.samples else first
            latest = max(latest, last)
        end = max(oldest, latest - self.span)
        self.view_start = end if self.follow_live else min(end, max(oldest, self.view_start))
        with QSignalBlocker(self.scrollbar):
            self.scrollbar.setRange(round(oldest * 1000), round(end * 1000))
            self.scrollbar.setPageStep(round(self.span * 1000))
            self.scrollbar.setSingleStep(max(1, round(self.span * 100)))
            self.scrollbar.setValue(round(self.view_start * 1000))
        self.plot.invalidate()
        self.plot.update()

    def _scroll_changed(self, milliseconds):
        self.follow_live = False
        self.view_start = milliseconds / 1000
        self.plot.invalidate()
        self.plot.update()

    def scroll_by(self, seconds):
        self.follow_live = False
        self.scrollbar.setValue(round((self.view_start + seconds) * 1000))

    def select_at(self, seconds):
        visible = [
            sample
            for sample in self.samples
            if self.view_start <= sample.timestamp - self.origin <= self.view_start + self.span
        ]
        if not visible:
            return
        self.selection = min(
            visible, key=lambda sample: abs(sample.timestamp - self.origin - seconds)
        )
        self.follow_live = False
        self.plot.update()
        self.selected.emit(self.selection)


class TracePlot(QWidget):
    def __init__(self, trace):
        super().__init__(trace)
        self.setMinimumHeight(135)
        self.setMaximumHeight(160)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self._cache = None

    @property
    def trace(self):
        return self.parent()

    def plot_rect(self):
        return QRectF(64, 26, max(1, self.width() - 78), max(1, self.height() - 50))

    def invalidate(self):
        self._cache = None

    def view_data(self):
        trace = self.trace
        start = trace.origin + trace.view_start
        end = start + trace.span
        key = (
            trace.measurements.cursor,
            trace.view_start,
            trace.span,
            self.width(),
            len(trace.measurements),
        )
        if self._cache is None or self._cache[0] != key:
            rows = trace.measurements.visible(start, end)
            points = envelope_points(rows, start, end, self.plot_rect().width())
            self._cache = (key, points, gap_spans(rows))
        return self._cache[1:]

    @staticmethod
    def axis_limits(points):
        values = [row.continuous_angle for row in points]
        if not values:
            return -90, 90, 90
        low, high = min(values), max(values)
        margin = max(10, (high - low) * 0.1)
        step = 45 if high - low <= 180 else 90 * math.ceil((high - low) / 360)
        return (
            math.floor((low - margin) / step) * step,
            math.ceil((high + margin) / step) * step,
            step,
        )

    def mouseMoveEvent(self, event):
        rect = self.plot_rect()
        if not rect.contains(event.position()):
            self.setToolTip("")
            return
        trace = self.trace
        timestamp = (
            trace.origin
            + trace.view_start
            + ((event.position().x() - rect.left()) / rect.width() * trace.span)
        )
        row = trace.measurements.nearest(timestamp)
        if row is None:
            self.setToolTip("")
            return
        language = trace.language
        text = translate(
            "t = {seconds:.3f} s · Messwinkel {raw} · Fortlaufend {angle}",
            language,
            seconds=row.timestamp - trace.origin,
            raw="—" if row.raw_angle is None else f"{row.raw_angle:+.2f}°",
            angle="—" if row.continuous_angle is None else f"{row.continuous_angle:+.2f}°",
        )
        for reason in (row.reason, row.break_reason):
            if reason is not None:
                text += "\n" + (
                    translate(
                        "Zeitlücke über {milliseconds:g} ms", language,
                        milliseconds=row.gap_limit_ms,
                    ) if reason == "time_gap" else translate(REASONS[reason], language)
                )
        text += "\n" + translate("Klick rastet zum nächsten Vorschauframe ein.", language)
        self.setToolTip(text)

    def mousePressEvent(self, event):
        rect = self.plot_rect()
        if event.button() == Qt.MouseButton.LeftButton and rect.contains(event.position()):
            seconds = self.trace.view_start + (
                (event.position().x() - rect.left()) / rect.width() * self.trace.span
            )
            self.trace.select_at(seconds)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta:
            self.trace.scroll_by(-delta / 120 * self.trace.span / 5)
        elif not event.pixelDelta().isNull():
            self.trace.scroll_by(-event.pixelDelta().x() / self.width() * self.trace.span)
        event.accept()

    def paintEvent(self, event):
        trace = self.trace
        points, gaps = self.view_data()
        low, high, step = self.axis_limits(points)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111c30"))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(8, 17, trace.caption)
        rect = self.plot_rect()
        for angle in range(low, high + 1, step):
            y = rect.bottom() - (angle - low) / (high - low) * rect.height()
            painter.setPen(QColor("#334155"))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(QRectF(0, y - 9, 59, 18), Qt.AlignmentFlag.AlignRight, f"{angle}°")
        for tick in range(5):
            seconds = trace.view_start + tick * trace.span / 4
            x = rect.left() + rect.width() * tick / 4
            label_rect = QRectF(x - 40, rect.bottom() + 4, 80, 18)
            # Keep the edge labels inside the widget.
            label_rect.moveLeft(min(self.width() - 80, max(0, label_rect.left())))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, f"{seconds:.2f} s")
        painter.save()
        painter.setClipRect(rect.adjusted(-2, -2, 2, 2))
        start = trace.origin + trace.view_start
        for left, right, reason in gaps:
            x = rect.left() + (left - start) / trace.span * rect.width()
            width = max(2, (right - left) / trace.span * rect.width())
            color = (
                QColor(251, 191, 36, 35)
                if reason
                in {"no_contour", "ambiguous_axis", "border_contact", "invalid", "after_invalid"}
                else QColor(148, 163, 184, 55)
            )
            painter.fillRect(QRectF(x, rect.top(), width, rect.height()), color)
        painter.setPen(QPen(QColor("#38bdf8"), 2))
        previous = None
        for sample in points:
            point = QPointF(
                rect.left() + (sample.timestamp - start) / trace.span * rect.width(),
                rect.bottom() - (sample.continuous_angle - low) / (high - low) * rect.height(),
            )
            if previous is not None and sample.segment == previous[1]:
                painter.drawLine(previous[0], point)
            painter.drawPoint(point)
            previous = (point, sample.segment)
        if trace.selection is not None:
            seconds = trace.selection.timestamp - trace.origin
            x = rect.left() + (seconds - trace.view_start) / trace.span * rect.width()
            painter.setPen(QPen(QColor("#fbbf24"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        painter.restore()
