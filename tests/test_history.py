"""Time navigation and bounded asynchronous preview storage."""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from bibazu_vision import history as history_module
from bibazu_vision.history import HistorySample, PreviewArchive
from bibazu_vision.history_trace import AngleTrace


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def wait_for(application, predicate, timeout=3):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        application.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    raise AssertionError("Timed out waiting for history operation")


@pytest.fixture
def archives(qt_app):
    instances = []

    def create(**options):
        archive = PreviewArchive(**options)
        instances.append(archive)
        return archive

    yield create
    for archive in instances:
        archive.stop()
        assert archive.wait(3000)
    qt_app.processEvents()


def load(application, archive, key):
    results = []

    def receive(loaded_key, image):
        if key == loaded_key:
            results.append(image)

    archive.image_loaded.connect(receive)
    try:
        archive.request(key)
        wait_for(application, lambda: bool(results))
        return results[-1]
    finally:
        archive.image_loaded.disconnect(receive)


def test_trace_click_uses_time_not_frame_spacing_and_pins_selection(qt_app):
    trace = AngleTrace()
    try:
        trace.reset(100)
        trace.resize(800, 160)
        trace.show()
        qt_app.processEvents()
        samples = [
            trace.add(100 + seconds, index, angle)
            for index, (seconds, angle) in enumerate(((0, 1), (0.2, None), (8, 35)))
        ]
        selected = []
        trace.selected.connect(selected.append)
        rect = trace.plot.plot_rect()
        position = QPoint(round(rect.left() + 0.21 / 10 * rect.width()), round(rect.center().y()))
        QTest.mouseClick(trace.plot, Qt.MouseButton.LeftButton, pos=position)
        assert selected == [samples[1]]
        assert selected[0].angle is None
        assert not trace.follow_live
        trace.add(125, 4, -4)
        assert trace.selection is samples[1]
        assert trace.view_start == 0
        trace.live()
        assert trace.selection is None and trace.follow_live
        assert trace.view_start == 15
        trace.set_span(1)
        assert trace.view_start == 24
        # A wheel notch moves the viewport back by a fifth of its duration.
        wheel = QWheelEvent(
            QPointF(100, 50),
            QPointF(100, 50),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(trace.plot, wheel)
        assert trace.view_start == pytest.approx(23.8)
        trace.add(126, 5, 5)
        assert trace.view_start == pytest.approx(23.8)
        trace.scrollbar.setValue(trace.scrollbar.maximum())
        assert trace.view_start == 25
        trace.set_span(600)
        assert trace.span == 600
        # Off-plot clicks cannot select a historical frame.
        before = trace.selection
        QTest.mouseClick(trace.plot, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
        assert trace.selection is before
        assert trace.grab().width() == 800  # Exercise paint with gaps and axis labels.
    finally:
        trace.close()


def test_trace_retention_expires_selection_and_keeps_exact_boundary(qt_app):
    trace = AngleTrace()
    trace.reset(10)
    first = trace.add(10, 1, 0)
    trace.select_at(0)
    expired = []
    trace.selection_expired.connect(lambda: expired.append(True))
    trace.add(610, 2, 2)
    assert first in trace.samples
    trace.add(610.01, 3, 3)
    assert first not in trace.samples
    assert expired == [True]
    assert trace.selection is None
    trace.expire_before(1300)
    assert not trace.samples
    assert trace.scrollbar.maximum() == 0


def test_archive_compresses_off_thread_loads_matching_image_and_cleans_up(
    qt_app, archives, monkeypatch
):
    archive = archives()
    encode_threads = []
    original = archive.encode

    def encode(image):
        encode_threads.append(threading.get_ident())
        return original(image)

    monkeypatch.setattr(archive, "encode", encode)
    archive.start()
    assert archive.submit(HistorySample(0, 100, 8, 3), np.full((1200, 1600), 70, np.uint8))
    image = load(qt_app, archive, 0)
    assert image.shape == (720, 960, 3)
    assert image.mean() == pytest.approx(70, abs=1)
    assert encode_threads and encode_threads[0] != threading.get_ident()
    directory = archive.directory
    assert directory.exists()
    archive.stop()
    assert archive.wait(3000)
    assert not directory.exists()


def test_archive_queue_limits_and_dropped_images(qt_app, archives):
    archive = archives(queue_depth=1, queue_bytes=100)
    small = np.full((8, 8), 20, np.uint8)
    assert archive.submit(HistorySample(0, 1, 1, 0), small)
    assert not archive.submit(HistorySample(1, 2, 2, 1), small)
    archive.start()
    assert load(qt_app, archive, 1) is None
    assert load(qt_app, archive, 0) is not None
    assert not archive.submit(HistorySample(2, 3, 3, 2), np.zeros((11, 11), np.uint8))


def test_archive_expiry_quota_missing_and_corrupt_files(qt_app, archives):
    image = np.full((40, 60), 80, np.uint8)
    archive = archives(max_bytes=len(PreviewArchive.encode(image)) + 10)
    archive.start()
    assert archive.submit(HistorySample(0, 100, 1, 0), image)
    assert load(qt_app, archive, 0) is not None
    assert archive.submit(HistorySample(1, 101, 2, 3), image)
    assert load(qt_app, archive, 1) is not None
    assert load(qt_app, archive, 0) is None
    assert sum(path.stat().st_size for path in archive.directory.glob("*.jpg")) <= archive.max_bytes
    (archive.directory / "1.jpg").write_bytes(b"bad jpeg")
    assert load(qt_app, archive, 1) is None
    archive.expire_before(102)
    assert load(qt_app, archive, 1) is None
    assert not list(archive.directory.glob("*.jpg"))


def test_archive_write_failure_does_not_stop_later_images(qt_app, archives, monkeypatch):
    archive = archives()
    archive.start()
    image = np.full((20, 30), 45, np.uint8)

    def fail(image):
        raise OSError("Disk full")

    with monkeypatch.context() as patch:
        patch.setattr(archive, "encode", fail)
        assert archive.submit(HistorySample(0, 1, 1, 0), image)
        assert load(qt_app, archive, 0) is None
    assert archive.submit(HistorySample(1, 2, 2, 2), image)
    assert load(qt_app, archive, 1).mean() == pytest.approx(45, abs=1)
    assert archive.isRunning()


def test_archive_expiry_handles_prioritized_writes(qt_app, archives):
    archive = archives()
    image = np.full((20, 30), 60, np.uint8)
    assert archive.submit(HistorySample(0, 100, 1, 0), image)
    assert archive.submit(HistorySample(1, 200, 2, 0), image)
    archive.request(1)
    archive.start()
    assert load(qt_app, archive, 1) is not None
    assert load(qt_app, archive, 0) is not None
    archive.expire_before(150)
    assert load(qt_app, archive, 0) is None
    assert load(qt_app, archive, 1) is not None


def test_archive_unavailable_temp_volume_keeps_thread_responsive(qt_app, archives, monkeypatch):
    def fail(**options):
        raise OSError("Temp volume unavailable")

    monkeypatch.setattr(history_module, "TemporaryDirectory", fail)
    archive = archives()
    archive.start()
    assert archive.submit(HistorySample(0, 1, 1, None), np.zeros((20, 30), np.uint8))
    assert load(qt_app, archive, 0) is None
    assert archive.isRunning()
