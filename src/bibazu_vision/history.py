"""Bounded temporary preview archive, independent of acquisition and clip recording."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


@dataclass(frozen=True)
class HistorySample:
    key: int
    timestamp: float
    frame_id: int | str
    angle: float | None


class _PreviewFiles:
    """Thread-owned disk index: look up by key and evict by timestamp.

    A heap also handles preview requests that encode queued frames out of order.
    Cleanup only visits removed files, never the entire ten-minute history.
    Failed deletions remain accounted for and do not block other old files.
    """

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self.files = {}
        self._age = []

    def add(self, sample, path, size):
        previous = self.files.get(sample.key)
        if previous is not None:
            self.used_bytes -= previous[2]
        self.files[sample.key] = (sample.timestamp, path, size)
        self.used_bytes += size
        heappush(self._age, (sample.timestamp, sample.key))

    def _oldest(self):
        while self._age:
            timestamp, key = self._age[0]
            entry = self.files.get(key)
            if entry is not None and entry[0] == timestamp:
                return timestamp, key
            heappop(self._age)
        return None

    def prune(self, cutoff, *, required_bytes=0, limit=8):
        """Bound disk work per pass; return whether age cleanup is still pending."""
        failed = []
        for _ in range(limit):
            oldest = self._oldest()
            if oldest is None:
                break
            timestamp, key = oldest
            if timestamp >= cutoff and self.used_bytes + required_bytes <= self.max_bytes:
                break
            heappop(self._age)
            _, path, size = self.files[key]
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failed.append(oldest)
                continue
            del self.files[key]
            self.used_bytes -= size
        for oldest in failed:
            heappush(self._age, oldest)
        oldest = self._oldest()
        return oldest is not None and oldest[0] < cutoff

    def read(self, key, cutoff):
        entry = self.files.get(key)
        # Expired images are unavailable even while their disk cleanup is pending.
        if entry is None or entry[0] < cutoff:
            return None
        return entry[1].read_bytes()


class PreviewArchive(QThread):
    """Only this thread owns files. Submitted arrays must not be mutated afterwards.

    Writes are bounded by count and uncompressed bytes. Reads coalesce to the
    latest request and take priority over writes. Missing/failed images return
    None, never an image from a different sample.
    """

    image_loaded = pyqtSignal(int, object)
    image_stored = pyqtSignal(int, bool)

    def __init__(
        self, parent=None, *, max_bytes=2 * 1024**3, queue_depth=3, queue_bytes=32 * 1024**2
    ):
        super().__init__(parent)
        self.max_bytes = max_bytes
        self.queue_depth = queue_depth
        self.queue_bytes = queue_bytes
        self.directory: Path | None = None
        self._condition = threading.Condition()
        self._writes = deque()
        self._queued_bytes = 0
        self._request = None
        self._cutoff = float("-inf")
        self._stopping = False

    def submit(self, sample: HistorySample, image: np.ndarray) -> bool:
        with self._condition:
            if (
                self._stopping
                or len(self._writes) >= self.queue_depth
                or self._queued_bytes + image.nbytes > self.queue_bytes
            ):
                return False
            self._writes.append((sample, image))
            self._queued_bytes += image.nbytes
            self._condition.notify()
            return True

    def request(self, key: int):
        with self._condition:
            self._request = key
            self._condition.notify()

    def expire_before(self, timestamp: float):
        with self._condition:
            if timestamp <= self._cutoff:
                return
            self._cutoff = timestamp
            self._condition.notify()

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify()

    @staticmethod
    def encode(image):
        height, width = image.shape[:2]
        scale = min(1.0, 960 / max(height, width))
        if scale < 1:
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise OSError("Preview JPEG encoding failed")
        return data.tobytes()

    def run(self):
        files = _PreviewFiles(self.max_bytes)
        applied_cutoff = float("-inf")
        cleanup_pending = False
        temporary = None
        try:
            try:
                temporary = TemporaryDirectory(prefix="usbazu-preview-")
                self.directory = Path(temporary.name)
            except OSError:
                # Angles remain usable even if the temp volume is unavailable.
                pass
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda applied_cutoff=applied_cutoff: (
                            self._stopping
                            or self._writes
                            or self._request is not None
                            or self._cutoff != applied_cutoff
                        ),
                        timeout=0.05 if cleanup_pending else None,
                    )
                    if self._stopping:
                        self._writes.clear()
                        self._queued_bytes = 0
                        break
                    cutoff = self._cutoff
                    request = self._request
                    self._request = None
                    # Drop expired pending images without doing unnecessary IO.
                    while self._writes and self._writes[0][0].timestamp < cutoff:
                        _, image = self._writes.popleft()
                        self._queued_bytes -= image.nbytes
                    job = None
                    if request is not None:
                        for i, pending in enumerate(self._writes):
                            if pending[0].key == request:
                                job = self._writes[i]
                                del self._writes[i]
                                break
                    elif self._writes:
                        job = self._writes.popleft()
                    if job is not None:
                        self._queued_bytes -= job[1].nbytes

                data = None
                if job is not None and self.directory is not None:
                    try:
                        data = self.encode(job[1])
                    except (OSError, cv2.error):
                        pass
                # A large expiry backlog must not monopolize this thread or
                # flood the disk. Continue in small batches, even without frames.
                required = len(data) if data is not None and len(data) <= self.max_bytes else 0
                cleanup_pending = files.prune(cutoff, required_bytes=required)
                applied_cutoff = cutoff

                if job is not None:
                    sample, image = job
                    saved = False
                    if self.directory is not None:
                        path = self.directory / f"{sample.key}.jpg"
                        try:
                            if data is not None and files.used_bytes + len(data) <= self.max_bytes:
                                path.write_bytes(data)
                                files.add(sample, path, len(data))
                                saved = True
                        except (OSError, cv2.error):
                            # A failed write must not leave partial unaccounted files.
                            try:
                                path.unlink(missing_ok=True)
                            except OSError:
                                pass
                    self.image_stored.emit(sample.key, saved)
                    del image, job

                if request is not None:
                    image = None
                    try:
                        encoded = files.read(request, cutoff)
                        if encoded is not None:
                            data = np.frombuffer(encoded, dtype=np.uint8)
                            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    except (OSError, cv2.error):
                        pass
                    self.image_loaded.emit(request, image)
        finally:
            if temporary is not None:
                try:
                    temporary.cleanup()
                except OSError:
                    # Antivirus/file locks must not abort the application on exit.
                    pass
