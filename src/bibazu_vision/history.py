"""Bounded temporary preview archive, independent of acquisition and clip recording."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
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
            self._cutoff = max(self._cutoff, timestamp)
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
        files = {}
        used_bytes = 0
        applied_cutoff = float("-inf")
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
                        )
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

                for key, (timestamp, path, size) in list(files.items()):
                    if timestamp >= cutoff:
                        continue
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        # Keep accounting for files that could not be deleted.
                        continue
                    del files[key]
                    used_bytes -= size
                applied_cutoff = cutoff

                if job is not None:
                    sample, image = job
                    saved = False
                    if self.directory is not None:
                        path = self.directory / f"{sample.key}.jpg"
                        try:
                            data = self.encode(image)
                            while files and used_bytes + len(data) > self.max_bytes:
                                key = min(files)
                                _, oldest, size = files[key]
                                oldest.unlink(missing_ok=True)
                                del files[key]
                                used_bytes -= size
                            if len(data) <= self.max_bytes:
                                path.write_bytes(data)
                                files[sample.key] = (sample.timestamp, path, len(data))
                                used_bytes += len(data)
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
                    if request in files:
                        try:
                            data = np.frombuffer(files[request][1].read_bytes(), dtype=np.uint8)
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
