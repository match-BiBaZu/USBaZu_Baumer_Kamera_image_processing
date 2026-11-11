"""Process-wide OpenCV setup, called before acquisition or GUI worker threads start."""

from __future__ import annotations

import os

import cv2

DEFAULT_OPENCV_THREADS = 4


def configure_opencv_threads(threads: int | None = None) -> int:
    """Use a small thread budget; call once at application startup, never per frame.

    CLI overrides the environment. Explicit positive counts avoid OpenCV's special
    zero/negative settings and let other PCs be tuned without changing code.
    """
    if threads is None:
        try:
            threads = int(os.environ.get("BIBAZU_OPENCV_THREADS", DEFAULT_OPENCV_THREADS))
        except ValueError as exc:
            raise ValueError("OpenCV-Rechenthreads müssen eine positive ganze Zahl sein.") from exc
    if type(threads) is not int or threads < 1:
        raise ValueError("OpenCV-Rechenthreads müssen eine positive ganze Zahl sein.")
    cv2.setNumThreads(threads)
    return cv2.getNumThreads()
