"""Deterministic synthetic chute for UI and pipeline checks without hardware."""

import math
import time

import cv2
import numpy as np

from .camera import CameraFrame


class DemoCamera:
    def __init__(self, config):
        self.config = config
        self.frame_id = 0
        self.started = 0.0
        self.deadline = 0.0
        self.diagnostics = {"model": "Synthetische Demo (keine Kamera)", "fps": 120}

    def open(self):
        self.started = time.perf_counter()
        self.deadline = self.started

    @staticmethod
    def background():
        image = np.full((480, 800), 185, np.uint8)
        cv2.line(image, (0, 390), (799, 350), 90, 3)
        cv2.line(image, (0, 410), (799, 370), 130, 2)
        return image

    def read(self, timeout=0.5):
        now = time.perf_counter()
        if now < self.deadline:
            time.sleep(min(self.deadline - now, timeout))
        now = time.perf_counter()
        self.deadline = now + 1 / 120
        elapsed = now - self.started
        phase = elapsed % 6
        image = self.background()
        active = 1.0 <= phase < 3.4
        if active:
            progress = (phase - 1) / 2.4
            x = -110 + progress * 1030
            angle = 2.86 + 36 * math.sin(progress * math.pi)
            center = (x, 340 - x * 0.05 - 38)
            points = cv2.boxPoints((center, (170, 54), -angle)).astype(np.int32)
            cv2.fillConvexPoly(image, points, 38)
            cv2.circle(image, (int(x), int(center[1])), 10, 85, -1)
        self.frame_id += 1
        previous = (elapsed - 1 / 120) % 6
        triggered = previous < 1 <= phase
        return CameraFrame(image, now, self.frame_id, triggered)

    def close(self):
        pass
