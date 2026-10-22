import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def qt_application_lifetime():
    # Qt requires one QApplication per process. Keep a strong reference across
    # API and GUI test modules instead of destroying/recreating the native app
    # while deferred widget deletions from earlier tests may still be pending.
    from PyQt6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


@pytest.fixture(autouse=True)
def isolated_camera_api_port(monkeypatch):
    # GUI tests start the real API; each window gets a free loopback port.
    monkeypatch.setenv("BIBAZU_VISION_API_PORT", "0")
