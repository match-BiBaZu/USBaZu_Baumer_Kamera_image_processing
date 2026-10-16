import pytest


@pytest.fixture(autouse=True)
def isolated_camera_api_port(monkeypatch):
    # GUI tests start the real API; each window gets a free loopback port.
    monkeypatch.setenv("BIBAZU_VISION_API_PORT", "0")
