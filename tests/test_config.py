from server.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.udp_host == "0.0.0.0"
    assert settings.udp_port == 5005
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.num_subcarriers == 56
    assert settings.csi_sample_rate == 20
    assert settings.num_joints == 24


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("UDP_PORT", "6000")
    settings = Settings()
    assert settings.udp_port == 6000
