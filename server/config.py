from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from pydantic_settings import BaseSettings
from pydantic import ConfigDict


# ======================================================================
# Hardware profiles — each defines CSI parameters + matched model weights
# ======================================================================


@dataclass(frozen=True)
class HardwareProfile:
    """CSI hardware configuration + matched model weights path."""

    id: str
    name: str
    description: str

    # CSI parameters (determine model input_dim)
    num_subcarriers: int
    max_nodes: int
    csi_sample_rate: int  # Hz per node
    frequency_ghz: float  # 2.4 or 5.0
    bandwidth_mhz: int  # 20, 40, 80

    # Model weights trained for this profile
    model_path: str

    # Training dataset this profile was calibrated on
    dataset: str = ""


# Built-in profiles
HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "esp32s3": HardwareProfile(
        id="esp32s3",
        name="ESP32-S3 (Default)",
        description="ESP32-S3 with LWIP CSI, 56 subcarriers, 2.4 GHz, 3x3 MIMO",
        num_subcarriers=56,
        max_nodes=6,
        csi_sample_rate=20,
        frequency_ghz=2.4,
        bandwidth_mhz=20,
        model_path="models/esp32s3/pose_model.pth",
        dataset="synthetic",
    ),
    "esp32s3_mmfi": HardwareProfile(
        id="esp32s3_mmfi",
        name="ESP32-S3 + MM-Fi",
        description="ESP32-S3 profile trained on MM-Fi dataset (NeurIPS 2023)",
        num_subcarriers=56,
        max_nodes=4,
        csi_sample_rate=20,
        frequency_ghz=2.4,
        bandwidth_mhz=20,
        model_path="models/esp32s3_mmfi/pose_model.pth",
        dataset="mmfi",
    ),
    "tplink_n750": HardwareProfile(
        id="tplink_n750",
        name="TP-Link N750 (MM-Fi)",
        description="TP-Link N750 AP with Atheros CSI Tool — MM-Fi dataset hardware",
        num_subcarriers=114,
        max_nodes=2,
        csi_sample_rate=100,
        frequency_ghz=5.0,
        bandwidth_mhz=40,
        model_path="models/tplink_n750/pose_model.pth",
        dataset="mmfi",
    ),
    "intel5300": HardwareProfile(
        id="intel5300",
        name="Intel 5300 NIC",
        description="Intel 5300 with Linux CSI Tool — classic WiFi sensing setup",
        num_subcarriers=30,
        max_nodes=3,
        csi_sample_rate=100,
        frequency_ghz=5.0,
        bandwidth_mhz=20,
        model_path="models/intel5300/pose_model.pth",
        dataset="wipose",
    ),
    "esp32c6_wifi6": HardwareProfile(
        id="esp32c6_wifi6",
        name="ESP32-C6 (WiFi 6)",
        description="ESP32-C6 RISC-V with WiFi 6 (802.11ax) CSI support",
        num_subcarriers=64,
        max_nodes=4,
        csi_sample_rate=50,
        frequency_ghz=2.4,
        bandwidth_mhz=20,
        model_path="models/esp32c6/pose_model.pth",
        dataset="",
    ),
}

DEFAULT_PROFILE = "esp32s3"


class Settings(BaseSettings):
    model_config = ConfigDict(env_prefix="", env_file=".env")

    # UDP receiver
    udp_host: str = "0.0.0.0"
    udp_port: int = 5005

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Hardware profile (selects CSI params + model)
    hardware_profile: str = DEFAULT_PROFILE

    # CSI parameters (overridden by hardware profile if set)
    num_subcarriers: int = 56
    csi_sample_rate: int = 20  # Hz per node
    max_nodes: int = 6

    # Model (overridden by hardware profile if set)
    num_joints: int = 24
    model_path: str = "models/pose_model.pth"

    # Fall detection
    fall_threshold: float = 0.8
    fall_alert_cooldown: int = 30  # seconds

    def apply_hardware_profile(self) -> HardwareProfile | None:
        """Apply hardware profile settings, returning the profile used."""
        profile = HARDWARE_PROFILES.get(self.hardware_profile)
        if profile is None:
            return None
        self.num_subcarriers = profile.num_subcarriers
        self.max_nodes = profile.max_nodes
        self.csi_sample_rate = profile.csi_sample_rate
        self.model_path = profile.model_path
        return profile
