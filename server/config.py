from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_prefix="", env_file=".env")

    # UDP receiver
    udp_host: str = "0.0.0.0"
    udp_port: int = 5005

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # CSI parameters
    num_subcarriers: int = 56
    csi_sample_rate: int = 20  # Hz per node
    max_nodes: int = 6

    # Model
    num_joints: int = 24
    model_path: str = "models/pose_model.pth"

    # Fall detection
    fall_threshold: float = 0.8
    fall_alert_cooldown: int = 30  # seconds
