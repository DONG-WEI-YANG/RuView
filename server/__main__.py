"""Main entry point: python -m server

Usage:
    python -m server                        # default ESP32-S3 profile
    python -m server --profile esp32s3      # explicit profile
    python -m server --profile tplink_n750  # TP-Link for MM-Fi data
    python -m server --profile intel5300    # Intel 5300 for Wi-Pose data
    python -m server --list-profiles        # show available profiles
"""
import argparse
import logging
import uvicorn
from server.config import Settings, HARDWARE_PROFILES
from server.api import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="WiFi Body pose estimation server")
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Hardware profile ID (determines CSI params + model weights)",
    )
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="List available hardware profiles and exit",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="API server port (default: 8000)",
    )
    args = parser.parse_args()

    if args.list_profiles:
        print("\nAvailable hardware profiles:\n")
        for pid, p in HARDWARE_PROFILES.items():
            from pathlib import Path
            ready = "ready" if Path(p.model_path).exists() else "no weights"
            print(f"  {pid:18s}  {p.name}")
            print(f"                      {p.description}")
            print(f"                      {p.num_subcarriers} sub, {p.csi_sample_rate} Hz, "
                  f"{p.frequency_ghz} GHz, {p.bandwidth_mhz} MHz BW, "
                  f"{p.max_nodes} nodes max")
            print(f"                      model: {p.model_path} [{ready}]")
            if p.dataset:
                print(f"                      dataset: {p.dataset}")
            print()
        print("Usage: python -m server --profile <id>")
        return

    settings = Settings()
    if args.profile:
        if args.profile not in HARDWARE_PROFILES:
            print(f"Unknown profile '{args.profile}'. Use --list-profiles to see options.")
            raise SystemExit(1)
        settings.hardware_profile = args.profile

    if args.port:
        settings.api_port = args.port

    app = create_app(settings)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")


if __name__ == "__main__":
    main()
