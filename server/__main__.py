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
import socket
import uvicorn
from server.config import Settings, HARDWARE_PROFILES
from server.api import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def _get_local_ips() -> list[str]:
    """Get all non-loopback IPv4 addresses on this machine."""
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    # Deduplicate while preserving order
    return list(dict.fromkeys(ips))


def _print_qr(url: str) -> None:
    """Print a QR code to terminal if qrcode lib is available."""
    try:
        import qrcode  # type: ignore
        qr = qrcode.QRCode(box_size=1, border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ImportError:
        pass


def _print_banner(port: int, simulate: bool) -> None:
    mode = "SIMULATION" if simulate else "REAL HARDWARE"
    ips = _get_local_ips()
    primary_ip = ips[0] if ips else "localhost"
    url = f"http://{primary_ip}:{port}/dashboard/"

    print()
    print("  ==========================================")
    print(f"   WiFi Body — {mode} MODE")
    print("  ==========================================")
    print()
    print(f"   Dashboard:  {url}")
    for ip in ips[1:]:
        print(f"               http://{ip}:{port}/dashboard/")
    print(f"   Local:      http://localhost:{port}/dashboard/")
    print()
    print("   Scan QR code with tablet to connect:")
    print()
    _print_qr(url)
    print()
    print("   Press Ctrl+C to stop")
    print("  ==========================================")
    print()


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
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run in simulation mode (generate synthetic data)",
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

    if args.simulate:
        settings.simulate = True

    app = create_app(settings)
    _print_banner(settings.api_port, settings.simulate)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")


if __name__ == "__main__":
    main()
