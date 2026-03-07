"""Benchmark: measure inference latency, throughput, and memory for WiFi pose model.

Usage:
    python -m server.benchmark
    python -m server.benchmark --profile esp32s3 --window-size 60
    python -m server.benchmark --device cuda
"""
import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from server.config import Settings, HARDWARE_PROFILES
from server.pose_model import WiFiPoseModel
from server.signal_processor import SignalProcessor

logger = logging.getLogger(__name__)


def measure_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> dict:
    """Measure inference latency over n_runs forward passes."""
    model.to(device)
    x = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
        "n_runs": n_runs,
    }


def measure_throughput(
    model: torch.nn.Module,
    input_dim: int,
    window_size: int,
    device: torch.device,
    batch_sizes: list[int] = [1, 4, 8, 16, 32],
    n_runs: int = 50,
) -> list[dict]:
    """Measure throughput at different batch sizes."""
    results = []
    model.to(device)

    for bs in batch_sizes:
        x = torch.randn(bs, window_size, input_dim, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timed
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        samples_per_sec = (bs * n_runs) / elapsed
        results.append({
            "batch_size": bs,
            "total_time_s": round(elapsed, 3),
            "samples_per_sec": round(samples_per_sec, 1),
            "ms_per_sample": round(1000.0 / samples_per_sec, 2),
        })

    return results


def measure_signal_processing(
    settings: Settings,
    n_nodes: int,
    window_size: int,
    n_runs: int = 500,
) -> dict:
    """Measure signal processing pipeline latency."""
    proc = SignalProcessor(settings)
    n_sub = settings.num_subcarriers

    # Build a realistic window
    window = []
    for _ in range(window_size):
        frame = {}
        for nid in range(n_nodes):
            frame[nid] = np.random.randn(n_sub).astype(np.float32) * 0.5 + 1.0
        window.append(frame)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        proc.prepare_model_input(window, fs=float(settings.csi_sample_rate))
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": float(1000.0 / np.mean(latencies)),
        "n_runs": n_runs,
    }


def measure_memory(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> dict:
    """Measure model memory usage."""
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    result = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": round(param_mb, 2),
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        model.to(device)
        x = input_tensor.to(device)
        with torch.no_grad():
            model(x)
        result["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    return result


def measure_e2e_latency(
    model: torch.nn.Module,
    settings: Settings,
    n_nodes: int,
    window_size: int,
    device: torch.device,
    n_runs: int = 100,
) -> dict:
    """Measure full end-to-end: signal processing + inference."""
    proc = SignalProcessor(settings)
    n_sub = settings.num_subcarriers

    window = []
    for _ in range(window_size):
        frame = {}
        for nid in range(n_nodes):
            frame[nid] = np.random.randn(n_sub).astype(np.float32) * 0.5 + 1.0
        window.append(frame)

    model.to(device)
    latencies = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        # Signal processing
        prepared = proc.prepare_model_input(
            window, fs=float(settings.csi_sample_rate)
        )
        # To tensor
        x = torch.from_numpy(prepared).unsqueeze(0).to(device)
        # Inference
        with torch.no_grad():
            joints = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": float(1000.0 / np.mean(latencies)),
        "realtime_capable": bool(np.mean(latencies) < (1000.0 / settings.csi_sample_rate)),
        "target_hz": settings.csi_sample_rate,
        "budget_ms": round(1000.0 / settings.csi_sample_rate, 1),
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Benchmark WiFi pose model")
    parser.add_argument("--profile", type=str, default="esp32s3")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args(argv)

    # Setup
    settings = Settings()
    if args.profile in HARDWARE_PROFILES:
        settings.hardware_profile = args.profile
        profile = settings.apply_hardware_profile()
        n_nodes = profile.max_nodes
        print(f"Profile: {profile.name}")
        print(f"  Subcarriers: {profile.num_subcarriers}, Nodes: {n_nodes}, "
              f"Rate: {profile.csi_sample_rate} Hz")
    else:
        n_nodes = settings.max_nodes
        print(f"Profile: default")

    input_dim = settings.num_subcarriers * n_nodes
    window_size = args.window_size

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"  Device: {device}")
    print(f"  Window: {window_size} frames, Input dim: {input_dim}")
    print(f"  Budget: {1000.0 / settings.csi_sample_rate:.1f} ms/frame "
          f"({settings.csi_sample_rate} Hz)")
    print()

    # Create model
    model = WiFiPoseModel(input_dim=input_dim, num_joints=settings.num_joints)
    model.train(False)

    # Load weights if available
    model_path = Path(settings.model_path)
    if model_path.exists():
        try:
            state = torch.load(str(model_path), map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded weights: {model_path}")
        except Exception:
            print(f"Could not load weights from {model_path}, using random init")
    else:
        print(f"No weights at {model_path}, using random init")
    print()

    sample_input = torch.randn(1, window_size, input_dim)

    # ── Memory ──
    print("=" * 60)
    print("MODEL SIZE")
    print("=" * 60)
    mem = measure_memory(model, sample_input, device)
    print(f"  Parameters:     {mem['total_params']:,}")
    print(f"  Trainable:      {mem['trainable_params']:,}")
    print(f"  Param size:     {mem['param_size_mb']} MB")
    if "gpu_peak_mb" in mem:
        print(f"  GPU peak:       {mem['gpu_peak_mb']} MB")
    print()

    # ── Inference Latency ──
    print("=" * 60)
    print("INFERENCE LATENCY (single sample)")
    print("=" * 60)
    inf = measure_inference(model, sample_input, device, n_runs=args.n_runs)
    print(f"  Mean:           {inf['mean_ms']:.2f} ms")
    print(f"  Median:         {inf['median_ms']:.2f} ms")
    print(f"  P95:            {inf['p95_ms']:.2f} ms")
    print(f"  P99:            {inf['p99_ms']:.2f} ms")
    print(f"  Min/Max:        {inf['min_ms']:.2f} / {inf['max_ms']:.2f} ms")
    print(f"  FPS:            {inf['fps']:.0f}")
    print()

    # ── Signal Processing ──
    print("=" * 60)
    print("SIGNAL PROCESSING LATENCY")
    print("=" * 60)
    sig = measure_signal_processing(settings, n_nodes, window_size, n_runs=500)
    print(f"  Mean:           {sig['mean_ms']:.2f} ms")
    print(f"  Median:         {sig['median_ms']:.2f} ms")
    print(f"  P95:            {sig['p95_ms']:.2f} ms")
    print(f"  FPS:            {sig['fps']:.0f}")
    print()

    # ── End-to-End ──
    print("=" * 60)
    print("END-TO-END LATENCY (signal processing + inference)")
    print("=" * 60)
    e2e = measure_e2e_latency(model, settings, n_nodes, window_size, device)
    budget = e2e["budget_ms"]
    mean = e2e["mean_ms"]
    verdict = "YES" if e2e["realtime_capable"] else "NO"
    print(f"  Mean:           {mean:.2f} ms")
    print(f"  Median:         {e2e['median_ms']:.2f} ms")
    print(f"  P95:            {e2e['p95_ms']:.2f} ms")
    print(f"  FPS:            {e2e['fps']:.0f}")
    print(f"  Target:         {e2e['target_hz']} Hz ({budget} ms budget)")
    print(f"  Real-time:      {verdict} ({mean:.1f} ms vs {budget} ms budget)")
    print()

    # ── Throughput ──
    print("=" * 60)
    print("BATCH THROUGHPUT")
    print("=" * 60)
    tp = measure_throughput(model, input_dim, window_size, device)
    print(f"  {'Batch':>6}  {'Time':>8}  {'Samples/s':>10}  {'ms/sample':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}")
    for r in tp:
        print(f"  {r['batch_size']:>6}  {r['total_time_s']:>7.2f}s  "
              f"{r['samples_per_sec']:>10.1f}  {r['ms_per_sample']:>10.2f}")
    print()

    # ── Summary ──
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Profile:        {args.profile}")
    print(f"  Device:         {device}")
    print(f"  Model params:   {mem['total_params']:,} ({mem['param_size_mb']} MB)")
    print(f"  Inference:      {inf['mean_ms']:.2f} ms ({inf['fps']:.0f} FPS)")
    print(f"  Signal proc:    {sig['mean_ms']:.2f} ms")
    print(f"  End-to-end:     {mean:.2f} ms ({e2e['fps']:.0f} FPS)")
    print(f"  Real-time:      {verdict} (need <{budget} ms, got {mean:.1f} ms)")
    print()


if __name__ == "__main__":
    main()
