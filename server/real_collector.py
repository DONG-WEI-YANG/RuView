"""Real-world data collector: pairs live ESP32 CSI with camera ground-truth poses.

Simultaneously captures:
  - Real CSI frames from ESP32 nodes via UDP
  - Ground-truth 3D poses from a webcam + rtmlib RTMPose

Outputs time-synchronised .npz files matching the WiFiPoseDataset format,
ready for model training.

Usage:
    python -m server.real_collector --activity walking --duration 30
    python -m server.real_collector --activity standing --duration 20 --nodes 4
    python -m server.real_collector --list-activities
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from server.config import Settings, HARDWARE_PROFILES

logger = logging.getLogger(__name__)

# Supported activity labels (same as SyntheticDataGenerator)
ACTIVITIES = ["standing", "walking", "sitting", "falling", "exercising",
              "waving", "stretching", "turning"]


class RealDataCollector:
    """Synchronised CSI + camera data collector for real-world training data.

    Parameters
    ----------
    settings : Settings
        Server settings (UDP host/port, num_subcarriers, etc.).
    camera_index : int
        OpenCV camera index.
    standing_height : float
        Subject's real height in metres for pixel-to-metre conversion.
    n_nodes : int
        Expected number of ESP32 nodes (for padding missing nodes).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        camera_index: int = 0,
        standing_height: float = 1.70,
        n_nodes: int = 4,
    ):
        if settings is None:
            settings = Settings()
        self.settings = settings
        self.camera_index = camera_index
        self.standing_height = standing_height
        self.n_nodes = n_nodes
        self.n_sub = settings.num_subcarriers

        # State
        self._csi_buffer: Dict[int, np.ndarray] = {}  # node_id -> latest amplitude
        self._recording = False
        self._frames_joints: List[np.ndarray] = []
        self._frames_csi: List[np.ndarray] = []
        self._frames_timestamps: List[float] = []
        self._csi_frame_count = 0

    # ------------------------------------------------------------------
    # CSI callback (called from CSIReceiver)
    # ------------------------------------------------------------------

    def on_csi_frame(self, frame) -> None:
        """Handle incoming CSI frame from UDP receiver."""
        self._csi_frame_count += 1
        if frame.amplitude is not None:
            self._csi_buffer[frame.node_id] = frame.amplitude.copy()

    # ------------------------------------------------------------------
    # Snapshot: grab synchronised CSI + camera at one time instant
    # ------------------------------------------------------------------

    def snapshot_csi(self) -> np.ndarray:
        """Return current multi-node CSI snapshot as (n_nodes, n_sub) array.

        Missing nodes are zero-padded.
        """
        csi = np.zeros((self.n_nodes, self.n_sub), dtype=np.float32)
        for node_id, amp in self._csi_buffer.items():
            if node_id < self.n_nodes:
                n = min(len(amp), self.n_sub)
                csi[node_id, :n] = amp[:n]
        return csi

    # ------------------------------------------------------------------
    # Recording session
    # ------------------------------------------------------------------

    def start_recording(self, activity: str) -> None:
        """Begin a recording session."""
        self._recording = True
        self._activity = activity
        self._frames_joints = []
        self._frames_csi = []
        self._frames_timestamps = []
        logger.info("Recording started: activity=%s", activity)

    def add_frame(self, joints: np.ndarray) -> None:
        """Add one synchronised frame (call after camera capture + CSI snapshot)."""
        if not self._recording:
            return
        csi = self.snapshot_csi()
        self._frames_joints.append(joints.copy())
        self._frames_csi.append(csi.copy())
        self._frames_timestamps.append(time.time())

    def stop_recording(self) -> Dict[str, np.ndarray] | None:
        """Stop and return accumulated data."""
        self._recording = False
        if len(self._frames_joints) == 0:
            logger.warning("No frames recorded")
            return None

        n = len(self._frames_joints)
        joints = np.stack(self._frames_joints, axis=0).astype(np.float32)
        csi = np.stack(self._frames_csi, axis=0).astype(np.float32)
        timestamps = np.array(self._frames_timestamps, dtype=np.float64)
        labels = np.array([self._activity] * n)

        logger.info(
            "Recording stopped: %d frames, activity=%s, CSI nodes seen=%d",
            n, self._activity, len(self._csi_buffer),
        )

        self._frames_joints = []
        self._frames_csi = []
        self._frames_timestamps = []

        return {
            "joints": joints,      # (F, 24, 3)
            "csi": csi,            # (F, n_nodes, n_sub)
            "labels": labels,      # (F,)
            "timestamps": timestamps,
        }

    def save_sequence(
        self, data: Dict[str, np.ndarray], output_dir: str, prefix: str = "real"
    ) -> str:
        """Save recorded sequence as .npz file."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        activity = data["labels"][0] if len(data["labels"]) > 0 else "unknown"
        fname = out / f"{prefix}_{activity}_{ts}.npz"

        np.savez(
            str(fname),
            csi=data["csi"],
            joints=data["joints"],
            labels=data["labels"],
            timestamps=data["timestamps"],
        )
        logger.info("Saved %s (%d frames)", fname, len(data["joints"]))
        return str(fname)


# ======================================================================
# CLI entry point
# ======================================================================


def _run_collection(args: argparse.Namespace) -> None:
    """Run the interactive collection session."""
    import cv2

    settings = Settings()
    if args.profile:
        if args.profile not in HARDWARE_PROFILES:
            logger.error("Unknown profile: %s", args.profile)
            raise SystemExit(1)
        settings.hardware_profile = args.profile
        settings.apply_hardware_profile()

    collector = RealDataCollector(
        settings=settings,
        camera_index=args.camera,
        standing_height=args.height,
        n_nodes=args.nodes,
    )

    # --- Start CSI receiver in background ---
    from server.csi_receiver import CSIReceiver
    receiver = CSIReceiver(settings)
    receiver.on_frame = collector.on_csi_frame

    loop = asyncio.new_event_loop()

    async def _receive_csi():
        await receiver.start()

    csi_task = loop.run_in_executor(None, lambda: asyncio.run(_receive_csi()))
    logger.info(
        "CSI receiver listening on %s:%d", settings.udp_host, settings.udp_port
    )

    # --- Start camera + pose detection ---
    try:
        from server.camera_collector import coco17_to_24joint, MIN_CONFIDENCE
        from rtmlib import Body
    except ImportError:
        logger.error(
            "Camera collection requires rtmlib. Install with: pip install rtmlib"
        )
        raise SystemExit(1)

    body = Body(mode=args.model_mode)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", args.camera)
        raise SystemExit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    print("\n=== WiFi Body Real-World Data Collector ===")
    print(f"Camera: {args.camera}  |  Nodes: {args.nodes}  |  Profile: {args.profile or 'default'}")
    print(f"Output: {output_dir}")
    print()
    print("Controls:")
    print("  [R] Start/stop recording")
    print("  [1-8] Set activity: " + ", ".join(f"[{i+1}]{a}" for i, a in enumerate(ACTIVITIES[:8])))
    print("  [Q] Quit")
    print()

    current_activity = args.activity
    recording = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose detection
        keypoints, scores = body(frame)
        joints_24 = None

        if keypoints is not None and len(keypoints) > 0:
            kp = keypoints[0]
            sc = scores[0]
            if np.sum(sc >= MIN_CONFIDENCE) >= 6:
                # Fill low-confidence keypoints
                for i in range(17):
                    if sc[i] < MIN_CONFIDENCE:
                        high_conf = sc >= MIN_CONFIDENCE
                        if np.any(high_conf):
                            weights = sc[high_conf]
                            weights = weights / weights.sum()
                            kp[i] = np.average(kp[high_conf], weights=weights, axis=0)
                joints_24 = coco17_to_24joint(kp, args.height)

            # Draw skeleton on frame
            for i in range(17):
                if sc[i] >= MIN_CONFIDENCE:
                    x, y = int(kp[i][0]), int(kp[i][1])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # If recording and we have a valid pose, add frame
        if recording and joints_24 is not None:
            collector.add_frame(joints_24)
            frame_count += 1

        # Draw status overlay
        status_color = (0, 0, 255) if recording else (200, 200, 200)
        status_text = f"REC [{current_activity}] {frame_count}f" if recording else f"READY [{current_activity}]"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        csi_status = f"CSI: {collector._csi_frame_count} frames, {len(collector._csi_buffer)} nodes"
        cv2.putText(frame, csi_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        if joints_24 is not None:
            cv2.putText(frame, "POSE OK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "NO POSE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow("WiFi Body - Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            if recording:
                data = collector.stop_recording()
                if data is not None:
                    path = collector.save_sequence(data, str(output_dir))
                    saved_files.append(path)
                    print(f"  Saved: {path} ({frame_count} frames)")
                recording = False
                frame_count = 0
            else:
                collector.start_recording(current_activity)
                recording = True
                frame_count = 0
        elif ord('1') <= key <= ord('8'):
            idx = key - ord('1')
            if idx < len(ACTIVITIES):
                current_activity = ACTIVITIES[idx]
                print(f"  Activity: {current_activity}")

    # Cleanup
    if recording:
        data = collector.stop_recording()
        if data is not None:
            path = collector.save_sequence(data, str(output_dir))
            saved_files.append(path)

    cap.release()
    cv2.destroyAllWindows()
    receiver.stop()

    print(f"\nCollection complete. {len(saved_files)} sequences saved to {output_dir}/")
    for f in saved_files:
        print(f"  {f}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Collect real-world CSI + camera data for training",
    )
    parser.add_argument(
        "--activity", type=str, default="standing",
        help="Initial activity label",
    )
    parser.add_argument(
        "--output", type=str, default="data/real",
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index",
    )
    parser.add_argument(
        "--nodes", type=int, default=4,
        help="Expected number of ESP32 nodes",
    )
    parser.add_argument(
        "--height", type=float, default=1.70,
        help="Subject standing height in metres",
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Hardware profile ID",
    )
    parser.add_argument(
        "--model-mode", type=str, default="lightweight",
        choices=["lightweight", "performance"],
        help="rtmlib model mode",
    )
    parser.add_argument(
        "--list-activities", action="store_true",
        help="List supported activities and exit",
    )

    args = parser.parse_args(argv)

    if args.list_activities:
        print("Supported activities:")
        for i, a in enumerate(ACTIVITIES):
            print(f"  [{i+1}] {a}")
        return

    _run_collection(args)


if __name__ == "__main__":
    main()
