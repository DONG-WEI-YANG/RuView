"""CLI entry point for the camera-based data collector.

Usage::

    python -m server.camera_cli --activity walking --frames 200 --output data/camera

Interactive controls (when preview is enabled):
    r  - Start recording
    s  - Stop recording and save
    q  - Quit
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from server.camera_collector import CameraCollector, SKELETON_CONNECTIONS

VALID_ACTIVITIES = ["standing", "walking", "sitting", "exercising"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera-based pose data collector for WiFi Body project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Interactive controls (preview mode):\n"
            "  r  - Start recording\n"
            "  s  - Stop recording and save\n"
            "  q  - Quit\n"
        ),
    )
    parser.add_argument(
        "--activity",
        type=str,
        default="standing",
        choices=VALID_ACTIVITIES,
        help="Activity label (default: standing)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Number of frames to record (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/camera",
        help="Output directory (default: data/camera)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=1.70,
        help="Standing height in meters (default: 1.70)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Show preview window (default: True)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        default=False,
        help="Disable preview window",
    )
    return parser.parse_args(argv)


def draw_status_overlay(
    image: np.ndarray,
    recording: bool,
    activity: str,
    frame_count: int,
    max_frames: int,
) -> np.ndarray:
    """Draw recording status text overlay on the image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # Background bar at the top
    cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 0), -1)

    if recording:
        # Red dot + recording text
        cv2.circle(vis, (20, 20), 8, (0, 0, 255), -1)
        status_text = f"REC [{activity}] {frame_count}/{max_frames}"
        colour = (0, 0, 255)
    else:
        status_text = f"READY [{activity}] - Press 'r' to record"
        colour = (0, 255, 0)

    cv2.putText(
        vis,
        status_text,
        (40, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2,
    )

    # Help text at bottom
    help_text = "r:Record  s:Stop+Save  q:Quit"
    cv2.putText(
        vis,
        help_text,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    return vis


def draw_skeleton_on_frame(
    image: np.ndarray,
    body_model,
    frame: np.ndarray,
) -> np.ndarray:
    """Run body detection and draw skeleton overlay."""
    keypoints, scores = body_model(frame)

    vis = image.copy()
    if keypoints is None or len(keypoints) == 0:
        return vis

    kp = keypoints[0].astype(int)  # first person, (17, 2)

    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        pt1 = tuple(kp[i])
        pt2 = tuple(kp[j])
        cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

    # Draw keypoints
    for pt in kp:
        cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)

    return vis


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    show_preview = args.preview and not args.no_preview

    print(f"Initialising camera collector (camera={args.camera}, "
          f"height={args.height}m, activity={args.activity})...")

    collector = CameraCollector(
        camera_index=args.camera,
        model_mode="lightweight",
        standing_height=args.height,
    )

    activity = args.activity
    max_frames = args.frames
    recording = False
    frame_count = 0

    print("Camera ready.")
    if show_preview:
        print("Controls: r=Record, s=Stop+Save, q=Quit")
    else:
        print("No-preview mode: recording automatically...")
        collector.start_recording(activity, n_frames=max_frames)
        recording = True

    try:
        while True:
            frame, joints, confidence = collector.capture_frame()

            if recording:
                frame_count = len(collector._joint_buffer)

                # Auto-stop when max frames reached
                if frame_count >= max_frames:
                    print(f"\nReached {max_frames} frames. Saving...")
                    data = collector.stop_recording()
                    recording = False
                    if data is not None:
                        path = collector.save_sequence(
                            data, args.output, prefix="camera"
                        )
                        print(f"Saved: {path}")
                        print(f"  joints: {data['joints'].shape}")
                        print(f"  csi:    {data['csi'].shape}")
                        print(f"  labels: {data['labels'].shape}")
                    if not show_preview:
                        break

            if show_preview:
                vis = draw_status_overlay(
                    frame, recording, activity, frame_count, max_frames
                )
                # Draw skeleton using cached raw keypoints (no re-detection)
                if joints is not None and collector.last_raw_keypoints is not None:
                    kp = collector.last_raw_keypoints.astype(int)
                    for i, j in SKELETON_CONNECTIONS:
                        pt1 = tuple(kp[i])
                        pt2 = tuple(kp[j])
                        cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
                    for pt in kp:
                        cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)

                cv2.imshow("WiFi Body - Camera Collector", vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Quitting...")
                    break
                elif key == ord("r") and not recording:
                    print(f"Recording started: {activity} ({max_frames} frames)")
                    collector.start_recording(activity, n_frames=max_frames)
                    recording = True
                    frame_count = 0
                elif key == ord("s") and recording:
                    data = collector.stop_recording()
                    recording = False
                    frame_count = 0
                    if data is not None:
                        path = collector.save_sequence(
                            data, args.output, prefix="camera"
                        )
                        print(f"Saved: {path}")
                        print(f"  joints: {data['joints'].shape}")
                        print(f"  csi:    {data['csi'].shape}")
                        print(f"  labels: {data['labels'].shape}")
                    else:
                        print("No frames were captured.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        collector.release()
        if show_preview:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
