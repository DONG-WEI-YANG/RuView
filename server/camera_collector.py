"""Camera-based data collector for ground truth pose estimation.

Uses a webcam and rtmlib's RTMPose body detector to capture 2D poses,
converts them to the project's 24-joint 3D skeleton format, and pairs
them with simulated CSI data for model training.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from server.data_generator import SyntheticDataGenerator

# ---------------------------------------------------------------------------
# COCO 17 keypoint indices (as returned by rtmlib Body)
# ---------------------------------------------------------------------------
COCO_NOSE = 0
COCO_LEFT_EYE = 1
COCO_RIGHT_EYE = 2
COCO_LEFT_EAR = 3
COCO_RIGHT_EAR = 4
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16

# Minimum confidence threshold for keypoint detection
MIN_CONFIDENCE = 0.3

# Skeleton connections for preview drawing (COCO 17 pairs)
SKELETON_CONNECTIONS = [
    (COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER),
    (COCO_LEFT_SHOULDER, COCO_LEFT_ELBOW),
    (COCO_LEFT_ELBOW, COCO_LEFT_WRIST),
    (COCO_RIGHT_SHOULDER, COCO_RIGHT_ELBOW),
    (COCO_RIGHT_ELBOW, COCO_RIGHT_WRIST),
    (COCO_LEFT_SHOULDER, COCO_LEFT_HIP),
    (COCO_RIGHT_SHOULDER, COCO_RIGHT_HIP),
    (COCO_LEFT_HIP, COCO_RIGHT_HIP),
    (COCO_LEFT_HIP, COCO_LEFT_KNEE),
    (COCO_LEFT_KNEE, COCO_LEFT_ANKLE),
    (COCO_RIGHT_HIP, COCO_RIGHT_KNEE),
    (COCO_RIGHT_KNEE, COCO_RIGHT_ANKLE),
    (COCO_NOSE, COCO_LEFT_EYE),
    (COCO_NOSE, COCO_RIGHT_EYE),
    (COCO_LEFT_EYE, COCO_LEFT_EAR),
    (COCO_RIGHT_EYE, COCO_RIGHT_EAR),
]


def coco17_to_24joint(
    keypoints_2d: np.ndarray,
    scores: np.ndarray,
    standing_height: float,
    image_height: int,
) -> np.ndarray:
    """Convert COCO 17-keypoint 2D detections to the project's 24-joint 3D format.

    Parameters
    ----------
    keypoints_2d : (17, 2)
        Pixel coordinates (x, y) for each COCO keypoint.
    scores : (17,)
        Confidence scores for each keypoint.
    standing_height : float
        Real-world standing height in meters for pixel-to-meter conversion.
    image_height : int
        Height of the source image in pixels (used for Y-axis flip).

    Returns
    -------
    joints : (24, 3)
        Joint positions in meters (X-right, Y-up, Z-depth).
    """
    kp = keypoints_2d.copy().astype(np.float64)

    # --- Compute pixel scale factor ---
    # Person height in pixels: top of head (nose) to midpoint of ankles
    nose = kp[COCO_NOSE]
    l_ankle = kp[COCO_LEFT_ANKLE]
    r_ankle = kp[COCO_RIGHT_ANKLE]
    mid_ankle = (l_ankle + r_ankle) / 2.0

    pixel_height = np.abs(mid_ankle[1] - nose[1])
    if pixel_height < 1.0:
        pixel_height = 1.0  # avoid division by zero

    scale = standing_height / pixel_height

    # --- Compute origin: mid-hip in pixel space ---
    l_hip = kp[COCO_LEFT_HIP]
    r_hip = kp[COCO_RIGHT_HIP]
    mid_hip = (l_hip + r_hip) / 2.0

    # --- Convert pixel coords to meters, centred on mid-hip ---
    # In pixel space: X-right, Y-down.  We need Y-up.
    def px_to_m(px_xy: np.ndarray) -> np.ndarray:
        mx = (px_xy[0] - mid_hip[0]) * scale
        my = -(px_xy[1] - mid_hip[1]) * scale  # flip Y
        return np.array([mx, my], dtype=np.float64)

    # Convert all 17 keypoints to metres (2D)
    kp_m = np.array([px_to_m(kp[i]) for i in range(17)])  # (17, 2)

    # --- Helper for derived points ---
    l_shoulder_m = kp_m[COCO_LEFT_SHOULDER]
    r_shoulder_m = kp_m[COCO_RIGHT_SHOULDER]
    l_hip_m = kp_m[COCO_LEFT_HIP]
    r_hip_m = kp_m[COCO_RIGHT_HIP]
    l_elbow_m = kp_m[COCO_LEFT_ELBOW]
    r_elbow_m = kp_m[COCO_RIGHT_ELBOW]
    l_wrist_m = kp_m[COCO_LEFT_WRIST]
    r_wrist_m = kp_m[COCO_RIGHT_WRIST]
    l_ankle_m = kp_m[COCO_LEFT_ANKLE]
    r_ankle_m = kp_m[COCO_RIGHT_ANKLE]

    mid_shoulder_m = (l_shoulder_m + r_shoulder_m) / 2.0
    mid_hip_m = (l_hip_m + r_hip_m) / 2.0  # should be ~(0, 0) by construction

    # --- Derived joints (2D metre coordinates) ---
    # head = nose position
    head = kp_m[COCO_NOSE]

    # neck = midpoint of shoulders, raised slightly (10% of shoulder-to-nose distance)
    nose_m = kp_m[COCO_NOSE]
    neck_offset = (nose_m - mid_shoulder_m) * 0.15
    neck = mid_shoulder_m + neck_offset

    # spine = midpoint between mid-shoulder and mid-hip
    spine = (mid_shoulder_m + mid_hip_m) / 2.0

    # chest = midpoint between neck and spine
    chest = (neck + spine) / 2.0

    # feet = ankles with small downward offset
    foot_offset = np.array([0.0, -0.05], dtype=np.float64)
    l_foot = l_ankle_m + foot_offset
    r_foot = r_ankle_m + foot_offset

    # hands = wrists extended slightly along forearm direction
    def extend_along(start: np.ndarray, end: np.ndarray, dist: float) -> np.ndarray:
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return end
        return end + direction / norm * dist

    l_hand = extend_along(l_elbow_m, l_wrist_m, 0.05)
    r_hand = extend_along(r_elbow_m, r_wrist_m, 0.05)

    # toes = feet with small forward offset (positive Z in 3D, but we approximate
    # in the 2D plane as slightly further down)
    l_toe = l_foot + np.array([0.0, -0.05], dtype=np.float64)
    r_toe = r_foot + np.array([0.0, -0.05], dtype=np.float64)

    # --- Assemble 24-joint array with pseudo-depth (Z) ---
    # Estimate Z from body proportions: torso joints are at Z=0,
    # limbs get slight depth variation based on angles.
    joints = np.zeros((24, 3), dtype=np.float32)

    def set_joint(idx: int, xy: np.ndarray, z: float = 0.0) -> None:
        joints[idx, 0] = xy[0]
        joints[idx, 1] = xy[1]
        joints[idx, 2] = z

    # Estimate pseudo-depth for limbs based on bone length ratios
    # (shorter-than-expected projected bone = limb pointing towards/away from camera)
    def estimate_limb_depth(
        parent: np.ndarray, child: np.ndarray, expected_len: float
    ) -> float:
        projected_len = np.linalg.norm(child - parent)
        if projected_len >= expected_len or expected_len < 1e-6:
            return 0.0
        ratio = projected_len / expected_len
        ratio = np.clip(ratio, 0.0, 1.0)
        return np.sqrt(1.0 - ratio**2) * expected_len * 0.5

    # Expected bone lengths (approximate, in meters) for depth estimation
    upper_arm_len = 0.30
    forearm_len = 0.25
    thigh_len = 0.45
    shin_len = 0.45

    l_elbow_z = estimate_limb_depth(l_shoulder_m, l_elbow_m, upper_arm_len)
    r_elbow_z = estimate_limb_depth(r_shoulder_m, r_elbow_m, upper_arm_len)
    l_wrist_z = l_elbow_z + estimate_limb_depth(l_elbow_m, l_wrist_m, forearm_len)
    r_wrist_z = r_elbow_z + estimate_limb_depth(r_elbow_m, r_wrist_m, forearm_len)
    l_knee_z = estimate_limb_depth(l_hip_m, kp_m[COCO_LEFT_KNEE], thigh_len)
    r_knee_z = estimate_limb_depth(r_hip_m, kp_m[COCO_RIGHT_KNEE], thigh_len)
    l_ankle_z = l_knee_z + estimate_limb_depth(
        kp_m[COCO_LEFT_KNEE], l_ankle_m, shin_len
    )
    r_ankle_z = r_knee_z + estimate_limb_depth(
        kp_m[COCO_RIGHT_KNEE], r_ankle_m, shin_len
    )

    # 0: head
    set_joint(0, head, 0.05)
    # 1: neck
    set_joint(1, neck, 0.0)
    # 2: chest
    set_joint(2, chest, 0.0)
    # 3: spine
    set_joint(3, spine, 0.0)
    # 4: l_shoulder
    set_joint(4, l_shoulder_m, 0.0)
    # 5: l_elbow
    set_joint(5, l_elbow_m, l_elbow_z)
    # 6: l_wrist
    set_joint(6, l_wrist_m, l_wrist_z)
    # 7: r_shoulder
    set_joint(7, r_shoulder_m, 0.0)
    # 8: r_elbow
    set_joint(8, r_elbow_m, r_elbow_z)
    # 9: r_wrist
    set_joint(9, r_wrist_m, r_wrist_z)
    # 10: l_hip
    set_joint(10, l_hip_m, 0.0)
    # 11: l_knee
    set_joint(11, kp_m[COCO_LEFT_KNEE], l_knee_z)
    # 12: l_ankle
    set_joint(12, l_ankle_m, l_ankle_z)
    # 13: r_hip
    set_joint(13, r_hip_m, 0.0)
    # 14: r_knee
    set_joint(14, kp_m[COCO_RIGHT_KNEE], r_knee_z)
    # 15: r_ankle
    set_joint(15, r_ankle_m, r_ankle_z)
    # 16: l_foot
    set_joint(16, l_foot, l_ankle_z + 0.05)
    # 17: r_foot
    set_joint(17, r_foot, r_ankle_z + 0.05)
    # 18: l_hand
    set_joint(18, l_hand, l_wrist_z + 0.02)
    # 19: r_hand
    set_joint(19, r_hand, r_wrist_z + 0.02)
    # 20: l_toe
    set_joint(20, l_toe, l_ankle_z + 0.15)
    # 21: r_toe
    set_joint(21, r_toe, r_ankle_z + 0.15)
    # 22: l_eye
    set_joint(22, kp_m[COCO_LEFT_EYE], 0.05)
    # 23: r_eye
    set_joint(23, kp_m[COCO_RIGHT_EYE], 0.05)

    return joints


class CameraCollector:
    """Camera-based data collection tool for ground truth pose capture.

    Captures frames from a webcam, runs pose detection via rtmlib,
    converts to the 24-joint skeleton, and optionally pairs with
    simulated CSI data for training.

    Parameters
    ----------
    camera_index : int
        OpenCV camera index (default 0).
    model_mode : str
        rtmlib model mode: ``'lightweight'`` or ``'performance'``.
    standing_height : float
        Real-world height of the subject in meters for pixel-to-meter
        conversion (default 1.70).
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_mode: str = "lightweight",
        standing_height: float = 1.70,
    ):
        self.camera_index = camera_index
        self.model_mode = model_mode
        self.standing_height = standing_height

        # Open camera
        self.cap = cv2.VideoCapture(camera_index)

        # Initialize rtmlib Body detector
        from rtmlib import Body

        self.body = Body(mode=model_mode)

        # CSI simulator (for mixed mode)
        self._csi_gen = SyntheticDataGenerator(seed=None, noise_std=0.05)

        # Recording state
        self._recording = False
        self._activity: str = ""
        self._max_frames: int = 0
        self._joint_buffer: List[np.ndarray] = []
        self._csi_buffer: List[np.ndarray] = []

        # Cache last raw 2D keypoints for preview drawing (avoids re-running detection)
        self.last_raw_keypoints: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_frame(
        self,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Capture one frame and run pose detection.

        Returns
        -------
        image : (H, W, 3) uint8
            The captured BGR image.
        keypoints_24x3 : (24, 3) float32 or None
            24-joint 3D positions in metres, or ``None`` if no person detected.
        confidence_17 : (17,) float32 or None
            Per-keypoint confidence scores from COCO 17, or ``None``.
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Return a blank frame if capture failed
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return frame, None, None

        # Run rtmlib pose detection
        # rtmlib Body returns (keypoints, scores) where:
        #   keypoints: (N, 17, 2) and scores: (N, 17)
        keypoints, scores = self.body(frame)

        if keypoints is None or len(keypoints) == 0:
            self.last_raw_keypoints = None
            return frame, None, None

        # Take the first (most confident) person
        kp = keypoints[0]  # (17, 2)
        sc = scores[0]  # (17,)

        # Cache raw 2D keypoints for preview drawing
        self.last_raw_keypoints = kp.copy()

        # Check if we have enough confident keypoints for a valid pose
        valid_count = np.sum(sc >= MIN_CONFIDENCE)
        if valid_count < 6:
            return frame, None, None

        # For low-confidence keypoints, use a simple interpolation strategy:
        # replace with the mean of nearby high-confidence keypoints
        for i in range(17):
            if sc[i] < MIN_CONFIDENCE:
                # Find high-confidence keypoints and use their weighted mean
                high_conf = sc >= MIN_CONFIDENCE
                if np.any(high_conf):
                    weights = sc[high_conf]
                    weights = weights / weights.sum()
                    kp[i] = np.average(kp[high_conf], weights=weights, axis=0)

        h = frame.shape[0]
        joints_24 = coco17_to_24joint(kp, sc, self.standing_height, h)

        # If recording, append to buffer
        if self._recording and len(self._joint_buffer) < self._max_frames:
            self._joint_buffer.append(joints_24.copy())

            # Generate simulated CSI from the detected pose
            csi_frame = self._simulate_csi_frame(joints_24)
            self._csi_buffer.append(csi_frame)

        return frame, joints_24, sc.astype(np.float32)

    def start_recording(self, activity: str, n_frames: int = 200) -> None:
        """Start recording a sequence for the given activity label.

        Parameters
        ----------
        activity : str
            Activity label (e.g. ``'walking'``, ``'standing'``).
        n_frames : int
            Maximum number of frames to record.
        """
        self._recording = True
        self._activity = activity
        self._max_frames = n_frames
        self._joint_buffer = []
        self._csi_buffer = []

    def stop_recording(self) -> Dict[str, np.ndarray] | None:
        """Stop recording and return accumulated data.

        Returns
        -------
        dict or None
            Dictionary with keys ``'joints'``, ``'labels'``, ``'csi'``
            matching the WiFiPoseDataset .npz format.  Returns ``None``
            if no frames were recorded.
        """
        self._recording = False

        if len(self._joint_buffer) == 0:
            return None

        n_frames = len(self._joint_buffer)
        joints = np.stack(self._joint_buffer, axis=0).astype(np.float32)  # (F, 24, 3)
        csi = np.stack(self._csi_buffer, axis=0).astype(np.float32)  # (F, N, S)
        labels = np.array([self._activity] * n_frames)

        self._joint_buffer = []
        self._csi_buffer = []

        return {
            "joints": joints,
            "labels": labels,
            "csi": csi,
        }

    def save_sequence(
        self, data: Dict[str, np.ndarray], output_dir: str, prefix: str = "camera"
    ) -> str:
        """Save recorded sequence as .npz file matching WiFiPoseDataset format.

        Parameters
        ----------
        data : dict
            Output from :meth:`stop_recording`.
        output_dir : str
            Directory to save into (created if needed).
        prefix : str
            Filename prefix (default ``'camera'``).

        Returns
        -------
        str
            Path to the saved .npz file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename with timestamp
        ts = int(time.time())
        activity = data["labels"][0] if len(data["labels"]) > 0 else "unknown"
        fname = out / f"{prefix}_{activity}_{ts}.npz"

        np.savez(
            str(fname),
            csi=data["csi"],
            joints=data["joints"],
            labels=data["labels"],
        )
        return str(fname)

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()

    # ------------------------------------------------------------------
    # Preview / drawing helpers (used by camera_cli.py)
    # ------------------------------------------------------------------

    def draw_skeleton(
        self,
        image: np.ndarray,
        keypoints_2d: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw detected skeleton on the image for preview.

        If *keypoints_2d* is not provided, returns the image unchanged.
        Expects raw COCO 17-format pixel keypoints from the last
        ``body()`` call.
        """
        if keypoints_2d is None:
            return image

        vis = image.copy()
        kp = keypoints_2d.astype(int)

        # Draw connections
        for i, j in SKELETON_CONNECTIONS:
            pt1 = tuple(kp[i])
            pt2 = tuple(kp[j])
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for pt in kp:
            cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)

        return vis

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_csi_frame(
        self,
        joints_24: np.ndarray,
        n_nodes: int = 4,
        n_sub: int = 56,
    ) -> np.ndarray:
        """Generate a simulated CSI frame from a single pose.

        Uses :class:`SyntheticDataGenerator`'s CSI simulation by wrapping
        the single frame in a length-1 sequence.

        Returns
        -------
        csi_frame : (n_nodes, n_sub) float32
        """
        # Wrap single frame: (1, 24, 3)
        joints_seq = joints_24[np.newaxis, :, :]
        csi_seq = self._csi_gen._simulate_csi(joints_seq, n_nodes, n_sub)  # (1, N, S)
        return csi_seq[0]
