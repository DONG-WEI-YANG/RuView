"""Synthetic paired CSI + pose data generator for training pipeline testing.

Generates realistic motion sequences for different activities and simulates
the WiFi CSI signals that each ESP32 node would observe for those poses.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# 24-joint human skeleton (metres, Y-up)
# ---------------------------------------------------------------------------
JOINT_NAMES: List[str] = [
    "head", "neck", "chest", "spine",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_hip", "l_knee", "l_ankle",
    "r_hip", "r_knee", "r_ankle",
    "l_foot", "r_foot",
    "l_hand", "r_hand",
    "l_toe", "r_toe",
    "l_eye", "r_eye",
]

BASE_SKELETON = np.array([
    [0.0, 1.70, 0.0],    # 0  head
    [0.0, 1.55, 0.0],    # 1  neck
    [0.0, 1.35, 0.0],    # 2  chest
    [0.0, 1.10, 0.0],    # 3  spine
    [-0.20, 1.50, 0.0],  # 4  l_shoulder
    [-0.35, 1.20, 0.0],  # 5  l_elbow
    [-0.35, 0.90, 0.0],  # 6  l_wrist
    [0.20, 1.50, 0.0],   # 7  r_shoulder
    [0.35, 1.20, 0.0],   # 8  r_elbow
    [0.35, 0.90, 0.0],   # 9  r_wrist
    [-0.10, 1.00, 0.0],  # 10 l_hip
    [-0.10, 0.50, 0.0],  # 11 l_knee
    [-0.10, 0.05, 0.0],  # 12 l_ankle
    [0.10, 1.00, 0.0],   # 13 r_hip
    [0.10, 0.50, 0.0],   # 14 r_knee
    [0.10, 0.05, 0.0],   # 15 r_ankle
    [-0.10, 0.0, 0.10],  # 16 l_foot
    [0.10, 0.0, 0.10],   # 17 r_foot
    [-0.40, 0.85, 0.0],  # 18 l_hand
    [0.40, 0.85, 0.0],   # 19 r_hand
    [-0.10, 0.0, 0.20],  # 20 l_toe
    [0.10, 0.0, 0.20],   # 21 r_toe
    [-0.03, 1.72, 0.05], # 22 l_eye
    [0.03, 1.72, 0.05],  # 23 r_eye
], dtype=np.float32)

# Relative "influence weight" of each joint on the WiFi signal.
# Larger body parts (torso, thighs) affect the signal more than small
# extremities (fingers, eyes).
JOINT_INFLUENCE = np.array([
    0.6,   # head
    0.5,   # neck
    1.0,   # chest
    0.9,   # spine
    0.5,   # l_shoulder
    0.3,   # l_elbow
    0.2,   # l_wrist
    0.5,   # r_shoulder
    0.3,   # r_elbow
    0.2,   # r_wrist
    0.7,   # l_hip
    0.6,   # l_knee
    0.3,   # l_ankle
    0.7,   # r_hip
    0.6,   # r_knee
    0.3,   # r_ankle
    0.2,   # l_foot
    0.2,   # r_foot
    0.15,  # l_hand
    0.15,  # r_hand
    0.1,   # l_toe
    0.1,   # r_toe
    0.05,  # l_eye
    0.05,  # r_eye
], dtype=np.float32)

ACTIVITIES = ["standing", "walking", "sitting", "falling", "exercising"]


class SyntheticDataGenerator:
    """Generates paired (CSI, joint-position) training data.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    noise_std : float
        Standard deviation of additive Gaussian noise on CSI.
    """

    def __init__(self, seed: int | None = None, noise_std: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sequence(
        self,
        activity: str,
        n_frames: int,
        n_nodes: int = 4,
        n_sub: int = 56,
    ) -> Dict[str, np.ndarray]:
        """Generate a single motion sequence with paired CSI.

        Returns
        -------
        dict with keys ``csi``, ``joints``, ``labels``.
        """
        if activity not in ACTIVITIES:
            raise ValueError(
                f"Unknown activity '{activity}'. Choose from {ACTIVITIES}."
            )

        joints = self._generate_motion(activity, n_frames)          # (F, 24, 3)
        csi = self.simulate_csi(joints, n_nodes, n_sub)              # (F, N, S)
        labels = np.array([activity] * n_frames)

        return {
            "csi": csi.astype(np.float32),
            "joints": joints.astype(np.float32),
            "labels": labels,
        }

    def generate_dataset(
        self,
        n_sequences_per_activity: int,
        n_frames: int,
        output_dir: str,
        n_nodes: int = 4,
        n_sub: int = 56,
    ) -> List[str]:
        """Generate multiple sequences and save as .npz files.

        Returns
        -------
        List of file paths written.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []

        for activity in ACTIVITIES:
            for i in range(n_sequences_per_activity):
                data = self.generate_sequence(activity, n_frames, n_nodes, n_sub)
                fname = out / f"{activity}_{i:04d}.npz"
                np.savez(
                    str(fname),
                    csi=data["csi"],
                    joints=data["joints"],
                    labels=data["labels"],
                )
                paths.append(str(fname))

        return paths

    # ------------------------------------------------------------------
    # Motion generation (private)
    # ------------------------------------------------------------------

    def _generate_motion(self, activity: str, n_frames: int) -> np.ndarray:
        """Return joint positions (n_frames, 24, 3) for *activity*."""
        dispatch = {
            "standing": self._motion_standing,
            "walking": self._motion_walking,
            "sitting": self._motion_sitting,
            "falling": self._motion_falling,
            "exercising": self._motion_exercising,
        }
        return dispatch[activity](n_frames)

    # -- standing --

    def _motion_standing(self, n: int) -> np.ndarray:
        """Minimal movement with slight body sway."""
        frames = np.tile(BASE_SKELETON, (n, 1, 1)).copy()
        t = np.arange(n, dtype=np.float32)

        # Small sinusoidal sway (breathing / balance)
        sway_freq = self.rng.uniform(0.3, 0.8)
        sway_amp = self.rng.uniform(0.005, 0.015)
        sway = sway_amp * np.sin(2 * np.pi * sway_freq * t / 20.0)

        # Apply lateral sway to all joints
        frames[:, :, 0] += sway[:, None]
        # Tiny random jitter
        frames += self.rng.normal(0, 0.002, frames.shape).astype(np.float32)
        return frames.astype(np.float32)

    # -- walking --

    def _motion_walking(self, n: int) -> np.ndarray:
        """Periodic arm/leg swing with forward motion along Z."""
        frames = np.tile(BASE_SKELETON, (n, 1, 1)).copy()
        t = np.arange(n, dtype=np.float32)
        stride_freq = self.rng.uniform(0.8, 1.2)
        phase = 2 * np.pi * stride_freq * t / 20.0
        speed = self.rng.uniform(0.01, 0.03)  # metres per frame

        # Forward translation along Z
        frames[:, :, 2] += (t * speed)[:, None]

        # Vertical bounce
        bounce = 0.02 * np.abs(np.sin(phase))
        frames[:, :, 1] += bounce[:, None]

        # Leg swing (hip -> knee -> ankle on each side, anti-phase)
        leg_amp = self.rng.uniform(0.08, 0.15)
        for leg_joints, sign in [([10, 11, 12, 16, 20], 1.0),
                                  ([13, 14, 15, 17, 21], -1.0)]:
            swing = sign * leg_amp * np.sin(phase)
            for j in leg_joints:
                frames[:, j, 2] += swing

        # Arm swing (opposite to legs)
        arm_amp = self.rng.uniform(0.05, 0.10)
        for arm_joints, sign in [([4, 5, 6, 18], -1.0),
                                  ([7, 8, 9, 19], 1.0)]:
            swing = sign * arm_amp * np.sin(phase)
            for j in arm_joints:
                frames[:, j, 2] += swing

        frames += self.rng.normal(0, 0.003, frames.shape).astype(np.float32)
        return frames.astype(np.float32)

    # -- sitting --

    def _motion_sitting(self, n: int) -> np.ndarray:
        """Transition from standing to a seated posture."""
        frames = np.tile(BASE_SKELETON, (n, 1, 1)).copy()
        t = np.arange(n, dtype=np.float32)

        # Smooth transition: sigmoid from standing to sitting
        transition_mid = n * 0.3
        transition_rate = 6.0 / n
        alpha = 1.0 / (1.0 + np.exp(-transition_rate * (t - transition_mid) * 10))

        # Seated offsets: lower torso and head, bend knees forward
        seat_height_drop = self.rng.uniform(0.35, 0.50)
        upper_body = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19, 22, 23]
        for j in upper_body:
            frames[:, j, 1] -= alpha * seat_height_drop

        # Knees bend forward
        knee_forward = self.rng.uniform(0.15, 0.25)
        for j in [11, 14]:
            frames[:, j, 2] += alpha * knee_forward
            frames[:, j, 1] -= alpha * 0.10  # knees drop slightly

        # Feet stay roughly in place but slide forward a bit
        for j in [12, 15, 16, 17, 20, 21]:
            frames[:, j, 2] += alpha * knee_forward * 0.5

        # Hips lower
        for j in [10, 13]:
            frames[:, j, 1] -= alpha * seat_height_drop * 0.8

        # Small idle sway while seated
        sway = 0.005 * np.sin(2 * np.pi * 0.5 * t / 20.0)
        frames[:, :, 0] += sway[:, None]
        frames += self.rng.normal(0, 0.002, frames.shape).astype(np.float32)
        return frames.astype(np.float32)

    # -- falling --

    def _motion_falling(self, n: int) -> np.ndarray:
        """Rapid descent of upper body, then collapse to the ground."""
        frames = np.tile(BASE_SKELETON, (n, 1, 1)).copy()
        t = np.arange(n, dtype=np.float32)

        # Two phases: stumble then collapse
        fall_start = int(n * 0.2)
        fall_end = int(n * 0.6)

        for i in range(n):
            if i < fall_start:
                # Slight stumble / sway before fall
                sway = 0.02 * np.sin(2 * np.pi * 2.0 * i / 20.0)
                frames[i, :, 0] += sway
            elif i < fall_end:
                # Rapid descent: quadratic drop
                progress = (i - fall_start) / max(fall_end - fall_start, 1)
                drop = progress ** 2
                # Upper body descends and tilts
                for j in range(24):
                    base_y = BASE_SKELETON[j, 1]
                    # Higher joints fall more (head drops to ~0.3m)
                    fall_amount = base_y * 0.8 * drop
                    frames[i, j, 1] = base_y - fall_amount
                    # Lateral tilt
                    frames[i, j, 0] += 0.3 * drop * (base_y / 1.7)
                    # Forward lean
                    frames[i, j, 2] += 0.2 * drop * (base_y / 1.7)
            else:
                # On the ground: all joints near floor level
                for j in range(24):
                    base_y = BASE_SKELETON[j, 1]
                    ground_y = max(0.0, base_y * 0.15)
                    frames[i, j, 1] = ground_y
                    frames[i, j, 0] = BASE_SKELETON[j, 0] + 0.3 * (base_y / 1.7)
                    frames[i, j, 2] = BASE_SKELETON[j, 2] + 0.2 * (base_y / 1.7)

        frames += self.rng.normal(0, 0.003, frames.shape).astype(np.float32)
        return frames.astype(np.float32)

    # -- exercising --

    def _motion_exercising(self, n: int) -> np.ndarray:
        """Repetitive arm raises (like overhead press / jumping jacks)."""
        frames = np.tile(BASE_SKELETON, (n, 1, 1)).copy()
        t = np.arange(n, dtype=np.float32)
        freq = self.rng.uniform(0.5, 1.0)
        phase = 2 * np.pi * freq * t / 20.0

        # Arm raise: shoulders/elbows/wrists/hands move upward on positive phase
        raise_amount = self.rng.uniform(0.25, 0.45)
        arm_joints = [4, 5, 6, 7, 8, 9, 18, 19]
        lift = raise_amount * (0.5 + 0.5 * np.sin(phase))  # 0 to raise_amount
        for j in arm_joints:
            frames[:, j, 1] += lift
            # Spread arms laterally on raise
            side = -1.0 if j in [4, 5, 6, 18] else 1.0
            frames[:, j, 0] += side * 0.10 * (0.5 + 0.5 * np.sin(phase))

        # Slight knee bend (squat component)
        squat = 0.05 * (0.5 + 0.5 * np.sin(phase))
        for j in [10, 11, 13, 14]:
            frames[:, j, 1] -= squat * 0.3
        # Upper body slight dip during squat phase
        for j in [0, 1, 2, 3, 22, 23]:
            frames[:, j, 1] -= squat * 0.15

        frames += self.rng.normal(0, 0.003, frames.shape).astype(np.float32)
        return frames.astype(np.float32)

    # ------------------------------------------------------------------
    # CSI simulation (private)
    # ------------------------------------------------------------------

    def simulate_csi(
        self,
        joints: np.ndarray,
        n_nodes: int,
        n_sub: int,
    ) -> np.ndarray:
        """Convert joint positions to simulated CSI amplitudes.

        Model: CSI = base_signal + body_influence(joints) + noise

        Each ESP32 node is placed at a fixed position in a ~4x4m room.
        Each subcarrier has a characteristic frequency (index-dependent).
        Body influence depends on distance and angle from each node to
        each joint, weighted by ``JOINT_INFLUENCE``.

        Parameters
        ----------
        joints : (n_frames, 24, 3)
        n_nodes : number of ESP32 receiver nodes
        n_sub   : number of OFDM subcarriers per node

        Returns
        -------
        csi : (n_frames, n_nodes, n_sub) float32
        """
        n_frames = joints.shape[0]

        # Place nodes evenly around the room perimeter at ~1m height
        node_positions = self._place_nodes(n_nodes)  # (n_nodes, 3)

        # Subcarrier centre-frequencies (normalised 0..1)
        sub_idx = np.arange(n_sub, dtype=np.float32) / n_sub

        # Base signal per subcarrier (slowly varying, node-specific)
        base = np.ones((n_nodes, n_sub), dtype=np.float32) * 0.5
        # Add a per-node spectral shape
        for nd in range(n_nodes):
            phase_offset = self.rng.uniform(0, 2 * np.pi)
            base[nd] += 0.1 * np.sin(2 * np.pi * sub_idx * 3 + phase_offset)

        csi = np.zeros((n_frames, n_nodes, n_sub), dtype=np.float32)

        for f in range(n_frames):
            frame_joints = joints[f]  # (24, 3)
            for nd in range(n_nodes):
                node_pos = node_positions[nd]  # (3,)

                # Distance from each joint to this node
                diff = frame_joints - node_pos[None, :]       # (24, 3)
                dist = np.linalg.norm(diff, axis=1)            # (24,)
                dist = np.clip(dist, 0.1, None)                # avoid div-by-0

                # Body influence: inverse-distance weighted by joint size
                # Each joint contributes a "bump" across nearby subcarriers
                influence = np.zeros(n_sub, dtype=np.float32)
                for j in range(24):
                    weight = JOINT_INFLUENCE[j] / (dist[j] ** 2)
                    # Joint maps to a region of subcarriers based on its
                    # angle from the node (azimuth mapped to subcarrier index)
                    angle = np.arctan2(diff[j, 0], diff[j, 2] + 1e-8)
                    # Map angle (-pi..pi) to subcarrier range (0..n_sub)
                    centre = ((angle + np.pi) / (2 * np.pi)) * n_sub
                    # Gaussian bump around that centre
                    spread = 3.0 + dist[j] * 2.0  # farther = broader
                    bump = np.exp(
                        -0.5 * ((np.arange(n_sub) - centre) / spread) ** 2
                    )
                    influence += weight * bump

                csi[f, nd] = base[nd] + influence

        # Additive Gaussian noise
        csi += self.rng.normal(0, self.noise_std, csi.shape).astype(np.float32)

        return csi.astype(np.float32)

    @staticmethod
    def _place_nodes(n_nodes: int) -> np.ndarray:
        """Distribute *n_nodes* evenly around a 4x4m room at 1m height."""
        positions = []
        for i in range(n_nodes):
            angle = 2 * np.pi * i / n_nodes
            x = 2.0 * np.cos(angle)
            z = 2.0 * np.sin(angle)
            positions.append([x, 1.0, z])
        return np.array(positions, dtype=np.float32)
