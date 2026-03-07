"""Download and convert public WiFi CSI pose datasets to our .npz format.

Supported datasets:
    - MM-Fi (NeurIPS 2023): 320k+ frames, 40 subjects, WiFi CSI + 3D pose
    - Wi-Pose: 166,600 CSI packets, 12 subjects x 12 actions
    - WiMANS (ECCV 2024): Multi-user WiFi activity sensing

Usage:
    # Download MM-Fi dataset (recommended — largest, best quality)
    python -m server.dataset_download --dataset mmfi --output data/mmfi

    # Download Wi-Pose dataset
    python -m server.dataset_download --dataset wipose --output data/wipose

    # Convert already-downloaded MM-Fi data
    python -m server.dataset_download --dataset mmfi --raw-dir /path/to/MMFi --output data/mmfi

    # List available datasets
    python -m server.dataset_download --list
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ======================================================================
# Dataset registry
# ======================================================================

DATASETS = {
    "mmfi": {
        "name": "MM-Fi (NeurIPS 2023)",
        "description": "Multi-modal 4D human dataset, 40 subjects, 27 actions, WiFi CSI + 3D pose",
        "github": "https://github.com/ybhbingo/MMFi_dataset",
        "paper": "https://arxiv.org/abs/2305.10345",
        "format": "mat",
        "subjects": 40,
        "actions": 27,
        "frames": "320k+",
    },
    "wipose": {
        "name": "Wi-Pose",
        "description": "WiFi CSI pose dataset, 12 subjects, 12 actions, .mat format",
        "github": "https://github.com/NjtechCVLab/Wi-PoseDataset",
        "format": "mat",
        "subjects": 12,
        "actions": 12,
        "frames": "166,600",
    },
    "wimans": {
        "name": "WiMANS (ECCV 2024)",
        "description": "Multi-user WiFi activity sensing, CSI + video",
        "github": "https://github.com/huangshk/WiMANS",
        "format": "mat/npy",
        "subjects": "multi-user",
        "actions": 12,
        "frames": "varied",
    },
}


# ======================================================================
# MM-Fi adapter
# ======================================================================


def convert_mmfi(raw_dir: str, output_dir: str, max_subjects: int = 0) -> int:
    """Convert MM-Fi dataset to our .npz format.

    MM-Fi structure:
        raw_dir/
            S01/ S02/ ... S40/
                A01/ A02/ ... A27/
                    wifi_csi.mat    -> CSI data
                    keypoint.npy    -> 3D pose (17 COCO keypoints)

    Our format (.npz):
        csi:    (T, n_nodes, n_sub)   float32
        joints: (T, 24, 3)           float32
        labels: (T,)                 str
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.error("MM-Fi raw directory not found: %s", raw_path)
        logger.info(
            "Download MM-Fi from: %s",
            DATASETS["mmfi"]["github"],
        )
        logger.info("Then run: python -m server.dataset_download --dataset mmfi --raw-dir <path>")
        return 0

    try:
        from scipy.io import loadmat
    except ImportError:
        logger.error("scipy required: pip install scipy")
        return 0

    subject_dirs = sorted(raw_path.glob("S*"))
    if max_subjects > 0:
        subject_dirs = subject_dirs[:max_subjects]

    n_converted = 0

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        action_dirs = sorted(subject_dir.glob("A*"))

        for action_dir in action_dirs:
            action_id = action_dir.name
            csi_file = action_dir / "wifi_csi.mat"
            pose_file = action_dir / "keypoint.npy"

            if not csi_file.exists() or not pose_file.exists():
                continue

            try:
                # Load CSI from .mat
                mat = loadmat(str(csi_file))
                # MM-Fi CSI key varies; try common ones
                csi_data = None
                for key in ["CSIamp", "csi_amp", "csi", "CSI", "amplitude"]:
                    if key in mat:
                        csi_data = mat[key]
                        break
                if csi_data is None:
                    # Use the first non-metadata key
                    data_keys = [k for k in mat.keys() if not k.startswith("_")]
                    if data_keys:
                        csi_data = mat[data_keys[0]]
                    else:
                        logger.warning("No CSI data found in %s", csi_file)
                        continue

                csi_data = np.array(csi_data, dtype=np.float32)

                # Load 3D pose keypoints
                pose_data = np.load(str(pose_file)).astype(np.float32)

                # Align lengths
                n_frames = min(csi_data.shape[0], pose_data.shape[0])
                csi_data = csi_data[:n_frames]
                pose_data = pose_data[:n_frames]

                # MM-Fi has 17 COCO keypoints, we need 24 joints
                # Pad with interpolated/zero joints for the extra 7
                joints_24 = _coco17_to_24joints(pose_data)

                # Reshape CSI if needed: ensure (T, n_nodes, n_sub)
                if csi_data.ndim == 2:
                    # (T, features) -> (T, 1, features)
                    csi_data = csi_data[:, np.newaxis, :]
                elif csi_data.ndim == 4:
                    # (T, n_rx, n_tx, n_sub) -> (T, n_rx*n_tx, n_sub)
                    t, nr, nt, ns = csi_data.shape
                    csi_data = csi_data.reshape(t, nr * nt, ns)

                # Create labels
                labels = np.array([action_id] * n_frames)

                # Save
                out_file = out_path / f"{subject_id}_{action_id}.npz"
                np.savez(
                    str(out_file),
                    csi=csi_data,
                    joints=joints_24,
                    labels=labels,
                )
                n_converted += 1
                logger.info("Converted %s/%s -> %s (%d frames)", subject_id, action_id, out_file.name, n_frames)

            except Exception as e:
                logger.warning("Failed to convert %s/%s: %s", subject_id, action_id, e)

    logger.info("MM-Fi conversion complete: %d sequences converted to %s", n_converted, out_path)
    return n_converted


def _coco17_to_24joints(pose_17: np.ndarray) -> np.ndarray:
    """Map 17 COCO keypoints to our 24-joint skeleton.

    COCO 17: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
             l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip,
             l_knee, r_knee, l_ankle, r_ankle

    Our 24: head, neck, chest, spine, l_shoulder, l_elbow, l_wrist,
            r_shoulder, r_elbow, r_wrist, l_hip, l_knee, l_ankle,
            r_hip, r_knee, r_ankle, l_foot, r_foot, l_hand, r_hand,
            l_toe, r_toe, l_eye, r_eye
    """
    n_frames = pose_17.shape[0]
    joints_24 = np.zeros((n_frames, 24, 3), dtype=np.float32)

    # Direct mappings
    joints_24[:, 0] = pose_17[:, 0]                    # head = nose
    joints_24[:, 22] = pose_17[:, 1]                   # l_eye
    joints_24[:, 23] = pose_17[:, 2]                   # r_eye
    joints_24[:, 4] = pose_17[:, 5]                    # l_shoulder
    joints_24[:, 5] = pose_17[:, 7]                    # l_elbow
    joints_24[:, 6] = pose_17[:, 9]                    # l_wrist
    joints_24[:, 7] = pose_17[:, 6]                    # r_shoulder
    joints_24[:, 8] = pose_17[:, 8]                    # r_elbow
    joints_24[:, 9] = pose_17[:, 10]                   # r_wrist
    joints_24[:, 10] = pose_17[:, 11]                  # l_hip
    joints_24[:, 11] = pose_17[:, 13]                  # l_knee
    joints_24[:, 12] = pose_17[:, 15]                  # l_ankle
    joints_24[:, 13] = pose_17[:, 12]                  # r_hip
    joints_24[:, 14] = pose_17[:, 14]                  # r_knee
    joints_24[:, 15] = pose_17[:, 16]                  # r_ankle

    # Interpolated joints
    joints_24[:, 1] = (pose_17[:, 5] + pose_17[:, 6]) / 2    # neck = mid(shoulders)
    joints_24[:, 2] = (joints_24[:, 1] + joints_24[:, 3]) / 2  # chest (will fix below)
    mid_hip = (pose_17[:, 11] + pose_17[:, 12]) / 2
    joints_24[:, 3] = mid_hip                                  # spine = mid(hips)
    joints_24[:, 2] = (joints_24[:, 1] + joints_24[:, 3]) / 2  # chest = mid(neck, spine)

    # Extrapolated extremities
    joints_24[:, 16] = pose_17[:, 15] + np.array([0, -0.05, 0.03])  # l_foot ≈ l_ankle + offset
    joints_24[:, 17] = pose_17[:, 16] + np.array([0, -0.05, 0.03])  # r_foot
    joints_24[:, 18] = pose_17[:, 9] + np.array([0, -0.05, 0])      # l_hand ≈ l_wrist + offset
    joints_24[:, 19] = pose_17[:, 10] + np.array([0, -0.05, 0])     # r_hand
    joints_24[:, 20] = joints_24[:, 16] + np.array([0, 0, 0.05])    # l_toe
    joints_24[:, 21] = joints_24[:, 17] + np.array([0, 0, 0.05])    # r_toe

    return joints_24


# ======================================================================
# Wi-Pose adapter
# ======================================================================


def convert_wipose(raw_dir: str, output_dir: str) -> int:
    """Convert Wi-Pose dataset (.mat files) to our .npz format.

    Wi-Pose structure:
        raw_dir/
            Train/ Test/
                *.mat files with CSI + pose annotations
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.error("Wi-Pose raw directory not found: %s", raw_path)
        logger.info("Download Wi-Pose from: %s", DATASETS["wipose"]["github"])
        return 0

    try:
        from scipy.io import loadmat
    except ImportError:
        logger.error("scipy required: pip install scipy")
        return 0

    mat_files = sorted(raw_path.rglob("*.mat"))
    n_converted = 0

    for mat_file in mat_files:
        try:
            mat = loadmat(str(mat_file))
            data_keys = [k for k in mat.keys() if not k.startswith("_")]

            # Wi-Pose typically has 'csi_data' and 'keypoints' or similar
            csi_data = None
            pose_data = None

            for key in data_keys:
                arr = np.array(mat[key], dtype=np.float32)
                if "csi" in key.lower() or "channel" in key.lower():
                    csi_data = arr
                elif "pose" in key.lower() or "keypoint" in key.lower() or "joint" in key.lower():
                    pose_data = arr

            if csi_data is None or pose_data is None:
                # Try to infer from shapes
                arrays = [(k, np.array(mat[k], dtype=np.float32)) for k in data_keys
                          if mat[k].ndim >= 2]
                arrays.sort(key=lambda x: x[1].shape[-1], reverse=True)
                if len(arrays) >= 2:
                    csi_data = arrays[0][1]  # Larger last dim = CSI subcarriers
                    pose_data = arrays[1][1]
                else:
                    logger.warning("Cannot identify CSI/pose data in %s", mat_file)
                    continue

            n_frames = min(csi_data.shape[0], pose_data.shape[0])
            csi_data = csi_data[:n_frames]
            pose_data = pose_data[:n_frames]

            # Ensure CSI shape (T, n_nodes, n_sub)
            if csi_data.ndim == 2:
                csi_data = csi_data[:, np.newaxis, :]

            # Map pose to 24 joints if needed
            n_kp = pose_data.shape[1] if pose_data.ndim >= 2 else 0
            if n_kp == 17:
                joints = _coco17_to_24joints(pose_data)
            elif n_kp == 24:
                joints = pose_data
            else:
                # Pad/truncate to 24
                joints = np.zeros((n_frames, 24, 3), dtype=np.float32)
                copy_n = min(n_kp, 24)
                if pose_data.ndim == 3:
                    joints[:, :copy_n] = pose_data[:, :copy_n]
                elif pose_data.ndim == 2:
                    # (T, n_kp*3) -> (T, n_kp, 3)
                    n_kp_actual = pose_data.shape[1] // 3
                    reshaped = pose_data[:, :n_kp_actual * 3].reshape(n_frames, n_kp_actual, 3)
                    copy_n = min(n_kp_actual, 24)
                    joints[:, :copy_n] = reshaped[:, :copy_n]

            # Extract label from folder/filename
            rel = mat_file.relative_to(raw_path)
            label = rel.stem
            labels = np.array([label] * n_frames)

            out_name = f"wipose_{rel.parent.name}_{rel.stem}.npz"
            np.savez(
                str(out_path / out_name),
                csi=csi_data,
                joints=joints,
                labels=labels,
            )
            n_converted += 1
            logger.info("Converted %s -> %s (%d frames)", mat_file.name, out_name, n_frames)

        except Exception as e:
            logger.warning("Failed to convert %s: %s", mat_file.name, e)

    logger.info("Wi-Pose conversion complete: %d files converted to %s", n_converted, out_path)
    return n_converted


# ======================================================================
# CLI
# ======================================================================


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download and convert WiFi CSI pose datasets",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--dataset", type=str, choices=list(DATASETS.keys()),
        help="Dataset to download/convert",
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Path to already-downloaded raw dataset",
    )
    parser.add_argument(
        "--output", type=str, default="data/real",
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=0,
        help="Limit number of subjects to convert (0=all)",
    )

    args = parser.parse_args()

    if args.list or args.dataset is None:
        print("\nAvailable WiFi CSI pose datasets:\n")
        for key, info in DATASETS.items():
            print(f"  {key:12s}  {info['name']}")
            print(f"               {info['description']}")
            print(f"               Subjects: {info['subjects']}, Actions: {info['actions']}, Frames: {info['frames']}")
            print(f"               GitHub: {info['github']}")
            print()
        print("Usage:")
        print("  1. Download dataset from GitHub link above")
        print("  2. Run: python -m server.dataset_download --dataset mmfi --raw-dir /path/to/MMFi --output data/mmfi")
        print("  3. Train: python -m server.train --data-dir data/mmfi --epochs 100")
        return

    if args.raw_dir is None:
        # Try default locations
        defaults = {
            "mmfi": ["data/raw/MMFi", "data/MMFi", "MMFi_dataset"],
            "wipose": ["data/raw/Wi-Pose", "data/Wi-Pose", "Wi-PoseDataset"],
            "wimans": ["data/raw/WiMANS", "data/WiMANS"],
        }
        for candidate in defaults.get(args.dataset, []):
            if Path(candidate).exists():
                args.raw_dir = candidate
                break

        if args.raw_dir is None:
            info = DATASETS[args.dataset]
            print(f"\nDataset '{args.dataset}' not found locally.")
            print(f"\nTo use {info['name']}:")
            print(f"  1. Clone: git clone {info['github']}")
            print(f"  2. Follow the README to download the data files")
            print(f"  3. Run: python -m server.dataset_download --dataset {args.dataset} --raw-dir <path> --output {args.output}")
            return

    if args.dataset == "mmfi":
        n = convert_mmfi(args.raw_dir, args.output, max_subjects=args.max_subjects)
    elif args.dataset == "wipose":
        n = convert_wipose(args.raw_dir, args.output)
    else:
        logger.error("Conversion not yet implemented for %s", args.dataset)
        return

    if n > 0:
        print(f"\nConversion complete! {n} sequences saved to {args.output}")
        print(f"\nNext step — train model:")
        print(f"  python -m server.train --data-dir {args.output} --epochs 100 --batch-size 32")
    else:
        print("\nNo sequences converted. Check the raw directory structure.")


if __name__ == "__main__":
    main()
