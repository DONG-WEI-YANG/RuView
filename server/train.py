"""Training pipeline for WiFi CSI pose estimation model.

Usage:
    # Generate synthetic data and train:
    python -m server.train --synthetic --epochs 50 --batch-size 32

    # Train on real collected data:
    python -m server.train --data-dir data/collected --epochs 100

    # Resume training:
    python -m server.train --data-dir data/collected --resume models/checkpoint.pth
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from server.config import Settings
from server.pose_model import WiFiPoseModel
from server.dataset import create_dataloaders
from server.data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skeleton connectivity for bone-length regularisation
# ---------------------------------------------------------------------------

# Each tuple is (parent_joint_idx, child_joint_idx) following the 24-joint
# layout defined in server.data_generator.JOINT_NAMES.
BONE_PAIRS = [
    (1, 0),    # neck  -> head
    (2, 1),    # chest -> neck
    (3, 2),    # spine -> chest
    (1, 4),    # neck  -> l_shoulder
    (4, 5),    # l_shoulder -> l_elbow
    (5, 6),    # l_elbow    -> l_wrist
    (6, 18),   # l_wrist    -> l_hand
    (1, 7),    # neck  -> r_shoulder
    (7, 8),    # r_shoulder -> r_elbow
    (8, 9),    # r_elbow    -> r_wrist
    (9, 19),   # r_wrist    -> r_hand
    (3, 10),   # spine -> l_hip
    (10, 11),  # l_hip  -> l_knee
    (11, 12),  # l_knee -> l_ankle
    (12, 16),  # l_ankle -> l_foot
    (16, 20),  # l_foot -> l_toe
    (3, 13),   # spine -> r_hip
    (13, 14),  # r_hip  -> r_knee
    (14, 15),  # r_knee -> r_ankle
    (15, 17),  # r_ankle -> r_foot
    (17, 21),  # r_foot -> r_toe
    (0, 22),   # head -> l_eye
    (0, 23),   # head -> r_eye
]


# ======================================================================
# Loss functions
# ======================================================================


def mpjpe_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Per Joint Position Error.

    Computes the L2 (Euclidean) distance for each joint, then averages
    across all joints and the batch dimension.

    Args:
        pred:   (batch, num_joints, 3)
        target: (batch, num_joints, 3)

    Returns:
        Scalar loss tensor.
    """
    # Per-joint L2 distance: (batch, num_joints)
    per_joint = torch.norm(pred - target, dim=-1)
    # Average over joints, then over batch
    return per_joint.mean()


def bone_length_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalise variance of bone lengths across the batch.

    For each bone (pair of connected joints), compute the Euclidean length
    in every sample, then take the variance across the batch.  The loss
    is the average variance across all bones.  This encourages the model
    to produce skeletons with consistent proportions.

    Args:
        pred: (batch, num_joints, 3)

    Returns:
        Scalar loss tensor (non-negative).
    """
    bone_vars = []
    for parent, child in BONE_PAIRS:
        bone_vec = pred[:, child] - pred[:, parent]        # (batch, 3)
        bone_len = torch.norm(bone_vec, dim=-1)            # (batch,)
        bone_vars.append(bone_len.var())
    return torch.stack(bone_vars).mean()


# ======================================================================
# Metrics
# ======================================================================


def compute_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute MPJPE (Mean Per Joint Position Error) as a plain float.

    Args:
        pred:   (batch, num_joints, 3)
        target: (batch, num_joints, 3)

    Returns:
        MPJPE value as a Python float.
    """
    with torch.no_grad():
        per_joint = torch.norm(pred - target, dim=-1)
        return per_joint.mean().item()


def compute_pck(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.05,
) -> float:
    """Percentage of Correct Keypoints.

    A joint is "correct" if its L2 distance from the target is at most
    *threshold*.

    Args:
        pred:      (batch, num_joints, 3)
        target:    (batch, num_joints, 3)
        threshold: Distance threshold (same units as joint coordinates).

    Returns:
        Fraction in [0, 1].
    """
    with torch.no_grad():
        distances = torch.norm(pred - target, dim=-1)   # (batch, num_joints)
        correct = (distances <= threshold).float()
        return correct.mean().item()


# ======================================================================
# Training & validation loops
# ======================================================================


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bone_weight: float = 0.01,
) -> float:
    """Run one training epoch.

    Args:
        model:       The pose model.
        loader:      Training DataLoader.
        optimizer:   Optimizer.
        device:      Torch device.
        bone_weight: Weight for the bone-length regulariser.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for csi_batch, joints_batch in loader:
        csi_batch = csi_batch.to(device)
        joints_batch = joints_batch.to(device)

        pred = model(csi_batch)

        loss = mpjpe_loss(pred, joints_batch)
        if bone_weight > 0.0:
            loss = loss + bone_weight * bone_length_loss(pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run validation and compute metrics.

    Args:
        model:  The pose model.
        loader: Validation DataLoader.
        device: Torch device.

    Returns:
        (avg_loss, mpjpe, pck)
    """
    model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    total_pck = 0.0
    n_batches = 0

    with torch.no_grad():
        for csi_batch, joints_batch in loader:
            csi_batch = csi_batch.to(device)
            joints_batch = joints_batch.to(device)

            pred = model(csi_batch)

            loss = mpjpe_loss(pred, joints_batch)
            total_loss += loss.item()
            total_mpjpe += compute_mpjpe(pred, joints_batch)
            total_pck += compute_pck(pred, joints_batch, threshold=0.05)
            n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_mpjpe / n, total_pck / n


# ======================================================================
# Main
# ======================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train WiFi CSI pose estimation model",
    )

    # Data source
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data before training",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing .npz training data",
    )

    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--n-nodes", type=int, default=4)
    parser.add_argument("--bone-weight", type=float, default=0.01)

    # Resume / output
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pth file to resume from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save models and training history",
    )

    # Synthetic data generation
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=10,
        help="Sequences per activity when using --synthetic",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=300,
        help="Frames per sequence when using --synthetic",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )

    args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args(argv)
    settings = Settings()

    # ------------------------------------------------------------------
    # 1. Determine data directory
    # ------------------------------------------------------------------
    if args.synthetic:
        data_dir = Path("data") / "synthetic"
        logger.info(
            "Generating synthetic data: %d sequences/activity, %d frames each",
            args.n_sequences,
            args.n_frames,
        )
        gen = SyntheticDataGenerator(seed=args.seed)
        gen.generate_dataset(
            n_sequences_per_activity=args.n_sequences,
            n_frames=args.n_frames,
            output_dir=str(data_dir),
            n_nodes=args.n_nodes,
        )
        logger.info("Synthetic data written to %s", data_dir)
    elif args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        logger.error("Specify --synthetic or --data-dir")
        raise SystemExit(1)

    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 2. Create dataloaders
    # ------------------------------------------------------------------
    train_loader, val_loader = create_dataloaders(
        str(data_dir),
        window_size=args.window_size,
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=0,
    )
    logger.info(
        "Data loaded: %d train samples, %d val samples",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # ------------------------------------------------------------------
    # 3. Determine input_dim from first batch
    # ------------------------------------------------------------------
    sample_csi, _ = next(iter(train_loader))
    input_dim = sample_csi.shape[-1]
    logger.info("Input dim: %d (window_size=%d)", input_dim, args.window_size)

    # ------------------------------------------------------------------
    # 4. Create model, optimizer, scheduler
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = WiFiPoseModel(
        input_dim=input_dim,
        num_joints=settings.num_joints,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # 5. Optionally resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            logger.error("Checkpoint not found: %s", ckpt_path)
            raise SystemExit(1)

        logger.info("Resuming from checkpoint: %s", ckpt_path)
        checkpoint = torch.load(
            str(ckpt_path), map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(
            "Resumed at epoch %d, best_val_loss=%.6f", start_epoch, best_val_loss
        )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_mpjpe": [],
        "val_pck": [],
        "lr": [],
        "epoch_time": [],
    }

    patience_counter = 0

    logger.info("Starting training for %d epochs", args.epochs)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, bone_weight=args.bone_weight
        )
        val_loss, val_mpjpe, val_pck = validate(model, val_loader, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mpjpe"].append(val_mpjpe)
        history["val_pck"].append(val_pck)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
            "MPJPE=%.4f | PCK=%.4f | lr=%.2e | %.1fs",
            epoch + 1,
            start_epoch + args.epochs,
            train_loss,
            val_loss,
            val_mpjpe,
            val_pck,
            current_lr,
            epoch_time,
        )

        # -- Checkpointing: save if val loss improved --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            ckpt_file = output_dir / "checkpoint.pth"
            torch.save(ckpt, str(ckpt_file))
            logger.info(
                "  -> Saved best checkpoint (val_loss=%.6f)", best_val_loss
            )
        else:
            patience_counter += 1

        # -- Early stopping --
        if patience_counter >= args.patience:
            logger.info(
                "Early stopping triggered (patience=%d)", args.patience
            )
            break

    # ------------------------------------------------------------------
    # 7. Save final model and training history
    # ------------------------------------------------------------------
    final_model_path = output_dir / "pose_model.pth"
    torch.save(model.state_dict(), str(final_model_path))
    logger.info("Final model saved to %s", final_model_path)

    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    logger.info(
        "Training complete. Best val_loss=%.6f, final MPJPE=%.4f, final PCK=%.4f",
        best_val_loss,
        history["val_mpjpe"][-1],
        history["val_pck"][-1],
    )


if __name__ == "__main__":
    main()
