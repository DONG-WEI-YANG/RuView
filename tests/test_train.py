"""Tests for the training pipeline (loss functions, metrics, training loop, checkpoints)."""
import numpy as np
import pytest
import tempfile
import torch
from pathlib import Path

from server.data_generator import SyntheticDataGenerator
from server.dataset import create_dataloaders
from server.pose_model import WiFiPoseModel
from server.train import (
    bone_length_loss,
    compute_mpjpe,
    compute_pck,
    mpjpe_loss,
    train_one_epoch,
    validate,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic data for fast tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_data_dir(tmp_path: Path) -> Path:
    """Generate a tiny synthetic dataset for training tests."""
    gen = SyntheticDataGenerator(seed=42)
    gen.generate_dataset(
        n_sequences_per_activity=2,
        n_frames=120,
        output_dir=str(tmp_path),
    )
    return tmp_path


@pytest.fixture()
def tiny_loaders(tiny_data_dir: Path):
    """Create tiny train/val dataloaders."""
    train_loader, val_loader = create_dataloaders(
        str(tiny_data_dir),
        window_size=10,
        batch_size=8,
        val_split=0.2,
        num_workers=0,
        stride=10,
    )
    return train_loader, val_loader


@pytest.fixture()
def tiny_model(tiny_loaders):
    """Create a small model whose input_dim matches the data."""
    train_loader, _ = tiny_loaders
    sample_csi, _ = next(iter(train_loader))
    input_dim = sample_csi.shape[-1]
    model = WiFiPoseModel(input_dim=input_dim, num_joints=24, hidden=32)
    return model


# ---------------------------------------------------------------------------
# 1. MPJPE loss: known input, verify correct L2 computation
# ---------------------------------------------------------------------------


class TestMPJPELoss:
    def test_identical_pred_target_gives_zero(self):
        """If prediction equals target, MPJPE should be zero."""
        target = torch.randn(4, 24, 3)
        loss = mpjpe_loss(target, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        """Manually computed MPJPE for a known offset."""
        # Batch=1, 24 joints, all shifted by (1, 0, 0) -> L2 = 1.0 per joint
        pred = torch.zeros(1, 24, 3)
        target = torch.ones(1, 24, 3) * 0.0
        target[:, :, 0] = 1.0  # offset only in x by 1.0
        loss = mpjpe_loss(pred, target)
        # L2 per joint = sqrt(1^2 + 0^2 + 0^2) = 1.0
        # Mean across joints and batch = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_batch_averaging(self):
        """MPJPE should be averaged across batch and joints."""
        # Batch=2: first sample offset by (3,4,0)=5.0, second by (0,0,0)=0.0
        pred = torch.zeros(2, 24, 3)
        target = torch.zeros(2, 24, 3)
        target[0, :, 0] = 3.0
        target[0, :, 1] = 4.0
        # sample 0: L2 = 5.0, sample 1: L2 = 0.0 -> mean = 2.5
        loss = mpjpe_loss(pred, target)
        assert loss.item() == pytest.approx(2.5, abs=1e-5)

    def test_gradient_flows(self):
        """Loss must be differentiable."""
        pred = torch.randn(2, 24, 3, requires_grad=True)
        target = torch.randn(2, 24, 3)
        loss = mpjpe_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


# ---------------------------------------------------------------------------
# 2. Bone length loss
# ---------------------------------------------------------------------------


class TestBoneLengthLoss:
    def test_returns_scalar(self):
        """bone_length_loss should return a scalar tensor."""
        pred = torch.randn(4, 24, 3)
        loss = bone_length_loss(pred)
        assert loss.dim() == 0  # scalar

    def test_non_negative(self):
        """bone_length_loss should be non-negative (variance-based)."""
        pred = torch.randn(4, 24, 3)
        loss = bone_length_loss(pred)
        assert loss.item() >= 0.0

    def test_constant_skeleton_low_loss(self):
        """If all samples have identical joint positions, bone variance = 0."""
        skeleton = torch.randn(1, 24, 3)
        pred = skeleton.expand(8, -1, -1)  # identical across batch
        loss = bone_length_loss(pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. PCK metric: verify percentage calculation with known distances
# ---------------------------------------------------------------------------


class TestPCKMetric:
    def test_perfect_prediction(self):
        """All joints within threshold -> PCK = 1.0."""
        target = torch.randn(4, 24, 3)
        pck = compute_pck(target, target, threshold=0.05)
        assert pck == pytest.approx(1.0, abs=1e-6)

    def test_all_wrong(self):
        """All joints far outside threshold -> PCK = 0.0."""
        pred = torch.zeros(2, 24, 3)
        target = torch.ones(2, 24, 3) * 100.0  # far away
        pck = compute_pck(pred, target, threshold=0.05)
        assert pck == pytest.approx(0.0, abs=1e-6)

    def test_half_correct(self):
        """Exactly half the joints within threshold."""
        pred = torch.zeros(1, 24, 3)
        target = torch.zeros(1, 24, 3)
        # First 12 joints: within threshold (distance=0)
        # Last 12 joints: outside threshold
        target[0, 12:, 0] = 10.0  # far away
        pck = compute_pck(pred, target, threshold=0.05)
        assert pck == pytest.approx(0.5, abs=1e-6)

    def test_threshold_boundary(self):
        """Joints exactly at threshold distance should count as correct."""
        pred = torch.zeros(1, 24, 3)
        target = torch.zeros(1, 24, 3)
        # All joints at exactly the threshold distance
        target[:, :, 0] = 0.05
        pck = compute_pck(pred, target, threshold=0.05)
        assert pck == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. compute_mpjpe metric
# ---------------------------------------------------------------------------


class TestComputeMPJPE:
    def test_returns_float(self):
        pred = torch.randn(4, 24, 3)
        target = torch.randn(4, 24, 3)
        result = compute_mpjpe(pred, target)
        assert isinstance(result, float)

    def test_zero_for_identical(self):
        target = torch.randn(4, 24, 3)
        assert compute_mpjpe(target, target) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        pred = torch.zeros(1, 24, 3)
        target = torch.zeros(1, 24, 3)
        target[:, :, 0] = 1.0
        # MPJPE = 1.0
        assert compute_mpjpe(pred, target) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 5. Training loop: verify loss decreases over a few steps
# ---------------------------------------------------------------------------


class TestTrainOneEpoch:
    def test_returns_average_loss(self, tiny_model, tiny_loaders):
        """train_one_epoch should return a positive average loss."""
        train_loader, _ = tiny_loaders
        device = torch.device("cpu")
        tiny_model.to(device)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        avg_loss = train_one_epoch(tiny_model, train_loader, optimizer, device)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0

    def test_loss_decreases_over_epochs(self, tiny_model, tiny_loaders):
        """Loss should decrease over a few epochs of training."""
        train_loader, _ = tiny_loaders
        device = torch.device("cpu")
        tiny_model.to(device)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        losses = []
        for _ in range(5):
            loss = train_one_epoch(tiny_model, train_loader, optimizer, device)
            losses.append(loss)

        # The loss at epoch 5 should be less than epoch 1
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_model_weights_change(self, tiny_model, tiny_loaders):
        """Model parameters should change after one epoch of training."""
        train_loader, _ = tiny_loaders
        device = torch.device("cpu")
        tiny_model.to(device)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        # Snapshot first param before training
        first_param = next(tiny_model.parameters()).clone()

        train_one_epoch(tiny_model, train_loader, optimizer, device)

        # Params should have changed
        new_first_param = next(tiny_model.parameters())
        assert not torch.equal(first_param, new_first_param)


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------


class TestValidate:
    def test_returns_metrics(self, tiny_model, tiny_loaders):
        """validate should return (avg_loss, mpjpe, pck)."""
        _, val_loader = tiny_loaders
        device = torch.device("cpu")
        tiny_model.to(device)
        avg_loss, mpjpe, pck = validate(tiny_model, val_loader, device)
        assert isinstance(avg_loss, float)
        assert isinstance(mpjpe, float)
        assert isinstance(pck, float)
        assert avg_loss > 0.0
        assert mpjpe > 0.0
        assert 0.0 <= pck <= 1.0

    def test_validate_does_not_change_weights(self, tiny_model, tiny_loaders):
        """Validation should not update model parameters."""
        _, val_loader = tiny_loaders
        device = torch.device("cpu")
        tiny_model.to(device)

        params_before = {
            name: p.clone() for name, p in tiny_model.named_parameters()
        }
        validate(tiny_model, val_loader, device)
        for name, p in tiny_model.named_parameters():
            assert torch.equal(params_before[name], p), (
                f"Parameter {name} changed during validation!"
            )


# ---------------------------------------------------------------------------
# 7. Checkpoint save and load
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_and_resume(self, tiny_model, tiny_loaders):
        """Train for a few epochs, save checkpoint, resume, verify continuity."""
        train_loader, val_loader = tiny_loaders
        device = torch.device("cpu")
        tiny_model.to(device)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        # Train 2 epochs
        for _ in range(2):
            train_one_epoch(tiny_model, train_loader, optimizer, device)
        _, mpjpe_before, _ = validate(tiny_model, val_loader, device)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            ckpt_path = f.name

        checkpoint = {
            "epoch": 2,
            "model_state_dict": tiny_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": 999.0,
        }
        torch.save(checkpoint, ckpt_path)

        # Create a fresh model with same architecture and load checkpoint
        sample_csi, _ = next(iter(train_loader))
        input_dim = sample_csi.shape[-1]
        new_model = WiFiPoseModel(input_dim=input_dim, num_joints=24, hidden=32)
        new_model.to(device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

        loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
        new_model.load_state_dict(loaded["model_state_dict"])
        new_optimizer.load_state_dict(loaded["optimizer_state_dict"])

        assert loaded["epoch"] == 2

        # Validate that the loaded model gives the same MPJPE
        _, mpjpe_after, _ = validate(new_model, val_loader, device)
        assert mpjpe_before == pytest.approx(mpjpe_after, abs=1e-5)

        # Clean up
        Path(ckpt_path).unlink(missing_ok=True)
