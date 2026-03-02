"""Tests for WiFiPoseDataset and create_dataloaders."""
import numpy as np
import torch
import pytest
from pathlib import Path

from server.dataset import WiFiPoseDataset, create_dataloaders


# ---------------------------------------------------------------------------
# Helpers: create synthetic .npz files that mimic real paired data
# ---------------------------------------------------------------------------

N_NODES = 4
N_SUB = 56
N_JOINTS = 24


def _make_npz(path: Path, n_frames: int = 120, seed: int = 0) -> None:
    """Write a fake .npz with CSI, joints, and labels."""
    rng = np.random.RandomState(seed)
    csi = rng.randn(n_frames, N_NODES, N_SUB).astype(np.float32)
    joints = rng.randn(n_frames, N_JOINTS, 3).astype(np.float32)
    labels = np.array(["walk"] * n_frames)
    np.savez(path, csi=csi, joints=joints, labels=labels)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with several .npz files."""
    for i in range(5):
        _make_npz(tmp_path / f"seq_{i:02d}.npz", n_frames=120, seed=i)
    return tmp_path


@pytest.fixture()
def small_data_dir(tmp_path: Path) -> Path:
    """A single short file for edge-case tests."""
    _make_npz(tmp_path / "short.npz", n_frames=30, seed=99)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Dataset loads .npz files and reports correct length
# ---------------------------------------------------------------------------


class TestDatasetLength:
    def test_correct_total_length(self, data_dir: Path):
        ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=1)
        # Each file has 120 frames -> (120 - 10) / 1 + 1 = 111 windows per file
        # 5 files -> 555 total
        assert len(ds) == 5 * 111

    def test_stride_reduces_length(self, data_dir: Path):
        ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=5)
        # Each file: (120 - 10) // 5 + 1 = 23 windows per file
        # 5 files -> 115
        assert len(ds) == 5 * 23

    def test_window_equals_frames(self, small_data_dir: Path):
        ds = WiFiPoseDataset(str(small_data_dir), window_size=30, stride=1)
        # 30 frames, window=30 -> exactly 1 sample
        assert len(ds) == 1

    def test_empty_dir_gives_zero(self, tmp_path: Path):
        ds = WiFiPoseDataset(str(tmp_path), window_size=10)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# 2. __getitem__ returns correct tensor shapes
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_shapes(self, data_dir: Path):
        ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=1)
        csi_window, target_joints = ds[0]
        assert isinstance(csi_window, torch.Tensor)
        assert isinstance(target_joints, torch.Tensor)
        assert csi_window.shape == (10, N_NODES * N_SUB)
        assert target_joints.shape == (N_JOINTS, 3)

    def test_dtype_is_float32(self, data_dir: Path):
        ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=1)
        csi_window, target_joints = ds[0]
        assert csi_window.dtype == torch.float32
        assert target_joints.dtype == torch.float32

    def test_last_sample_is_valid(self, data_dir: Path):
        ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=1)
        csi_window, target_joints = ds[len(ds) - 1]
        assert csi_window.shape == (10, N_NODES * N_SUB)
        assert target_joints.shape == (N_JOINTS, 3)

    def test_target_is_last_frame_of_window(self, data_dir: Path):
        """The target joints must correspond to the LAST frame in the window."""
        ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=1)
        # The first sample covers frames 0..9 of the first file.
        # Target should be joints[9].
        _, target = ds[0]

        # Load the raw file to verify
        files = sorted(Path(data_dir).glob("*.npz"))
        raw = np.load(files[0])
        expected = torch.from_numpy(raw["joints"][9])
        assert torch.allclose(target, expected)

    def test_index_out_of_range(self, data_dir: Path):
        ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=1)
        with pytest.raises(IndexError):
            ds[len(ds)]


# ---------------------------------------------------------------------------
# 3. DataLoader works with batching
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_batching(self, data_dir: Path):
        ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=5)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        batch_csi, batch_joints = next(iter(loader))
        assert batch_csi.shape == (8, 10, N_NODES * N_SUB)
        assert batch_joints.shape == (8, N_JOINTS, 3)

    def test_create_dataloaders_returns_two(self, data_dir: Path):
        train_loader, val_loader = create_dataloaders(
            str(data_dir),
            window_size=10,
            batch_size=4,
            val_split=0.4,
            num_workers=0,
        )
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)

    def test_create_dataloaders_covers_all_files(self, data_dir: Path):
        """Train + val datasets should together cover all .npz files."""
        train_loader, val_loader = create_dataloaders(
            str(data_dir),
            window_size=10,
            batch_size=4,
            val_split=0.4,
            num_workers=0,
        )
        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        full_ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=1)
        assert total_samples == len(full_ds)


# ---------------------------------------------------------------------------
# 4. Train/val split doesn't share files
# ---------------------------------------------------------------------------


class TestSplitIntegrity:
    def test_no_file_overlap(self, data_dir: Path):
        """Train and val datasets must use disjoint sets of .npz files."""
        train_loader, val_loader = create_dataloaders(
            str(data_dir),
            window_size=10,
            batch_size=4,
            val_split=0.4,
            num_workers=0,
        )
        train_files = set(train_loader.dataset.files)
        val_files = set(val_loader.dataset.files)
        assert len(train_files & val_files) == 0, "Train and val share files!"

    def test_all_files_used(self, data_dir: Path):
        """Every .npz file should appear in either train or val."""
        all_npz = set(sorted(Path(data_dir).glob("*.npz")))
        train_loader, val_loader = create_dataloaders(
            str(data_dir),
            window_size=10,
            batch_size=4,
            val_split=0.4,
            num_workers=0,
        )
        used = set(train_loader.dataset.files) | set(val_loader.dataset.files)
        assert used == all_npz


# ---------------------------------------------------------------------------
# 5. Augmentation changes the CSI data but not the joints
# ---------------------------------------------------------------------------


class TestAugmentation:
    def test_augmented_csi_differs(self, data_dir: Path):
        """With augmentation enabled, CSI values should differ across reads."""
        ds_aug = WiFiPoseDataset(str(data_dir), window_size=10, stride=1, augment=True)

        # Read the same sample twice; stochastic augmentation should differ.
        csi_a, _ = ds_aug[0]
        csi_b, _ = ds_aug[0]
        assert not torch.equal(csi_a, csi_b), "Augmented CSI should be stochastic"

    def test_augmentation_preserves_joints(self, data_dir: Path):
        """Joints should NOT be augmented."""
        ds_aug = WiFiPoseDataset(str(data_dir), window_size=10, stride=1, augment=True)
        _, joints_a = ds_aug[0]
        _, joints_b = ds_aug[0]
        assert torch.equal(joints_a, joints_b), "Joints should be identical"

    def test_no_augmentation_is_deterministic(self, data_dir: Path):
        ds = WiFiPoseDataset(str(data_dir), window_size=10, stride=1, augment=False)
        csi_a, joints_a = ds[0]
        csi_b, joints_b = ds[0]
        assert torch.equal(csi_a, csi_b)
        assert torch.equal(joints_a, joints_b)
