"""Tests for the synthetic CSI + pose data generator."""
import numpy as np
import pytest
import os
import tempfile
import shutil

from server.data_generator import SyntheticDataGenerator


@pytest.fixture
def gen():
    return SyntheticDataGenerator(seed=42)


# ---------- Shape tests ----------


class TestGenerateSequenceShapes:
    """generate_sequence must return arrays with correct dtypes and shapes."""

    def test_standing_default_shape(self, gen):
        result = gen.generate_sequence("standing", n_frames=100)
        assert result["csi"].shape == (100, 4, 56)
        assert result["joints"].shape == (100, 24, 3)
        assert result["labels"].shape == (100,)

    def test_custom_nodes_and_subcarriers(self, gen):
        result = gen.generate_sequence("walking", n_frames=50, n_nodes=3, n_sub=64)
        assert result["csi"].shape == (50, 3, 64)
        assert result["joints"].shape == (50, 24, 3)

    def test_dtypes(self, gen):
        result = gen.generate_sequence("standing", n_frames=10)
        assert result["csi"].dtype == np.float32
        assert result["joints"].dtype == np.float32

    def test_labels_content(self, gen):
        for activity in ["standing", "walking", "sitting", "falling", "exercising"]:
            result = gen.generate_sequence(activity, n_frames=20)
            assert all(label == activity for label in result["labels"])

    def test_all_activities_produce_output(self, gen):
        for activity in ["standing", "walking", "sitting", "falling", "exercising"]:
            result = gen.generate_sequence(activity, n_frames=30)
            assert result["csi"].shape[0] == 30
            assert result["joints"].shape[0] == 30


# ---------- CSI varies with pose ----------


class TestCSIVariesWithPose:
    """Different poses must produce meaningfully different CSI signals."""

    def test_standing_vs_sitting_csi_differs(self, gen):
        standing = gen.generate_sequence("standing", n_frames=60)
        sitting = gen.generate_sequence("sitting", n_frames=60)

        # Compare mean CSI across the last 30 frames (after sitting transition)
        standing_mean = standing["csi"][30:].mean(axis=0)
        sitting_mean = sitting["csi"][30:].mean(axis=0)

        # The mean CSI should differ noticeably
        diff = np.abs(standing_mean - sitting_mean).mean()
        assert diff > 0.01, (
            f"Standing vs sitting CSI difference too small: {diff}"
        )

    def test_standing_vs_falling_csi_differs(self, gen):
        standing = gen.generate_sequence("standing", n_frames=60)
        falling = gen.generate_sequence("falling", n_frames=60)

        standing_mean = standing["csi"][30:].mean(axis=0)
        falling_mean = falling["csi"][30:].mean(axis=0)

        diff = np.abs(standing_mean - falling_mean).mean()
        assert diff > 0.01, (
            f"Standing vs falling CSI difference too small: {diff}"
        )

    def test_csi_changes_over_time_for_walking(self, gen):
        result = gen.generate_sequence("walking", n_frames=100)
        # Walking CSI should change between early and late frames
        early = result["csi"][:20].mean(axis=0)
        late = result["csi"][60:80].mean(axis=0)
        diff = np.abs(early - late).mean()
        assert diff > 0.001, "Walking CSI should vary over time"


# ---------- Dataset generation ----------


class TestGenerateDataset:
    """generate_dataset must write well-formed .npz files to disk."""

    def test_creates_npz_files(self, gen):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen.generate_dataset(
                n_sequences_per_activity=2,
                n_frames=30,
                output_dir=tmpdir,
            )
            npz_files = [f for f in os.listdir(tmpdir) if f.endswith(".npz")]
            # 5 activities x 2 sequences = 10 files
            assert len(npz_files) == 10

    def test_npz_contents(self, gen):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen.generate_dataset(
                n_sequences_per_activity=1,
                n_frames=25,
                output_dir=tmpdir,
            )
            npz_files = [f for f in os.listdir(tmpdir) if f.endswith(".npz")]
            assert len(npz_files) > 0

            # Note: allow_pickle=True is required to load string label arrays
            # from our own generated .npz files (not untrusted data).
            # Explicitly close to release the file handle (Windows file locking).
            data = np.load(os.path.join(tmpdir, npz_files[0]), allow_pickle=True)
            try:
                assert "csi" in data
                assert "joints" in data
                assert "labels" in data
                assert data["csi"].shape[0] == 25
                assert data["joints"].shape[0] == 25
                assert data["labels"].shape[0] == 25
            finally:
                data.close()


# ---------- Joint position bounds ----------


class TestJointBounds:
    """Joint positions must stay within physically reasonable bounds."""

    def test_height_within_bounds(self, gen):
        """All joints should stay between floor (0m) and ~2m height."""
        for activity in ["standing", "walking", "sitting", "falling", "exercising"]:
            result = gen.generate_sequence(activity, n_frames=100)
            joints = result["joints"]
            # Y-axis is height: allow small margin below 0 and above 2
            assert joints[:, :, 1].min() >= -0.5, (
                f"{activity}: joint height below -0.5m"
            )
            assert joints[:, :, 1].max() <= 2.5, (
                f"{activity}: joint height above 2.5m"
            )

    def test_horizontal_within_room(self, gen):
        """X and Z positions should stay within a reasonable room (~5m radius)."""
        for activity in ["standing", "walking", "sitting", "falling", "exercising"]:
            result = gen.generate_sequence(activity, n_frames=100)
            joints = result["joints"]
            assert np.abs(joints[:, :, 0]).max() <= 5.0, (
                f"{activity}: x-position out of room bounds"
            )
            assert np.abs(joints[:, :, 2]).max() <= 5.0, (
                f"{activity}: z-position out of room bounds"
            )

    def test_no_nan_or_inf(self, gen):
        """Output arrays should never contain NaN or Inf."""
        for activity in ["standing", "walking", "sitting", "falling", "exercising"]:
            result = gen.generate_sequence(activity, n_frames=50)
            assert np.all(np.isfinite(result["csi"])), f"{activity}: CSI has NaN/Inf"
            assert np.all(np.isfinite(result["joints"])), (
                f"{activity}: joints have NaN/Inf"
            )


# ---------- Reproducibility ----------


class TestReproducibility:
    """Fixed seed must produce identical sequences."""

    def test_same_seed_same_output(self):
        gen1 = SyntheticDataGenerator(seed=123)
        gen2 = SyntheticDataGenerator(seed=123)
        r1 = gen1.generate_sequence("walking", n_frames=50)
        r2 = gen2.generate_sequence("walking", n_frames=50)
        np.testing.assert_array_equal(r1["csi"], r2["csi"])
        np.testing.assert_array_equal(r1["joints"], r2["joints"])

    def test_different_seed_different_output(self):
        gen1 = SyntheticDataGenerator(seed=1)
        gen2 = SyntheticDataGenerator(seed=2)
        r1 = gen1.generate_sequence("walking", n_frames=50)
        r2 = gen2.generate_sequence("walking", n_frames=50)
        assert not np.allclose(r1["csi"], r2["csi"])
