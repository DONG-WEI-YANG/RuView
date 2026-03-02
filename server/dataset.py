"""PyTorch Dataset for paired WiFi CSI + pose data."""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List


class WiFiPoseDataset(Dataset):
    """Dataset for paired CSI + pose data.

    Loads .npz files and creates sliding windows of CSI data
    paired with the joint positions at the end of each window.

    Each .npz file must contain:
        - ``csi``:    (n_frames, n_nodes, n_sub) float32
        - ``joints``: (n_frames, 24, 3) float32
        - ``labels``: (n_frames,) - activity label strings
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 60,
        stride: int = 1,
        augment: bool = False,
        *,
        files: List[Path] | None = None,
    ):
        """
        Args:
            data_dir: Directory containing .npz files.
            window_size: Number of CSI frames per sample.
            stride: Step size between consecutive windows.
            augment: Whether to apply data augmentation (Gaussian noise
                     on CSI + random time shift).
            files: If given, use only these .npz files instead of scanning
                   *data_dir*.  Used internally by :func:`create_dataloaders`
                   to split by file.
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.augment = augment

        # Discover or accept file list
        if files is not None:
            self.files: List[Path] = sorted(files)
        else:
            self.files = sorted(self.data_dir.glob("*.npz"))

        # Pre-load every file and build an index that maps a flat integer
        # index to (file_idx, start_frame).
        self._csi: List[np.ndarray] = []      # each: (n_frames, n_nodes*n_sub)
        self._joints: List[np.ndarray] = []   # each: (n_frames, 24, 3)
        self._index: List[tuple[int, int]] = []  # (file_idx, start_frame)

        for file_idx, fpath in enumerate(self.files):
            data = np.load(fpath)
            csi = data["csi"].astype(np.float32)      # (T, n_nodes, n_sub)
            joints = data["joints"].astype(np.float32)  # (T, 24, 3)

            n_frames = csi.shape[0]
            # Flatten the spatial dimensions: (T, n_nodes, n_sub) -> (T, n_nodes*n_sub)
            csi_flat = csi.reshape(n_frames, -1)

            self._csi.append(csi_flat)
            self._joints.append(joints)

            # Sliding window indices for this file
            for start in range(0, n_frames - window_size + 1, stride):
                self._index.append((file_idx, start))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(csi_window, target_joints)``.

        ``csi_window``:    shape ``(window_size, n_nodes * n_sub)``
        ``target_joints``: shape ``(24, 3)`` -- joints at the **last** frame
                           of the window.
        """
        if idx < 0 or idx >= len(self._index):
            raise IndexError(
                f"index {idx} out of range for dataset with {len(self)} samples"
            )

        file_idx, start = self._index[idx]
        end = start + self.window_size

        csi_window = self._csi[file_idx][start:end].copy()   # (W, D)
        target_joints = self._joints[file_idx][end - 1].copy()  # (24, 3)

        # ----- augmentation -----
        if self.augment:
            csi_window = self._augment_csi(csi_window)

        return (
            torch.from_numpy(csi_window),
            torch.from_numpy(target_joints),
        )

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_csi(csi: np.ndarray) -> np.ndarray:
        """Apply stochastic augmentations to a CSI window.

        * Additive Gaussian noise  (std ~ 1 % of per-window std)
        * Random circular time shift of up to 3 frames
        """
        # Gaussian noise
        std = csi.std() * 0.01 + 1e-8
        csi = csi + np.random.randn(*csi.shape).astype(np.float32) * std

        # Random time shift (circular roll)
        max_shift = min(3, csi.shape[0] // 2)
        if max_shift > 0:
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift != 0:
                csi = np.roll(csi, shift, axis=0)

        return csi


# ======================================================================
# DataLoader factory
# ======================================================================


def create_dataloaders(
    data_dir: str,
    window_size: int = 60,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    stride: int = 1,
    augment_train: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    The split is performed **at the file level** (not per-sample) so that
    consecutive frames from the same recording never leak across the
    train/val boundary.

    Args:
        data_dir: Directory containing .npz files.
        window_size: Sliding window length.
        batch_size: Mini-batch size.
        val_split: Fraction of *files* reserved for validation.
        num_workers: DataLoader worker processes.
        stride: Sliding window stride.
        augment_train: Enable augmentation for the training set.

    Returns:
        ``(train_loader, val_loader)``
    """
    all_files = sorted(Path(data_dir).glob("*.npz"))

    n_val = max(1, int(len(all_files) * val_split))
    n_train = len(all_files) - n_val

    train_files = all_files[:n_train]
    val_files = all_files[n_train:]

    train_ds = WiFiPoseDataset(
        data_dir,
        window_size=window_size,
        stride=stride,
        augment=augment_train,
        files=train_files,
    )
    val_ds = WiFiPoseDataset(
        data_dir,
        window_size=window_size,
        stride=stride,
        augment=False,
        files=val_files,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
