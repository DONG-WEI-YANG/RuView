"""CSI signal processing: filtering, normalization, multi-node fusion."""
import numpy as np
from scipy.signal import butter, filtfilt

from server.config import Settings


class SignalProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def bandpass_filter(
        self, data: np.ndarray, low: float, high: float, fs: float, order: int = 4
    ) -> np.ndarray:
        """Apply bandpass Butterworth filter along time axis (axis=0)."""
        nyq = fs / 2.0
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, data, axis=0).astype(np.float32)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalize per subcarrier (axis=0)."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return ((data - mean) / std).astype(np.float32)

    def fuse_nodes(self, node_data: dict[int, np.ndarray]) -> np.ndarray:
        """Concatenate amplitude vectors from multiple nodes."""
        arrays = [node_data[nid] for nid in sorted(node_data.keys())]
        return np.concatenate(arrays).astype(np.float32)

    def prepare_model_input(
        self,
        window: list[dict[int, np.ndarray]],
        fs: float | None = None,
    ) -> np.ndarray:
        """Full pipeline: fuse per frame, stack, filter, normalize."""
        if fs is None:
            fs = self.settings.csi_sample_rate

        stacked = np.array([self.fuse_nodes(frame) for frame in window])
        filtered = self.bandpass_filter(stacked, low=0.1, high=8.0, fs=fs)
        return self.normalize(filtered)
