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

    def normalize(self, data: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
        """Z-score normalize per subcarrier (axis=0). Optionally subtract background."""
        if background is not None and data.shape[-1] == background.shape[-1]:
            # Simple background subtraction (static vector removal)
            data = data - background
            
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return ((data - mean) / std).astype(np.float32)

    def fuse_nodes(
        self,
        node_data: dict[int, np.ndarray],
        target_nodes: int | None = None,
        node_weights: dict[int, float] | None = None,
    ) -> np.ndarray:
        """Fuse amplitude vectors from multiple nodes into fixed-width feature vector.

        Args:
            node_data: {node_id: amplitude_vector}
            target_nodes: pad/truncate to this many nodes
            node_weights: {node_id: 0.0-1.0} signal quality weights.
                          Scales each node's contribution so weak nodes
                          don't pollute the fused signal.
        """
        sorted_ids = sorted(node_data.keys())
        arrays = []
        for nid in sorted_ids:
            arr = node_data[nid].astype(np.float32)
            if node_weights and nid in node_weights:
                arr = arr * node_weights[nid]
            arrays.append(arr)
        fused = np.concatenate(arrays)

        if target_nodes is not None:
            n_sub = self.settings.num_subcarriers
            expected_len = target_nodes * n_sub
            if len(fused) < expected_len:
                fused = np.pad(fused, (0, expected_len - len(fused)))
            elif len(fused) > expected_len:
                fused = fused[:expected_len]

        return fused

    def prepare_model_input(
        self,
        window: list[dict[int, np.ndarray]],
        fs: float | None = None,
        background_profile: dict[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Full pipeline: fuse per frame, stack, filter, normalize."""
        if fs is None:
            fs = self.settings.csi_sample_rate

        # 1. Fuse frames
        stacked = np.array([self.fuse_nodes(frame) for frame in window])
        
        # 2. Prepare background vector if available
        bg_vector = None
        if background_profile:
            # Fuse background profile same way as data
            # Check if all nodes in current data are present in background
            nodes_in_data = sorted(window[0].keys())
            if all(nid in background_profile for nid in nodes_in_data):
                bg_data = {nid: background_profile[nid] for nid in nodes_in_data}
                bg_vector = self.fuse_nodes(bg_data) # (total_subcarriers,)
                # Broadcast to window size (T, S)
                bg_vector = np.tile(bg_vector, (len(stacked), 1))

        # 3. Filter
        filtered = self.bandpass_filter(stacked, low=0.1, high=8.0, fs=fs)
        
        # 4. Normalize (with background subtraction if available)
        return self.normalize(filtered, background=bg_vector)
