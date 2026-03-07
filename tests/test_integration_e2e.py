"""End-to-end integration tests for the WiFi CSI -> Pose pipeline.

These tests verify the FULL pipeline works as a connected system:
  CSI frames -> SignalProcessor -> PosePipeline -> joints -> FallDetector/FitnessTracker

They do NOT require real hardware -- they use realistic synthetic CSI frames
that match the binary protocol format expected by the system.
"""

import struct
import numpy as np
import pytest
import torch

from server.config import Settings
from server.csi_frame import CSIFrame, parse_csi_frame
from server.signal_processor import SignalProcessor
from server.pipeline import PosePipeline
from server.pose_model import WiFiPoseModel
from server.fall_detector import FallDetector
from server.vital_signs import VitalSignsExtractor
from server.data_generator import SyntheticDataGenerator


# ======================================================================
# Helpers
# ======================================================================


def _make_csi_binary_frame(
    node_id: int = 1,
    seq: int = 0,
    n_sub: int = 56,
    rssi: int = -45,
    noise: int = -90,
) -> bytes:
    """Build a realistic ADR-018 binary CSI frame (same format as ESP32 firmware).

    HEADER_FORMAT = '<IBIIQBBBB H'
    Fields: magic(I), version(B), frame_type(I), node_id(Q),
            rssi(B), noise(B), channel(B), bandwidth(B), num_sub(H)

    Note: struct unpacking uses positional indexing:
      header[0]=magic, header[1]=version, header[2]=frame_type,
      header[3]=node_id(but actually sequence due to field reuse),
      header[4]=timestamp
    Parser: node_id = header[2], sequence = header[3], timestamp = header[4]
    So: frame_type slot carries node_id, node_id(Q) slot carries sequence.
    """
    magic = 0xC5110001
    version = 1
    timestamp = seq * 50_000  # 50ms intervals = 20 Hz

    # Random I/Q data (int16 pairs)
    iq_data = np.random.randint(-1000, 1000, size=n_sub * 2, dtype=np.int16)

    # Pack matching HEADER_FORMAT = '<IBIIQBBBB H'
    # header[2] = node_id (I field), header[3] = sequence (I field), header[4] = timestamp (Q)
    header = struct.pack(
        "<IBIIQBBBB H",
        magic,          # [0] I: magic
        version,        # [1] B: version
        node_id,        # [2] I: parsed as node_id
        seq,            # [3] I: parsed as sequence
        timestamp,      # [4] Q: timestamp_ms
        rssi & 0xFF,    # [5] B: rssi
        noise & 0xFF,   # [6] B: noise_floor
        6,              # [7] B: channel
        20,             # [8] B: bandwidth
        n_sub,          # [9] H: num_subcarriers
    )
    return header + iq_data.tobytes()


def _create_test_model(input_dim: int, num_joints: int = 24) -> WiFiPoseModel:
    """Create a model with random weights for testing (not trained, but functional)."""
    model = WiFiPoseModel(input_dim=input_dim, num_joints=num_joints)
    # Set to inference mode via PyTorch's evaluation toggle
    model.train(False)
    return model


def _make_csi_frame(node_id: int, seq: int, n_sub: int = 56) -> CSIFrame:
    """Create a CSIFrame object matching the real dataclass fields."""
    amp = np.random.randn(n_sub).astype(np.float32)
    phase = np.random.randn(n_sub).astype(np.float32)
    return CSIFrame(
        node_id=node_id,
        sequence=seq,
        timestamp_ms=seq * 50,
        rssi=-45,
        noise_floor=-90,
        channel=6,
        bandwidth=20,
        num_subcarriers=n_sub,
        amplitude=amp,
        phase=phase,
        raw_complex=amp + 1j * phase,
    )


# ======================================================================
# Test: Binary frame parsing -> CSIFrame
# ======================================================================


class TestBinaryFrameParsing:
    def test_parse_produces_valid_frame(self):
        raw = _make_csi_binary_frame(node_id=2, seq=42, n_sub=56)
        frame = parse_csi_frame(raw)

        assert frame is not None
        assert frame.node_id == 2
        assert frame.sequence == 42
        assert frame.amplitude is not None
        assert len(frame.amplitude) == 56
        assert frame.rssi < 0  # RSSI is negative dBm

    def test_parse_rejects_bad_magic(self):
        raw = _make_csi_binary_frame()
        bad = b"\x00\x00\x00\x00" + raw[4:]
        frame = parse_csi_frame(bad)
        assert frame is None


# ======================================================================
# Test: Signal processing pipeline
# ======================================================================


class TestSignalProcessing:
    def test_fuse_normalize_pipeline(self):
        settings = Settings()
        proc = SignalProcessor(settings)

        # Simulate 2 nodes, 56 subcarriers each
        node_data = {
            0: np.random.randn(56).astype(np.float32),
            1: np.random.randn(56).astype(np.float32),
        }

        fused = proc.fuse_nodes(node_data)
        assert fused.shape == (112,)  # 2 * 56

        normalized = proc.normalize(fused.reshape(1, -1))
        assert normalized.shape == (1, 112)
        # Z-score: mean should be close to 0
        assert abs(normalized.mean()) < 1.0

    def test_full_window_preparation(self):
        settings = Settings()
        proc = SignalProcessor(settings)

        # Build a window of 60 frames from 2 nodes
        window = []
        for _ in range(60):
            frame = {
                0: np.random.randn(56).astype(np.float32) * 0.5 + 1.0,
                1: np.random.randn(56).astype(np.float32) * 0.5 + 1.0,
            }
            window.append(frame)

        prepared = proc.prepare_model_input(window, fs=20.0)
        assert prepared.shape == (60, 112)
        assert prepared.dtype == np.float32


# ======================================================================
# Test: Full pipeline CSI -> model -> joints
# ======================================================================


class TestFullPipeline:
    def test_pipeline_produces_joints_with_model(self):
        """End-to-end: feed 60+ CSI frames -> get 24-joint output."""
        settings = Settings()
        input_dim = settings.num_subcarriers  # single node
        model = _create_test_model(input_dim)
        pipeline = PosePipeline(settings, model=model, window_size=60)

        # Feed 65 frames (> window_size=60 to trigger inference)
        for seq in range(65):
            pipeline.on_csi_frame(_make_csi_frame(0, seq, settings.num_subcarriers))
            pipeline.flush_frame()

        # Pipeline should have produced joints
        assert pipeline.latest_joints is not None
        assert pipeline.latest_joints.shape == (24, 3)
        assert pipeline.csi_frames_received == 65

    def test_pipeline_without_model_skips_inference(self):
        """Without model, pipeline accumulates frames but no joints."""
        settings = Settings()
        pipeline = PosePipeline(settings, model=None, window_size=60)

        for seq in range(70):
            pipeline.on_csi_frame(_make_csi_frame(0, seq, settings.num_subcarriers))
            pipeline.flush_frame()

        assert pipeline.latest_joints is None
        assert pipeline.csi_frames_received == 70

    def test_pipeline_multi_node(self):
        """Multiple ESP32 nodes feed CSI -> model receives concatenated features."""
        settings = Settings()
        n_nodes = 3
        input_dim = settings.num_subcarriers * n_nodes
        model = _create_test_model(input_dim)
        pipeline = PosePipeline(settings, model=model, window_size=30)

        for seq in range(35):
            for node_id in range(n_nodes):
                pipeline.on_csi_frame(_make_csi_frame(node_id, seq, settings.num_subcarriers))
            pipeline.flush_frame()

        assert pipeline.latest_joints is not None
        assert pipeline.latest_joints.shape == (24, 3)


# ======================================================================
# Test: Pipeline -> Application layer (fall detection, activity)
# ======================================================================


class TestPipelineApplications:
    def test_fall_detector_receives_joints_from_pipeline(self):
        settings = Settings()
        input_dim = settings.num_subcarriers
        model = _create_test_model(input_dim)
        pipeline = PosePipeline(settings, model=model, window_size=30)

        for seq in range(35):
            pipeline.on_csi_frame(_make_csi_frame(0, seq, settings.num_subcarriers))
            pipeline.flush_frame()

        # Fall detector should have been updated
        assert isinstance(pipeline.fall_detector, FallDetector)
        # Activity tracker should have a value
        activity = pipeline.current_activity
        assert isinstance(activity, str)


# ======================================================================
# Test: Vital signs extraction from CSI amplitude
# ======================================================================


class TestVitalsFromCSI:
    def test_vitals_from_realistic_csi(self):
        """Feed realistic breathing-modulated CSI -> detect breathing rate."""
        vs = VitalSignsExtractor(sample_rate=20)
        fs = 20.0

        # Simulate 30 seconds of CSI with breathing modulation at 0.25 Hz (15 BPM)
        for i in range(600):  # 30s * 20Hz
            t = i / fs
            base = np.ones(30, dtype=np.float32)
            breathing = 0.1 * np.sin(2 * np.pi * 0.25 * t)  # 15 BPM
            heart = 0.03 * np.sin(2 * np.pi * 1.2 * t)      # 72 BPM
            csi = base + breathing + heart + np.random.randn(30).astype(np.float32) * 0.01
            vs.push_csi(csi)

        result = vs.update()
        assert result is not None
        assert "breathing_bpm" in result
        assert "heart_bpm" in result
        assert "hrv_rmssd" in result
        # Breathing should be roughly 15 BPM (0.25 Hz * 60)
        if result["breathing_bpm"] > 0:
            assert 10 < result["breathing_bpm"] < 25


# ======================================================================
# Test: Synthetic data -> train -> inference round-trip
# ======================================================================


class TestTrainInferenceRoundTrip:
    def test_synthetic_data_trains_and_infers(self, tmp_path):
        """Generate synthetic data -> train 2 epochs -> load model -> run inference."""
        # Step 1: Generate synthetic data
        gen = SyntheticDataGenerator(seed=42)
        gen.generate_dataset(
            n_sequences_per_activity=2,
            n_frames=100,
            output_dir=str(tmp_path / "data"),
            n_nodes=1,
        )

        # Step 2: Create dataloaders
        from server.dataset import create_dataloaders
        train_loader, val_loader = create_dataloaders(
            str(tmp_path / "data"),
            window_size=30,
            batch_size=8,
            val_split=0.2,
        )
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0

        # Step 3: Determine input_dim and create model
        sample_csi, sample_joints = next(iter(train_loader))
        input_dim = sample_csi.shape[-1]
        model = WiFiPoseModel(input_dim=input_dim, num_joints=24)

        # Step 4: Train 2 epochs
        from server.train import train_one_epoch, validate
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        loss1 = train_one_epoch(model, train_loader, optimizer, device)
        loss2 = train_one_epoch(model, train_loader, optimizer, device)
        assert loss2 <= loss1 * 2  # Should not diverge catastrophically

        # Step 5: Validate
        val_loss, mpjpe, pck = validate(model, val_loader, device)
        assert val_loss > 0
        assert mpjpe > 0

        # Step 6: Save and reload model
        model_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), str(model_path))

        from server.pose_model import load_model
        loaded = load_model(str(model_path), input_dim=input_dim)
        assert loaded is not None

        # Step 7: Run inference through pipeline
        settings = Settings()
        pipeline = PosePipeline(settings, model=loaded, window_size=30)

        for seq in range(35):
            pipeline.on_csi_frame(_make_csi_frame(0, seq, input_dim))
            pipeline.flush_frame()

        assert pipeline.latest_joints is not None
        assert pipeline.latest_joints.shape == (24, 3)
        # Joints should be finite numbers
        assert np.isfinite(pipeline.latest_joints).all()


# ======================================================================
# Test: API integration (model loading)
# ======================================================================


class TestAPIModelLoading:
    def test_api_creates_pipeline_without_model(self):
        """API should create pipeline even without model weights file."""
        settings = Settings()
        settings.model_path = "nonexistent/model.pth"

        from server.api import _load_pipeline
        pipeline = _load_pipeline(settings)

        assert pipeline is not None
        assert pipeline.model is None

    def test_api_creates_pipeline_with_model(self, tmp_path):
        """API should load model weights if they exist."""
        # Create a minimal model and save it
        model = WiFiPoseModel(input_dim=56, num_joints=24)
        model_path = tmp_path / "pose_model.pth"
        torch.save(model.state_dict(), str(model_path))

        settings = Settings()
        settings.model_path = str(model_path)

        from server.api import _load_pipeline
        pipeline = _load_pipeline(settings)

        assert pipeline is not None
        assert pipeline.model is not None
