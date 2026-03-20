import time
import pytest
from server.protocol.envelope import (
    Envelope, PoseData, VitalsData, CsiData, StatusData, ErrorData,
    HelloMessage, WelcomeMessage, PingMessage, PongMessage,
    make_envelope, parse_client_message,
)


def test_make_pose_envelope():
    joints = [[0.1, 0.2, 0.3]] * 24
    env = make_envelope("pose", PoseData(joints=joints, confidence=0.92))
    assert env.v == 1
    assert env.type == "pose"
    assert env.seq > 0
    assert env.ts > 0
    assert env.data.joints == joints
    assert env.data.confidence == 0.92


def test_make_vitals_envelope():
    data = VitalsData(
        heart_bpm=72.0, heart_confidence=0.85,
        breathing_bpm=16.0, breathing_confidence=0.90,
        hrv_rmssd=45.0, hrv_sdnn=55.0,
        stress_index=30.0, motion_intensity=5.0,
        body_movement="still", breath_regularity=0.85,
        sleep_stage="awake", respiratory_distress=False,
        apnea_events=0,
    )
    env = make_envelope("vitals", data)
    assert env.type == "vitals"
    assert env.data.heart_bpm == 72.0


def test_make_csi_envelope():
    amps = [0.5] * 56
    env = make_envelope("csi", CsiData(amplitudes=amps))
    assert env.type == "csi"
    assert len(env.data.amplitudes) == 56


def test_make_status_envelope():
    data = StatusData(
        model_loaded=True, csi_frames_received=100,
        inference_active=True, is_simulating=False,
        connected_clients=2, hardware_profile="esp32s3",
    )
    env = make_envelope("status", data)
    assert env.data.model_loaded is True


def test_make_error_envelope():
    env = make_envelope("error", ErrorData(code="INFERENCE_FAIL", message="Model error"))
    assert env.data.code == "INFERENCE_FAIL"


def test_envelope_to_json():
    env = make_envelope("pose", PoseData(joints=[[0, 0, 0]] * 24, confidence=0.5))
    j = env.model_dump_json()
    assert '"v": 1' in j or '"v":1' in j


def test_seq_increments():
    e1 = make_envelope("csi", CsiData(amplitudes=[1.0]))
    e2 = make_envelope("csi", CsiData(amplitudes=[2.0]))
    assert e2.seq > e1.seq


def test_parse_hello():
    msg = parse_client_message({"v": 1, "type": "hello", "capabilities": ["pose", "vitals"]})
    assert isinstance(msg, HelloMessage)
    assert "pose" in msg.capabilities


def test_parse_pong():
    msg = parse_client_message({"v": 1, "type": "pong", "ts": 12345})
    assert isinstance(msg, PongMessage)


def test_parse_unknown_returns_none():
    msg = parse_client_message({"v": 1, "type": "unknown_type"})
    assert msg is None


def test_parse_v0_returns_none():
    msg = parse_client_message({"joints": [[0, 0, 0]]})
    assert msg is None
