# tests/test_protocol_v0_adapter.py
import pytest
from server.protocol.v0_adapter import v1_to_v0, v0_to_v1_parts
from server.protocol.envelope import (
    make_envelope, PoseData, VitalsData, CsiData,
)


def test_v1_pose_to_v0():
    joints = [[i * 0.1, i * 0.2, i * 0.3] for i in range(24)]
    env = make_envelope("pose", PoseData(joints=joints, confidence=0.9))
    v0 = v1_to_v0(env)
    assert "joints" in v0
    assert len(v0["joints"]) == 24
    assert v0["joints"][0] == joints[0]


def test_v1_vitals_merged_into_v0():
    vitals_data = VitalsData(heart_bpm=72.0, breathing_bpm=16.0)
    env = make_envelope("vitals", vitals_data)
    v0 = v1_to_v0(env)
    assert v0["vitals"]["heart_bpm"] == 72.0


def test_v1_csi_merged_into_v0():
    env = make_envelope("csi", CsiData(amplitudes=[0.5] * 30))
    v0 = v1_to_v0(env)
    assert len(v0["csi_amplitudes"]) == 30


def test_v0_payload_to_v1_parts():
    v0_payload = {
        "joints": [[0, 0, 0]] * 24,
        "vitals": {"heart_bpm": 72, "breathing_bpm": 16},
        "csi_amplitudes": [0.5] * 30,
    }
    parts = v0_to_v1_parts(v0_payload)
    assert "pose" in parts
    assert "vitals" in parts
    assert "csi" in parts
    assert parts["pose"].joints == [[0, 0, 0]] * 24


def test_v0_payload_missing_vitals():
    v0_payload = {"joints": [[0, 0, 0]] * 24}
    parts = v0_to_v1_parts(v0_payload)
    assert "pose" in parts
    assert "vitals" not in parts
    assert "csi" not in parts
