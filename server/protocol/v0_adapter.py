"""Bidirectional conversion between v0 (legacy) and v1 protocol formats.

v0 format (single payload):
  {"joints": [...], "vitals": {...}, "csi_amplitudes": [...]}

v1 format (separate envelopes per stream type).
"""
from __future__ import annotations

from server.protocol.envelope import (
    Envelope, PoseData, VitalsData, CsiData,
)


def v1_to_v0(envelope: Envelope) -> dict:
    """Convert a v1 envelope to v0 legacy payload dict."""
    result = {}
    if envelope.type == "pose":
        result["joints"] = envelope.data.joints
    elif envelope.type == "vitals":
        result["vitals"] = envelope.data.model_dump()
    elif envelope.type == "csi":
        result["csi_amplitudes"] = envelope.data.amplitudes
    return result


def v0_to_v1_parts(payload: dict) -> dict[str, PoseData | VitalsData | CsiData]:
    """Split a v0 payload into separate v1 data objects."""
    parts = {}
    if "joints" in payload:
        parts["pose"] = PoseData(joints=payload["joints"])
    if "vitals" in payload and payload["vitals"]:
        parts["vitals"] = VitalsData(**payload["vitals"])
    if "csi_amplitudes" in payload and payload["csi_amplitudes"]:
        parts["csi"] = CsiData(amplitudes=payload["csi_amplitudes"])
    return parts
