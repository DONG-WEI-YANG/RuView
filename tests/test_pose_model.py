import torch
import pytest
from server.pose_model import WiFiPoseModel


def test_model_output_shape():
    model = WiFiPoseModel(input_dim=56 * 4, num_joints=24)
    x = torch.randn(1, 60, 56 * 4)
    joints = model(x)
    assert joints.shape == (1, 24, 3)


def test_model_deterministic():
    model = WiFiPoseModel(input_dim=56 * 3, num_joints=24)
    model.eval()
    x = torch.randn(1, 60, 56 * 3)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.allclose(y1, y2)


def test_model_different_node_counts():
    for n_nodes in [2, 3, 4, 6]:
        model = WiFiPoseModel(input_dim=56 * n_nodes, num_joints=24)
        x = torch.randn(2, 40, 56 * n_nodes)
        y = model(x)
        assert y.shape == (2, 24, 3)
