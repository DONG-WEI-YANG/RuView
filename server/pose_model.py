"""WiFi CSI to 24-joint pose estimation model."""
import torch
import torch.nn as nn


class WiFiPoseModel(nn.Module):
    """Maps a window of CSI features to 24 body joint coordinates.
    Architecture: 1D Conv encoder -> attention -> FC decoder -> 24 joints x 3 (xyz).
    """

    def __init__(self, input_dim: int, num_joints: int = 24, hidden: int = 256):
        super().__init__()
        self.num_joints = num_joints

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_joints * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: (batch, time_steps, input_dim)
        Returns: joints: (batch, num_joints, 3)
        """
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        encoded_t = encoded.transpose(1, 2)
        attn_weights = self.attention(encoded_t)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (encoded_t * attn_weights).sum(dim=1)
        out = self.decoder(pooled)
        return out.view(-1, self.num_joints, 3)


def load_model(path: str, input_dim: int, device: str = "cpu") -> WiFiPoseModel:
    model = WiFiPoseModel(input_dim=input_dim)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
