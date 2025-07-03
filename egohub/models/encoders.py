import torch.nn as nn

class ImageEncoder(nn.Module):
    """
    A simple convolutional encoder for RGB images.
    Takes an image and returns a flattened feature vector.
    """
    def __init__(self, in_channels: int = 3, out_features: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # (32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (64, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (128, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # (256, H/16, W/16)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # Ensures the output is always 4x4
            nn.Flatten(),
        )
        # The input to the linear layer is now fixed at 256 * 4 * 4
        self.final_layer = nn.Linear(256 * 4 * 4, out_features)

    def forward(self, x):
        x = self.encoder(x)
        return self.final_layer(x)

class PoseEncoder(nn.Module):
    """
    A simple MLP encoder for 3D pose data.
    Takes a flattened pose vector and returns a feature vector.
    """
    def __init__(self, in_features: int, out_features: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        # The input x is expected to be already flattened
        return self.encoder(x) 