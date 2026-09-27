import torch
from torch import nn


class BananaCNN(nn.Module):
    """A small CNN intentionally designed for learning.

    This model is NOT pretrained.

    The network learns visual patterns directly from the banana dataset:
    edges -> textures -> color/shape patterns -> class decision.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()

        # Block 1: learn low-level image features such as edges.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: learn slightly more complex textures.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: learn higher-level banana appearance features.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: make the representation more expressive.
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Dropout reduces overfitting, which matters because the
        # user's dataset is relatively small.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]
