"""Binary banana-vs-non-banana gate model architecture and loader.

The gate is intentionally separate from the 4-class ripeness model and is
trained to answer a single question: does this image contain a banana?

Class mapping:
0 = non_banana
1 = banana
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torchvision import transforms


CLASS_NAMES = ["non_banana", "banana"]
CLASS_TO_IDX = {"non_banana": 0, "banana": 1}


class BananaGateCNN(nn.Module):
    """Custom 4-layer CNN binary gate classifier."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class BananaGateMobileNetV3(nn.Module):
    """MobileNetV3-Small binary banana classifier.

    ImageNet pretrained backbone with custom binary classification head.
    Optimized for CPU inference (<15ms per image).
    Class 0 = non_banana, Class 1 = banana.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def create_gate_model(architecture: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    """Factory function for gate models."""
    arch_lower = architecture.lower()
    if "mobilenet" in arch_lower:
        return BananaGateMobileNetV3(num_classes=2, pretrained=pretrained)
    return BananaGateCNN(num_classes=2)


def build_gate_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and eval transforms for the banana gate."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def load_gate_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> Tuple[nn.Module, transforms.Compose, float, dict]:
    """Load gate checkpoint and return (model, transform, threshold, metadata)."""
    device = torch.device(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    architecture = checkpoint.get("model_architecture", "BananaGateCNN")
    if "MobileNet" in architecture:
        model = BananaGateMobileNetV3(num_classes=2, pretrained=False)
    else:
        model = BananaGateCNN(num_classes=2)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_size = checkpoint.get("image_size", checkpoint.get("input_size", 224))
    _, eval_transform = build_gate_transforms(image_size)
    threshold = float(checkpoint.get("threshold", 0.50))

    return model, eval_transform, threshold, checkpoint


def predict_binary_gate(image_tensor: torch.Tensor, model: nn.Module, device: torch.device | str = "cpu") -> float:
    """Return banana probability (class 1) for a preprocessed tensor."""
    device = torch.device(device)
    model.eval()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
    return float(probabilities[1].item())
