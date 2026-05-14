"""
Model architecture definitions for all 7 models.
These are reused in both the backend (inference) and training scripts.

All models use torchvision pretrained backbones with replaced classifier heads.
For dummy-first pipeline: instantiate with weights=None (random weights).
For trained models: load weights from .pth files via model_registry.
"""
import torch.nn as nn
from torchvision import models


def create_efficientnet_b0(num_classes: int) -> nn.Module:
    """Create an EfficientNet-B0 with a custom classifier head."""
    model = models.efficientnet_b0(weights=None)
    # Replace the classifier: default is (dropout, Linear(1280, 1000))
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes),
    )
    return model


def create_resnet50(num_classes: int) -> nn.Module:
    """Create a ResNet50 with a custom fully-connected head."""
    model = models.resnet50(weights=None)
    # Replace the fc layer: default is Linear(2048, 1000)
    model.fc = nn.Linear(2048, num_classes)
    return model


# ── Factory: model_name → architecture builder ──
_ARCHITECTURE_MAP = {
    "efficientnet_b0": create_efficientnet_b0,
    "resnet50": create_resnet50,
}


def build_model(architecture: str, num_classes: int) -> nn.Module:
    """
    Build a model by architecture name.

    Args:
        architecture: 'efficientnet_b0' or 'resnet50'
        num_classes: number of output classes

    Returns:
        nn.Module with the correct architecture and output head
    """
    if architecture not in _ARCHITECTURE_MAP:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(_ARCHITECTURE_MAP.keys())}")
    return _ARCHITECTURE_MAP[architecture](num_classes)
