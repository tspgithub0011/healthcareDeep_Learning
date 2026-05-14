"""
Central training configuration.
Shared hyperparameters, paths, and device selection for all training scripts.
"""
import os
import torch
import random
import numpy as np

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "trained_models")

# Ensure output directory exists
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

# ── Device Selection ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"🚀 Using GPU: {GPU_NAME}")
else:
    GPU_NAME = None
    print("⚠️  Using CPU (training will be slow)")

# ── Shared Training Hyperparameters ──
TRAINING_CONFIG = {
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "loss_function": "CrossEntropyLoss",
    "scheduler": "ReduceLROnPlateau",
    "scheduler_patience": 3,
    "scheduler_factor": 0.5,
    "epochs": 50,
    "batch_size": 16,           # RTX 3050 (4GB VRAM) safe default
    "early_stopping_patience": 7,
    "input_size": (224, 224),
    "seed": 42,
    "data_split": {"train": 0.7, "val": 0.15, "test": 0.15},
    "num_workers": 2,           # Windows-safe default
    "pin_memory": True,
    "use_mixed_precision": True, # RTX 3050 supports AMP
}

# ── ImageNet Normalization ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(architecture: str, num_classes: int, pretrained: bool = True):
    """
    Build a model with optional pretrained ImageNet weights.

    Args:
        architecture: 'efficientnet_b0' or 'resnet50'
        num_classes: number of output classes
        pretrained: if True, use ImageNet pretrained weights (transfer learning)

    Returns:
        nn.Module with the correct architecture and output head
    """
    from torchvision import models

    if architecture == "efficientnet_b0":
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        # Replace classifier head
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(1280, num_classes),
        )

    elif architecture == "resnet50":
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
        # Replace fc head
        model.fc = torch.nn.Linear(2048, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model
