"""Image preprocessing for model inference."""
import torch
from PIL import Image

from app.config import INFERENCE_TRANSFORMS


def preprocess_image(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.

    Steps:
    1. Apply inference transforms (resize 256, center crop 224, normalize)
    2. Add batch dimension
    3. Move to target device (CPU or GPU)

    Args:
        pil_image: PIL Image in RGB mode
        device: torch device to move the tensor to

    Returns:
        torch.Tensor of shape (1, 3, 224, 224) on the target device
    """
    tensor = INFERENCE_TRANSFORMS(pil_image)
    tensor = tensor.unsqueeze(0)  # Add batch dim: (3, 224, 224) → (1, 3, 224, 224)
    tensor = tensor.to(device)
    return tensor
