"""
Grad-CAM — real heatmap generation from the last convolutional layer.

Implementation follows §8 of the final_plan.md.
Works with both EfficientNet-B0 and ResNet50 architectures.
"""
import io
import base64

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.services.model_registry import registry
from app.config import MODEL_CONFIG
from app.utils.logger import logger


class GradCAM:
    """Generate Grad-CAM heatmaps from a model's last convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: shape (1, 3, 224, 224) — must have requires_grad=False
            class_idx: target class index

        Returns:
            numpy array of shape (H, W) with values in [0, 1]
        """
        # Enable gradients temporarily for Grad-CAM
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()

        # Backward pass from the target class
        output[0, class_idx].backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # Global average pooling of gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)  # Normalize to [0, 1]

        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def _get_target_layer(model: torch.nn.Module, architecture: str) -> torch.nn.Module:
    """Get the last convolutional layer for Grad-CAM based on architecture."""
    if architecture == "efficientnet_b0":
        # EfficientNet: features[-1] is the last block
        return model.features[-1]
    elif architecture == "resnet50":
        # ResNet: layer4 is the last residual block
        return model.layer4
    else:
        raise ValueError(f"Unknown architecture for Grad-CAM: {architecture}")


def _apply_colormap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Apply a colormap to a heatmap and resize.

    Args:
        heatmap: numpy array (H, W) with values in [0, 1]
        size: target (width, height)

    Returns:
        RGB numpy array of shape (height, width, 3) with uint8 values
    """
    # Resize heatmap to image size
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize(size, Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil).astype(np.float32) / 255.0

    # Apply a "jet"-like colormap manually (red = hot, blue = cold)
    r = np.clip(1.5 - np.abs(4 * heatmap_resized - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heatmap_resized - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heatmap_resized - 1), 0, 1)

    colormap = np.stack([r, g, b], axis=-1)
    return (colormap * 255).astype(np.uint8)


def generate_gradcam(
    model_name: str,
    input_tensor: torch.Tensor,
    class_idx: int,
    original_image: Image.Image,
    alpha: float = 0.4,
) -> dict:
    """
    Generate a Grad-CAM heatmap overlay and return as base64.

    Args:
        model_name: Name of the model in the registry
        input_tensor: Preprocessed tensor (1, 3, 224, 224)
        class_idx: Target class index for Grad-CAM
        original_image: Original PIL Image for overlay
        alpha: Heatmap overlay opacity

    Returns:
        dict: {image: "data:image/png;base64,...", model_used: "pneumonia"}
    """
    try:
        model = registry.get_model(model_name)
        architecture = MODEL_CONFIG[model_name]["architecture"]
        target_layer = _get_target_layer(model, architecture)

        # Need to temporarily enable gradients
        model.eval()

        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(input_tensor, class_idx)
        gradcam.remove_hooks()

        # Apply colormap
        img_size = original_image.size  # (width, height)
        colored_heatmap = _apply_colormap(heatmap, img_size)

        # Overlay on original image
        original_np = np.array(original_image.resize(img_size))
        overlay = (original_np * (1 - alpha) + colored_heatmap * alpha).astype(np.uint8)

        # Convert to base64 PNG
        overlay_img = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        logger.info(f"Grad-CAM generated for model '{model_name}', class index {class_idx}")

        return {
            "image": f"data:image/png;base64,{b64_string}",
            "model_used": model_name,
        }

    except Exception as e:
        logger.error(f"Grad-CAM generation failed for {model_name}: {e}")
        # Return empty result on failure — don't break the whole response
        return {
            "image": "",
            "model_used": model_name,
        }
