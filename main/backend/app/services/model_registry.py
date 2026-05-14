"""
Model Registry — loads all 7 models at startup, caches them in memory.

Supports both dummy (random weights) and trained (.pth) models.
If a .pth file exists in MODEL_DIR for a model, those weights are loaded.
Otherwise, random weights are used (dummy-first pipeline).
"""
import os
import torch
import torch.nn as nn

from app.config import settings, MODEL_CONFIG
from app.models.architectures import build_model
from app.utils.logger import logger


class ModelRegistry:
    """Centralized model loading, caching, and serving."""

    def __init__(self):
        self._models: dict[str, nn.Module] = {}
        self._device: torch.device = torch.device("cpu")
        self._loaded = False

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def num_loaded(self) -> int:
        return len(self._models)

    def load_all_models(self):
        """Load all 7 models into memory. Called once at startup."""
        # Select device
        if settings.USE_GPU and torch.cuda.is_available():
            self._device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using device: cuda ({gpu_name})")
        else:
            self._device = torch.device("cpu")
            logger.info("Using device: cpu")

        # Load each model
        for model_name, config in MODEL_CONFIG.items():
            try:
                model = build_model(config["architecture"], config["num_classes"])

                # Check for trained weights
                pth_path = os.path.join(settings.MODEL_DIR, f"{model_name}.pth")
                if os.path.exists(pth_path):
                    state_dict = torch.load(pth_path, map_location=self._device, weights_only=True)
                    model.load_state_dict(state_dict)
                    logger.info(f"  ✓ {model_name}: loaded trained weights from {pth_path}")
                else:
                    logger.info(f"  ✓ {model_name}: using random weights (dummy mode)")

                model.to(self._device)
                model.eval()
                self._models[model_name] = model

            except Exception as e:
                logger.error(f"  ✗ {model_name}: failed to load — {e}")

        # Warm-up inference (forces CUDA kernel compilation)
        self._warmup()

        self._loaded = True
        logger.info(f"Model registry ready: {self.num_loaded}/{len(MODEL_CONFIG)} models loaded")

    def _warmup(self):
        """Run one dummy inference per model to warm up CUDA kernels."""
        logger.info("Running warm-up inference...")
        dummy = torch.randn(1, 3, 224, 224).to(self._device)
        with torch.no_grad():
            for name, model in self._models.items():
                try:
                    model(dummy)
                except Exception as e:
                    logger.warning(f"  Warm-up failed for {name}: {e}")

    def get_model(self, model_name: str) -> nn.Module:
        """Get a loaded model by name."""
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found in registry. Loaded: {list(self._models.keys())}")
        return self._models[model_name]

    def has_model(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self._models


# Global singleton
registry = ModelRegistry()
