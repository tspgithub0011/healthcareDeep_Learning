"""
Model Registry — Lazy-loading strategy for memory-constrained deployments.

Instead of loading all 7 models at startup (~500+ MB RAM), this registry:
1. Loads ONLY the image_classifier at startup (always needed first).
2. Loads disease-specific models ON DEMAND when a request needs them.
3. Keeps at most MAX_CACHED models in memory using LRU eviction.

This keeps RAM usage under ~350 MB, fitting within Render's 512 MB free tier.
"""
import gc
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from app.config import settings, MODEL_CONFIG
from app.models.architectures import build_model
from app.utils.logger import logger

# Maximum number of models to keep in memory simultaneously.
# image_classifier (always loaded) + 1 disease model = 2 total.
MAX_CACHED = 3


class ModelRegistry:
    """Centralized model loading with LRU caching and lazy loading."""

    def __init__(self):
        self._models: OrderedDict[str, nn.Module] = OrderedDict()
        self._device: torch.device = torch.device("cpu")
        self._ready = False

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._ready

    @property
    def num_loaded(self) -> int:
        return len(self._models)

    # ── Startup: load only the image classifier ──

    def load_all_models(self):
        """
        Called once at startup. Only loads the image_classifier model
        to minimize memory usage. Other models are loaded on demand.
        """
        # Select device
        if settings.USE_GPU and torch.cuda.is_available():
            self._device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using device: cuda ({gpu_name})")
        else:
            self._device = torch.device("cpu")
            logger.info("Using device: cpu")

        # Load only the image classifier at startup (always needed first)
        self._load_single_model("image_classifier")

        self._ready = True
        logger.info(f"Model registry ready (lazy mode): image_classifier loaded, "
                     f"other models will load on demand")

    # ── Core: load a single model into memory ──

    def _load_single_model(self, model_name: str) -> nn.Module:
        """Load one model into the cache, evicting oldest if at capacity."""
        if model_name in self._models:
            # Move to end (most recently used)
            self._models.move_to_end(model_name)
            return self._models[model_name]

        config = MODEL_CONFIG.get(model_name)
        if config is None:
            raise KeyError(f"Unknown model: '{model_name}'. "
                           f"Available: {list(MODEL_CONFIG.keys())}")

        # Evict oldest model if at capacity (but never evict image_classifier)
        while len(self._models) >= MAX_CACHED:
            oldest_name, oldest_model = next(iter(self._models.items()))
            if oldest_name == "image_classifier" and len(self._models) > 1:
                # Skip image_classifier, evict the next one
                self._models.move_to_end("image_classifier", last=False)
                oldest_name, oldest_model = list(self._models.items())[1] if len(self._models) > 1 else (oldest_name, oldest_model)

            logger.info(f"  ♻ Evicting {oldest_name} from memory to make room")
            del self._models[oldest_name]
            del oldest_model
            gc.collect()  # Force garbage collection to reclaim memory immediately

        # Build and load
        try:
            model = build_model(config["architecture"], config["num_classes"])

            pth_path = os.path.join(settings.MODEL_DIR, f"{model_name}.pth")
            if os.path.exists(pth_path):
                state_dict = torch.load(pth_path, map_location=self._device, weights_only=True)
                model.load_state_dict(state_dict)
                del state_dict  # Free the duplicate copy immediately
                gc.collect()
                logger.info(f"  ✓ {model_name}: loaded trained weights from {pth_path}")
            else:
                logger.info(f"  ✓ {model_name}: using random weights (dummy mode)")

            model.to(self._device)
            model.eval()

            self._models[model_name] = model
            return model

        except Exception as e:
            logger.error(f"  ✗ {model_name}: failed to load — {e}")
            raise

    # ── Public API ──

    def get_model(self, model_name: str) -> nn.Module:
        """
        Get a model by name. Loads it on demand if not already in memory.
        Uses LRU eviction to stay within memory limits.
        """
        if model_name in self._models:
            self._models.move_to_end(model_name)
            return self._models[model_name]

        # Lazy load
        logger.info(f"  ⏳ Lazy-loading model: {model_name}")
        return self._load_single_model(model_name)

    def has_model(self, model_name: str) -> bool:
        """Check if a model config exists (not necessarily loaded in memory)."""
        return model_name in MODEL_CONFIG


# Global singleton
registry = ModelRegistry()
