"""
Predictor service — async-safe inference with thread pool executor.

Runs PyTorch inference in a thread pool to avoid blocking FastAPI's event loop.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch

from app.config import settings, MODEL_CONFIG, DISEASE_DISPLAY_NAMES, NORMAL_CLASSES
from app.services.model_registry import registry
from app.utils.logger import logger

# Thread pool for blocking PyTorch inference
_executor = ThreadPoolExecutor(max_workers=2)


def _sync_inference(model_name: str, tensor: torch.Tensor) -> list[float]:
    """
    Run synchronous inference on a single model.

    Args:
        model_name: Name of the model in the registry
        tensor: Preprocessed image tensor (1, 3, 224, 224)

    Returns:
        List of softmax probabilities
    """
    model = registry.get_model(model_name)
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1).squeeze()
    return probabilities.tolist()


async def run_inference(model_name: str, tensor: torch.Tensor) -> list[float]:
    """Run inference in a thread pool (async-safe)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _sync_inference, model_name, tensor)


def get_risk_status(probability: float) -> str:
    """Determine risk level from probability."""
    if probability >= settings.HIGH_RISK_THRESHOLD:
        return "high_risk"
    elif probability >= settings.MEDIUM_RISK_THRESHOLD:
        return "medium_risk"
    else:
        return "low_risk"


async def predict_diseases(model_names: list[str], tensor: torch.Tensor) -> list[dict]:
    """
    Run inference on multiple disease models and aggregate results.

    Args:
        model_names: List of disease model names to run
        tensor: Preprocessed image tensor

    Returns:
        List of PredictionItem dicts: [{disease, probability, status}, ...]
    """
    all_predictions = []

    for model_name in model_names:
        if not registry.has_model(model_name):
            logger.warning(f"Model '{model_name}' not available, skipping")
            continue

        config = MODEL_CONFIG[model_name]
        class_names = config["class_names"]
        display_names = DISEASE_DISPLAY_NAMES.get(model_name, {})

        # Run inference
        probabilities = await run_inference(model_name, tensor)

        # Build prediction items for each class
        for class_name, prob in zip(class_names, probabilities):
            display_name = display_names.get(class_name, class_name.replace("_", " ").title())

            # For normal/healthy classes, invert risk logic:
            # High probability of "no disease" = LOW risk (good news)
            is_normal = class_name in NORMAL_CLASSES
            if is_normal:
                status = get_risk_status(1.0 - prob)
            else:
                status = get_risk_status(prob)

            all_predictions.append({
                "disease": display_name,
                "probability": round(prob, 4),
                "status": status,
                "_model": model_name,  # internal — used for Grad-CAM routing
                "_class_idx": class_names.index(class_name),
            })

    # Sort by probability descending
    all_predictions.sort(key=lambda x: x["probability"], reverse=True)

    logger.info(f"Predictions complete: {len(all_predictions)} results from {len(model_names)} models")
    return all_predictions
