"""Image type classification service — detects xray, mri, ct_scan, or skin.

Includes a confidence-threshold fallback: when the classifier is uncertain,
the top-2 modalities are both returned so the router can run models for both.
"""
import torch

from app.config import MODEL_CONFIG
from app.services.model_registry import registry
from app.utils.logger import logger

# Minimum confidence to trust the top-1 prediction outright.
# Below this, we flag the result as low-confidence and include the runner-up.
CONFIDENCE_THRESHOLD = 0.60

# If the gap between top-1 and top-2 is smaller than this, treat it as ambiguous.
AMBIGUITY_GAP = 0.15


def classify_image_type(tensor: torch.Tensor) -> dict:
    """
    Classify an image as xray, mri, ct_scan, or skin.

    Args:
        tensor: Preprocessed image tensor of shape (1, 3, 224, 224) on the correct device

    Returns:
        dict with keys:
            detected (str)          — primary modality
            confidence (float)      — top-1 softmax probability
            probabilities (dict)    — all class probabilities
            low_confidence (bool)   — True when confidence < threshold
            secondary_type (str|None) — runner-up modality if ambiguous
    """
    model = registry.get_model("image_classifier")
    class_names = MODEL_CONFIG["image_classifier"]["class_names"]

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1).squeeze()

    probs_dict = {name: round(prob.item(), 4) for name, prob in zip(class_names, probabilities)}

    # Sort by probability descending
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    top_type, top_conf = sorted_probs[0]
    runner_up_type, runner_up_conf = sorted_probs[1]

    # Determine if the classification is uncertain
    is_low_confidence = top_conf < CONFIDENCE_THRESHOLD
    is_ambiguous = (top_conf - runner_up_conf) < AMBIGUITY_GAP

    secondary_type = None
    if is_low_confidence or is_ambiguous:
        secondary_type = runner_up_type
        logger.warning(
            f"Low-confidence image classification: {top_type}={top_conf:.2%} vs "
            f"{runner_up_type}={runner_up_conf:.2%} — will run both routes"
        )
    else:
        logger.info(f"Image classified as: {top_type} ({top_conf:.2%})")

    return {
        "detected": top_type,
        "confidence": round(top_conf, 4),
        "probabilities": probs_dict,
        "low_confidence": is_low_confidence or is_ambiguous,
        "secondary_type": secondary_type,
    }
