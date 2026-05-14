"""Model router — routes an image type to the correct disease models."""
from app.config import IMAGE_TYPE_ROUTES, MODEL_CONFIG, DISEASE_DISPLAY_NAMES
from app.utils.logger import logger


def get_models_for_image_type(image_type: str) -> list[str]:
    """
    Get the list of disease model names to run for a given image type.

    Args:
        image_type: 'xray', 'mri', or 'skin'

    Returns:
        List of model name strings, e.g. ['pneumonia', 'covid', 'lung_cancer', 'cardiomegaly']
    """
    if image_type not in IMAGE_TYPE_ROUTES:
        logger.warning(f"Unknown image type '{image_type}', defaulting to all disease models")
        # Fallback: run all disease models (excluding i mage_classifier)
        return [name for name in MODEL_CONFIG if name != "image_classifier"]

    models = IMAGE_TYPE_ROUTES[image_type]
    logger.info(f"Routing '{image_type}' → models: {models}")
    return models


def get_class_display_names(model_name: str) -> dict[str, str]:
    """
    Get human-readable display names for a model's output classes.

    Args:
        model_name: e.g. 'pneumonia'

    Returns:
        dict mapping class_name → display_name, e.g. {'pneumonia': 'Pneumonia'}
    """
    return DISEASE_DISPLAY_NAMES.get(model_name, {})
