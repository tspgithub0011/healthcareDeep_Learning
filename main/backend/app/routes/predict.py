"""POST /api/predict — full inference pipeline."""
import time

from fastapi import APIRouter, UploadFile, File

from app.config import settings
from app.models.schemas import PredictResponse, ImageTypeResult, PredictionItem, GradCAMResult
from app.utils.validators import validate_upload
from app.utils.image_processing import preprocess_image
from app.services.model_registry import registry
from app.services.image_classifier import classify_image_type
from app.services.model_router import get_models_for_image_type
from app.services.predictor import predict_diseases
from app.services.gradcam import generate_gradcam
from app.utils.logger import logger

router = APIRouter()


@router.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a medical image and get disease predictions.

    Pipeline:
    1. Validate uploaded file (type, size, corruption)
    2. Preprocess image for model input
    3. Classify image type (xray / mri / ct_scan / skin)
    4. Route to correct disease models (multi-route if uncertain)
    5. Run inference on all relevant models
    6. Generate Grad-CAM for the top prediction
    7. Return full response
    """
    start_time = time.time()

    # 1. Validate
    pil_image = await validate_upload(file)
    logger.info(f"Processing image: {file.filename}")

    # 2. Preprocess
    tensor = preprocess_image(pil_image, registry.device)

    # 3. Classify image type
    image_type_result = classify_image_type(tensor)

    # 4. Route to disease models
    #    If the classifier is uncertain, run BOTH the primary and secondary routes
    primary_models = get_models_for_image_type(image_type_result["detected"])

    secondary_models = []
    if image_type_result.get("secondary_type"):
        secondary_models = get_models_for_image_type(image_type_result["secondary_type"])
        logger.info(
            f"Multi-route fallback: primary={primary_models}, secondary={secondary_models}"
        )

    # Combine and deduplicate while preserving order
    all_disease_models = list(dict.fromkeys(primary_models + secondary_models))

    # 5. Run inference on all models
    raw_predictions = await predict_diseases(all_disease_models, tensor)

    # 6. Determine top prediction and generate Grad-CAM
    top_pred = raw_predictions[0] if raw_predictions else None

    gradcam_result = {"image": "", "model_used": ""}
    if top_pred:
        gradcam_result = generate_gradcam(
            model_name=top_pred["_model"],
            input_tensor=tensor,
            class_idx=top_pred["_class_idx"],
            original_image=pil_image,
        )

    # 7. Build response (strip internal fields)
    predictions = [
        PredictionItem(
            disease=p["disease"],
            probability=p["probability"],
            status=p["status"],
            model=p["_model"],
        )
        for p in raw_predictions
    ]

    top_prediction = PredictionItem(
        disease=top_pred["disease"],
        probability=top_pred["probability"],
        status=top_pred["status"],
        model=top_pred["_model"],
    ) if top_pred else PredictionItem(disease="Unknown", probability=0, status="low_risk")

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Build image type result, passing through low_confidence flag
    image_type_for_response = {
        "detected": image_type_result["detected"],
        "confidence": image_type_result["confidence"],
        "probabilities": image_type_result["probabilities"],
        "low_confidence": image_type_result.get("low_confidence", False),
        "secondary_type": image_type_result.get("secondary_type"),
    }

    response = PredictResponse(
        image_type=ImageTypeResult(**image_type_for_response),
        predictions=predictions,
        top_prediction=top_prediction,
        gradcam=GradCAMResult(**gradcam_result),
        processing_time_ms=elapsed_ms,
        disclaimer=settings.DISCLAIMER,
    )

    logger.info(f"Prediction complete in {elapsed_ms}ms — top: {top_prediction.disease} ({top_prediction.probability:.2%})")
    return response
