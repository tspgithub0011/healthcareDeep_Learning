from pydantic import BaseModel, Field
from typing import Optional


class ImageTypeResult(BaseModel):
    """Result of the image type classification step."""
    detected: str = Field(..., description="Detected image type: xray, mri, ct_scan, or skin")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    probabilities: dict[str, float] = Field(
        ..., description="Probability for each image type"
    )
    low_confidence: bool = Field(
        default=False, description="True when classifier confidence is below threshold"
    )
    secondary_type: Optional[str] = Field(
        default=None, description="Runner-up modality when classification is ambiguous"
    )


class PredictionItem(BaseModel):
    """A single disease prediction result."""
    disease: str = Field(..., description="Human-readable disease name")
    probability: float = Field(..., ge=0, le=1, description="Prediction probability 0-1")
    status: str = Field(..., description="Risk level: high_risk, medium_risk, or low_risk")
    model: str = Field(default="", description="Source model name for unique identification")


class GradCAMResult(BaseModel):
    """Grad-CAM heatmap result."""
    model_config = {"protected_namespaces": ()}

    image: str = Field(..., description="Base64-encoded PNG heatmap overlay")
    model_used: str = Field(..., description="Name of the model used for Grad-CAM")


class PredictResponse(BaseModel):
    """Full prediction response from POST /api/predict."""
    image_type: ImageTypeResult
    predictions: list[PredictionItem]
    top_prediction: PredictionItem
    gradcam: GradCAMResult
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    disclaimer: str = Field(..., description="Medical disclaimer text")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_type": {
                        "detected": "xray",
                        "confidence": 0.92,
                        "probabilities": {"xray": 0.92, "mri": 0.05, "skin": 0.03},
                    },
                    "predictions": [
                        {"disease": "Pneumonia", "probability": 0.87, "status": "high_risk"},
                        {"disease": "COVID-19", "probability": 0.12, "status": "low_risk"},
                    ],
                    "top_prediction": {
                        "disease": "Pneumonia",
                        "probability": 0.87,
                        "status": "high_risk",
                    },
                    "gradcam": {
                        "image": "data:image/png;base64,iVBOR...",
                        "model_used": "pneumonia",
                    },
                    "processing_time_ms": 342,
                    "disclaimer": "This is an AI screening tool — not a diagnosis.",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response from GET /api/health."""
    status: str = Field(..., description="Server status: ready or loading")
    models_loaded: int = Field(..., description="Number of models loaded in memory")
    gpu_available: bool = Field(..., description="Whether GPU/CUDA is available")
    version: str = Field(..., description="API version string")


class ReportRequest(BaseModel):
    """Request body for POST /api/report."""
    prediction: dict = Field(..., description="Full prediction response JSON")
    image: Optional[str] = Field(None, description="Base64-encoded original image")
