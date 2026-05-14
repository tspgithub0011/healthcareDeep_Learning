"""GET /api/health — liveness check + model readiness."""
import torch
from fastapi import APIRouter

from app.config import settings
from app.models.schemas import HealthResponse
from app.services.model_registry import registry

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Return server status, loaded model count, and GPU availability."""
    return HealthResponse(
        status="ready" if registry.is_loaded else "loading",
        models_loaded=registry.num_loaded,
        gpu_available=torch.cuda.is_available(),
        version=settings.VERSION,
    )
