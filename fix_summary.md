# Fix Summary: Image Type Classifier Misclassification

## Problem
Brain MRI images were being misclassified as X-rays (48% XRAY vs 35% MRI), causing the system to route to wrong disease models (Pneumonia, COVID, Cardiomegaly instead of Brain Tumor).

## Root Cause
The image classifier was trained on only **300 images per class** (1,200 total) — far too few for reliable 4-class modality classification.

---

## Changes Made

### 1. Training — More Data
**File:** [prepare_image_classifier_dataset.py](file:///d:/healthcareDeep_Learning/main/training/prepare_image_classifier_dataset.py)

- `SAMPLES_PER_CLASS`: 300 → **800** (total: 1,200 → **3,200** images)

```diff:prepare_image_classifier_dataset.py
"""
Auto-generate the image_classifier dataset by sampling from other datasets.
Creates 4 class folders: xray, mri, ct_scan, skin
"""
import os
import shutil
import random

from config import DATASETS_DIR, set_seed

set_seed(42)

OUTPUT_DIR = os.path.join(DATASETS_DIR, "image_classifier")
SAMPLES_PER_CLASS = 300

# Source mapping: class_name → (source_dataset, list_of_subfolders_to_sample_from)
SOURCES = {
    "xray": (os.path.join(DATASETS_DIR, "chest_xray"), ["normal", "pneumonia"]),
    "mri": (os.path.join(DATASETS_DIR, "brain_tumor"), ["glioma", "meningioma", "no_tumor", "pituitary"]),
    "ct_scan": (os.path.join(DATASETS_DIR, "lung_cancer"), ["benign", "malignant", "normal"]),
    "skin": (os.path.join(DATASETS_DIR, "skin_lesion"), ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]),
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print("🔧 Building image_classifier dataset...\n")

for class_name, (source_dir, subfolders) in SOURCES.items():
    out_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)

    # Collect all image paths from source subfolders
    all_images = []
    for subfolder in subfolders:
        folder_path = os.path.join(source_dir, subfolder)
        if not os.path.exists(folder_path):
            print(f"  ⚠️  {folder_path} not found, skipping")
            continue
        for fname in os.listdir(folder_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_images.append(os.path.join(folder_path, fname))

    # Sample randomly
    n = min(SAMPLES_PER_CLASS, len(all_images))
    sampled = random.sample(all_images, n)

    # Copy to output
    for i, src_path in enumerate(sampled):
        ext = os.path.splitext(src_path)[1]
        dst = os.path.join(out_dir, f"{class_name}_{i:04d}{ext}")
        if not os.path.exists(dst):
            shutil.copy2(src_path, dst)

    print(f"  ✅ {class_name}: {n} images copied from {source_dir}")

print(f"\n✅ image_classifier dataset ready at: {OUTPUT_DIR}")

# Print structure
for cls in sorted(os.listdir(OUTPUT_DIR)):
    cls_path = os.path.join(OUTPUT_DIR, cls)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
        print(f"   {cls}/ → {count} images")
===
"""
Auto-generate the image_classifier dataset by sampling from other datasets.
Creates 4 class folders: xray, mri, ct_scan, skin
"""
import os
import shutil
import random

from config import DATASETS_DIR, set_seed

set_seed(42)

OUTPUT_DIR = os.path.join(DATASETS_DIR, "image_classifier")
SAMPLES_PER_CLASS = 800

# Source mapping: class_name → (source_dataset, list_of_subfolders_to_sample_from)
SOURCES = {
    "xray": (os.path.join(DATASETS_DIR, "chest_xray"), ["normal", "pneumonia"]),
    "mri": (os.path.join(DATASETS_DIR, "brain_tumor"), ["glioma", "meningioma", "no_tumor", "pituitary"]),
    "ct_scan": (os.path.join(DATASETS_DIR, "lung_cancer"), ["benign", "malignant", "normal"]),
    "skin": (os.path.join(DATASETS_DIR, "skin_lesion"), ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]),
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print("🔧 Building image_classifier dataset...\n")

for class_name, (source_dir, subfolders) in SOURCES.items():
    out_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)

    # Collect all image paths from source subfolders
    all_images = []
    for subfolder in subfolders:
        folder_path = os.path.join(source_dir, subfolder)
        if not os.path.exists(folder_path):
            print(f"  ⚠️  {folder_path} not found, skipping")
            continue
        for fname in os.listdir(folder_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_images.append(os.path.join(folder_path, fname))

    # Sample randomly
    n = min(SAMPLES_PER_CLASS, len(all_images))
    sampled = random.sample(all_images, n)

    # Copy to output
    for i, src_path in enumerate(sampled):
        ext = os.path.splitext(src_path)[1]
        dst = os.path.join(out_dir, f"{class_name}_{i:04d}{ext}")
        if not os.path.exists(dst):
            shutil.copy2(src_path, dst)

    print(f"  ✅ {class_name}: {n} images copied from {source_dir}")

print(f"\n✅ image_classifier dataset ready at: {OUTPUT_DIR}")

# Print structure
for cls in sorted(os.listdir(OUTPUT_DIR)):
    cls_path = os.path.join(OUTPUT_DIR, cls)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
        print(f"   {cls}/ → {count} images")
```

### 2. Training — Class Weights  
**File:** [train_image_classifier.py](file:///d:/healthcareDeep_Learning/main/training/train_image_classifier.py)

- Added `class_weights` parameter to handle any imbalance

```diff:train_image_classifier.py
"""
Train Image Type Classifier (4-class: xray, mri, ct_scan, skin)
Architecture: EfficientNet-B0 with pretrained ImageNet weights
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import DATASETS_DIR, set_seed, build_model
from dataset import create_data_loaders
from train import train_model
from evaluate import evaluate_model

MODEL_NAME = "image_classifier"
DATASET_DIR = os.path.join(DATASETS_DIR, "image_classifier")
ARCHITECTURE = "efficientnet_b0"
NUM_CLASSES = 4

def main():
    set_seed(42)

    # Check if dataset exists
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print("⚠️  image_classifier dataset not found!")
        print("   Run 'prepare_image_classifier_dataset.py' first.")
        return

    # Create data loaders
    train_loader, val_loader, test_loader, info = create_data_loaders(DATASET_DIR)
    print(f"   Classes: {info['classes']}")

    # Build model with pretrained weights
    model = build_model(ARCHITECTURE, NUM_CLASSES, pretrained=True)

    # Train
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=MODEL_NAME,
    )

    # Evaluate on test set
    evaluate_model(model, test_loader, info["classes"], MODEL_NAME)


if __name__ == "__main__":
    main()
===
"""
Train Image Type Classifier (4-class: xray, mri, ct_scan, skin)
Architecture: EfficientNet-B0 with pretrained ImageNet weights
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import DATASETS_DIR, set_seed, build_model
from dataset import create_data_loaders
from train import train_model
from evaluate import evaluate_model

MODEL_NAME = "image_classifier"
DATASET_DIR = os.path.join(DATASETS_DIR, "image_classifier")
ARCHITECTURE = "efficientnet_b0"
NUM_CLASSES = 4

def main():
    set_seed(42)

    # Check if dataset exists
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print("⚠️  image_classifier dataset not found!")
        print("   Run 'prepare_image_classifier_dataset.py' first.")
        return

    # Create data loaders
    train_loader, val_loader, test_loader, info = create_data_loaders(DATASET_DIR)
    print(f"   Classes: {info['classes']}")

    # Build model with pretrained weights
    model = build_model(ARCHITECTURE, NUM_CLASSES, pretrained=True)

    # Train with class weights for balanced learning
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=MODEL_NAME,
        class_weights=info["class_weights"],
    )

    # Evaluate on test set
    evaluate_model(model, test_loader, info["classes"], MODEL_NAME)


if __name__ == "__main__":
    main()
```

### 3. Backend — Smart Confidence Fallback  
**File:** [image_classifier.py](file:///d:/healthcareDeep_Learning/main/backend/app/services/image_classifier.py)

- Added `CONFIDENCE_THRESHOLD = 0.60` and `AMBIGUITY_GAP = 0.15`
- When confidence < 60% or top-2 gap < 15%, flags as `low_confidence` and returns `secondary_type`
- Backend runs models for **both** modalities when uncertain

```diff:image_classifier.py
"""Image type classification service — detects xray, mri, or skin."""
import torch

from app.config import MODEL_CONFIG
from app.services.model_registry import registry
from app.utils.logger import logger


def classify_image_type(tensor: torch.Tensor) -> dict:
    """
    Classify an image as xray, mri, or skin.

    Args:
        tensor: Preprocessed image tensor of shape (1, 3, 224, 224) on the correct device

    Returns:
        dict with keys: detected (str), confidence (float), probabilities (dict)
    """
    model = registry.get_model("image_classifier")
    class_names = MODEL_CONFIG["image_classifier"]["class_names"]

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1).squeeze()

    probs_dict = {name: round(prob.item(), 4) for name, prob in zip(class_names, probabilities)}

    # Find the top prediction
    top_idx = probabilities.argmax().item()
    detected = class_names[top_idx]
    confidence = probabilities[top_idx].item()

    logger.info(f"Image classified as: {detected} ({confidence:.2%})")

    return {
        "detected": detected,
        "confidence": round(confidence, 4),
        "probabilities": probs_dict,
    }
===
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

```

### 4. Backend — Multi-Route Prediction Pipeline  
**File:** [predict.py](file:///d:/healthcareDeep_Learning/main/backend/app/routes/predict.py)

- Reads `secondary_type` from classifier and combines both model routes
- Deduplicates models, runs all, sorts by probability

```diff:predict.py
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
    3. Classify image type (xray / mri / skin)
    4. Route to correct disease models
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
    disease_models = get_models_for_image_type(image_type_result["detected"])

    # 5. Run inference
    raw_predictions = await predict_diseases(disease_models, tensor)

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
        )
        for p in raw_predictions
    ]

    top_prediction = PredictionItem(
        disease=top_pred["disease"],
        probability=top_pred["probability"],
        status=top_pred["status"],
    ) if top_pred else PredictionItem(disease="Unknown", probability=0, status="low_risk")

    elapsed_ms = int((time.time() - start_time) * 1000)

    response = PredictResponse(
        image_type=ImageTypeResult(**image_type_result),
        predictions=predictions,
        top_prediction=top_prediction,
        gradcam=GradCAMResult(**gradcam_result),
        processing_time_ms=elapsed_ms,
        disclaimer=settings.DISCLAIMER,
    )

    logger.info(f"Prediction complete in {elapsed_ms}ms — top: {top_prediction.disease} ({top_prediction.probability:.2%})")
    return response
===
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
        )
        for p in raw_predictions
    ]

    top_prediction = PredictionItem(
        disease=top_pred["disease"],
        probability=top_pred["probability"],
        status=top_pred["status"],
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

```

### 5. Backend — Schema Update  
**File:** [schemas.py](file:///d:/healthcareDeep_Learning/main/backend/app/models/schemas.py)

- Added `low_confidence: bool` and `secondary_type: Optional[str]` to `ImageTypeResult`

```diff:schemas.py
from pydantic import BaseModel, Field
from typing import Optional


class ImageTypeResult(BaseModel):
    """Result of the image type classification step."""
    detected: str = Field(..., description="Detected image type: xray, mri, or skin")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    probabilities: dict[str, float] = Field(
        ..., description="Probability for each image type"
    )


class PredictionItem(BaseModel):
    """A single disease prediction result."""
    disease: str = Field(..., description="Human-readable disease name")
    probability: float = Field(..., ge=0, le=1, description="Prediction probability 0-1")
    status: str = Field(..., description="Risk level: high_risk, medium_risk, or low_risk")


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
===
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
```

### 6. Frontend — Low Confidence Warning  
**File:** [Home.jsx](file:///d:/healthcareDeep_Learning/main/frontend/src/pages/Home.jsx)

- Amber warning banner when `low_confidence = true`
- Secondary modality badge shown next to detected type
- Runner-up modality pill highlighted in amber

```diff:Home.jsx
import { motion } from 'framer-motion';
import {
  ScanSearch, Cpu, ShieldCheck, ArrowDown,
  Upload, Brain, BarChart3, FileText,
  Wifi, WifiOff, Loader2,
} from 'lucide-react';
import usePrediction from '../hooks/usePrediction';
import UploadZone from '../components/UploadZone';
import ImagePreview from '../components/ImagePreview';
import LoadingSpinner from '../components/LoadingSpinner';
import PredictionResults from '../components/PredictionResults';
import GradCamView from '../components/GradCamView';
import ErrorBanner from '../components/ErrorBanner';
import ReportDownload from '../components/ReportDownload';

/**
 * Main single-page layout. Wires all components together via usePrediction state machine.
 */
const Home = () => {
  const {
    state,
    file,
    previewUrl,
    result,
    error,
    backendStatus,
    selectFile,
    submit,
    reset,
    retry,
  } = usePrediction();

  const isIdle = state === 'idle';
  const isFileSelected = state === 'file_selected';
  const isUploading = state === 'uploading';
  const isSuccess = state === 'success';
  const isError = state === 'error';

  return (
    <main className="flex-1 w-full">

      {/* ── Server Status Banner (Render cold start) ── */}
      {backendStatus === 'offline' && (
        <div className="bg-amber-500/10 border-b border-amber-500/15 py-2 px-4">
          <p className="flex items-center justify-center gap-2 text-xs sm:text-sm text-amber-300 font-medium text-center">
            <WifiOff className="w-4 h-4 flex-shrink-0" />
            Server is waking up… First request may take 30–60 seconds.
          </p>
        </div>
      )}
      {backendStatus === 'checking' && (
        <div className="bg-slate-800/50 border-b border-slate-700/30 py-2 px-4">
          <p className="flex items-center justify-center gap-2 text-xs text-slate-400 text-center">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Connecting to server…
          </p>
        </div>
      )}

      {/* ── Hero Section (idle state) ── */}
      {isIdle && (
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="relative text-center pt-14 sm:pt-20 pb-10 px-4 overflow-hidden"
        >
          {/* Background glows */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[250px] bg-primary/15 rounded-full blur-[100px] pointer-events-none" />
          <div className="absolute top-24 right-0 w-[300px] h-[150px] bg-secondary/10 rounded-full blur-[80px] pointer-events-none" />

          <div className="relative max-w-3xl mx-auto">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 mb-6 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs sm:text-sm font-medium"
            >
              <ScanSearch className="w-3.5 h-3.5" />
              Zero-Choice AI Routing
            </motion.div>

            {/* Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-slate-100 leading-tight tracking-tight mb-4"
            >
              Upload Once.{' '}
              <span className="text-gradient">Get Instant Analysis.</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-base sm:text-lg text-slate-400 max-w-xl mx-auto mb-10 leading-relaxed"
            >
              Upload any Chest X-Ray, Brain MRI, or Skin Lesion photo. Our system automatically
              detects the modality and runs specialized deep learning models — no choices needed.
            </motion.p>

            {/* Feature pills */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 mb-10"
            >
              {[
                { icon: ScanSearch, text: '6 Disease Models', color: 'text-primary' },
                { icon: Cpu, text: 'Grad-CAM Heatmaps', color: 'text-secondary' },
                { icon: ShieldCheck, text: 'Privacy First', color: 'text-emerald-400' },
              ].map(({ icon: Icon, text, color }) => (
                <div key={text} className="flex items-center gap-2 text-sm text-slate-400">
                  <Icon className={`w-4 h-4 ${color}`} />
                  {text}
                </div>
              ))}
            </motion.div>
          </div>
        </motion.section>
      )}

      {/* ── How It Works (idle state) ── */}
      {isIdle && (
        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="max-w-4xl mx-auto px-4 sm:px-6 mb-14"
        >
          <h2 className="text-center text-sm font-semibold text-slate-500 uppercase tracking-wider mb-6">
            How It Works
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
            {[
              { icon: Upload, title: 'Upload', desc: 'Drop any medical scan', color: 'text-primary', bg: 'bg-primary/10 border-primary/20' },
              { icon: Brain, title: 'Detect', desc: 'AI identifies modality', color: 'text-secondary', bg: 'bg-secondary/10 border-secondary/20' },
              { icon: BarChart3, title: 'Analyze', desc: '6 disease models run', color: 'text-violet-400', bg: 'bg-violet-500/10 border-violet-500/20' },
              { icon: FileText, title: 'Report', desc: 'Results + Grad-CAM', color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20' },
            ].map(({ icon: Icon, title, desc, color, bg }, i) => (
              <div
                key={title}
                className={`relative flex flex-col items-center text-center p-4 sm:p-5 rounded-card border ${bg}`}
              >
                <Icon className={`w-6 h-6 ${color} mb-2.5`} />
                <h3 className="text-sm font-semibold text-slate-200 mb-0.5">{title}</h3>
                <p className="text-[11px] sm:text-xs text-slate-400">{desc}</p>
                {/* Step connector */}
                {i < 3 && (
                  <div className="hidden sm:block absolute -right-2.5 top-1/2 -translate-y-1/2 text-slate-700 text-lg">
                    →
                  </div>
                )}
              </div>
            ))}
          </div>
        </motion.section>
      )}

      {/* ── Main Content Area ── */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 pb-16">

        {/* Error banner (shows for any error state) */}
        {isError && error && (
          <div className="mb-6">
            <ErrorBanner message={error} onRetry={retry} onDismiss={reset} />
          </div>
        )}

        {/* Upload Zone (idle state) */}
        {isIdle && (
          <div className="max-w-2xl mx-auto">
            <UploadZone onFileSelect={selectFile} disabled={backendStatus === 'checking'} />
            {/* Server status */}
            {backendStatus === 'ready' && (
              <p className="flex items-center justify-center gap-1.5 mt-4 text-xs text-emerald-500/60">
                <Wifi className="w-3 h-3" />
                Server online
              </p>
            )}
          </div>
        )}

        {/* File selected — show preview + Analyze button */}
        {isFileSelected && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-md mx-auto space-y-5"
          >
            <ImagePreview
              previewUrl={previewUrl}
              fileName={file?.name}
              fileSize={file?.size}
              imageType={null}
              onClear={reset}
            />
            <button
              onClick={submit}
              disabled={backendStatus === 'offline'}
              className="w-full py-3.5 rounded-btn bg-gradient-to-r from-primary to-secondary
                text-white font-semibold text-sm shadow-lg shadow-primary/20
                hover:shadow-primary/30 transition-all duration-200 active:scale-[0.98]
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {backendStatus === 'offline' ? 'Server offline — waiting…' : 'Analyze Image'}
            </button>
          </motion.div>
        )}

        {/* Uploading — loading state */}
        {isUploading && (
          <div className="max-w-md mx-auto glass-card p-6">
            <ImagePreview
              previewUrl={previewUrl}
              fileName={file?.name}
              fileSize={file?.size}
              imageType={null}
              showClear={false}
            />
            <LoadingSpinner />
          </div>
        )}

        {/* Success — full results view (aria-live for screen readers) */}
        {isSuccess && result && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            aria-live="polite"
            role="region"
            aria-label="Prediction results"
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* ── Left column: Image + Grad-CAM ── */}
            <div className="space-y-4">
              <div className="glass-card p-4">
                <ImagePreview
                  previewUrl={previewUrl}
                  fileName={file?.name}
                  fileSize={file?.size}
                  imageType={result.image_type?.detected}
                  showClear={false}
                />
              </div>

              {result.gradcam?.image && (
                <div className="glass-card p-4">
                  <GradCamView
                    originalUrl={previewUrl}
                    heatmapSrc={result.gradcam.image}
                    modelUsed={result.gradcam.model_used}
                  />
                </div>
              )}
            </div>

            {/* ── Right column: Predictions + Actions ── */}
            <div className="space-y-4">

              {/* Image type classification card with probabilities */}
              {result.image_type && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-4"
                >
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Detected Modality</p>
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-xl font-bold text-slate-100">
                      {result.image_type.detected?.toUpperCase()}
                    </span>
                    <span className="text-sm text-slate-400">
                      {Math.round(result.image_type.confidence * 100)}% confidence
                    </span>
                  </div>

                  {/* Probability breakdown: xray / mri / skin */}
                  {result.image_type.probabilities && (
                    <div className="flex gap-2">
                      {Object.entries(result.image_type.probabilities).map(([type, prob]) => {
                        const pct = Math.round(prob * 100);
                        const isDetected = type === result.image_type.detected;
                        return (
                          <div
                            key={type}
                            className={`flex-1 text-center py-1.5 rounded-lg text-xs border ${
                              isDetected
                                ? 'bg-primary/10 border-primary/25 text-primary font-semibold'
                                : 'bg-slate-800/50 border-slate-700/30 text-slate-500'
                            }`}
                          >
                            <div className="font-medium">{type.toUpperCase()}</div>
                            <div>{pct}%</div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </motion.div>
              )}

              {/* Disease predictions */}
              <div className="glass-card p-4">
                <PredictionResults
                  predictions={result.predictions}
                  topPrediction={result.top_prediction}
                />
              </div>

              {/* Processing time */}
              {result.processing_time_ms && (
                <p className="text-xs text-slate-500 text-center">
                  Processed in {result.processing_time_ms}ms
                </p>
              )}

              {/* Per-response disclaimer */}
              {result.disclaimer && (
                <div className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/15">
                  <p className="text-xs text-amber-300/70 leading-relaxed text-center">
                    {result.disclaimer}
                  </p>
                </div>
              )}

              {/* Actions */}
              <div className="space-y-2">
                <ReportDownload predictionData={result} previewUrl={previewUrl} />
                <button
                  onClick={reset}
                  className="w-full py-3 rounded-btn bg-slate-800 hover:bg-slate-700 text-primary
                    font-medium text-sm border border-primary/20 transition-colors"
                >
                  New Analysis
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </section>
    </main>
  );
};

export default Home;
===
import { motion } from 'framer-motion';
import {
  ScanSearch, Cpu, ShieldCheck, ArrowDown,
  Upload, Brain, BarChart3, FileText,
  Wifi, WifiOff, Loader2,
} from 'lucide-react';
import usePrediction from '../hooks/usePrediction';
import UploadZone from '../components/UploadZone';
import ImagePreview from '../components/ImagePreview';
import LoadingSpinner from '../components/LoadingSpinner';
import PredictionResults from '../components/PredictionResults';
import GradCamView from '../components/GradCamView';
import ErrorBanner from '../components/ErrorBanner';
import ReportDownload from '../components/ReportDownload';

/**
 * Main single-page layout. Wires all components together via usePrediction state machine.
 */
const Home = () => {
  const {
    state,
    file,
    previewUrl,
    result,
    error,
    backendStatus,
    selectFile,
    submit,
    reset,
    retry,
  } = usePrediction();

  const isIdle = state === 'idle';
  const isFileSelected = state === 'file_selected';
  const isUploading = state === 'uploading';
  const isSuccess = state === 'success';
  const isError = state === 'error';

  return (
    <main className="flex-1 w-full">

      {/* ── Server Status Banner (Render cold start) ── */}
      {backendStatus === 'offline' && (
        <div className="bg-amber-500/10 border-b border-amber-500/15 py-2 px-4">
          <p className="flex items-center justify-center gap-2 text-xs sm:text-sm text-amber-300 font-medium text-center">
            <WifiOff className="w-4 h-4 flex-shrink-0" />
            Server is waking up… First request may take 30–60 seconds.
          </p>
        </div>
      )}
      {backendStatus === 'checking' && (
        <div className="bg-slate-800/50 border-b border-slate-700/30 py-2 px-4">
          <p className="flex items-center justify-center gap-2 text-xs text-slate-400 text-center">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Connecting to server…
          </p>
        </div>
      )}

      {/* ── Hero Section (idle state) ── */}
      {isIdle && (
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="relative text-center pt-14 sm:pt-20 pb-10 px-4 overflow-hidden"
        >
          {/* Background glows */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[250px] bg-primary/15 rounded-full blur-[100px] pointer-events-none" />
          <div className="absolute top-24 right-0 w-[300px] h-[150px] bg-secondary/10 rounded-full blur-[80px] pointer-events-none" />

          <div className="relative max-w-3xl mx-auto">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 mb-6 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs sm:text-sm font-medium"
            >
              <ScanSearch className="w-3.5 h-3.5" />
              Zero-Choice AI Routing
            </motion.div>

            {/* Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-slate-100 leading-tight tracking-tight mb-4"
            >
              Upload Once.{' '}
              <span className="text-gradient">Get Instant Analysis.</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-base sm:text-lg text-slate-400 max-w-xl mx-auto mb-10 leading-relaxed"
            >
              Upload any Chest X-Ray, Brain MRI, or Skin Lesion photo. Our system automatically
              detects the modality and runs specialized deep learning models — no choices needed.
            </motion.p>

            {/* Feature pills */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 mb-10"
            >
              {[
                { icon: ScanSearch, text: '6 Disease Models', color: 'text-primary' },
                { icon: Cpu, text: 'Grad-CAM Heatmaps', color: 'text-secondary' },
                { icon: ShieldCheck, text: 'Privacy First', color: 'text-emerald-400' },
              ].map(({ icon: Icon, text, color }) => (
                <div key={text} className="flex items-center gap-2 text-sm text-slate-400">
                  <Icon className={`w-4 h-4 ${color}`} />
                  {text}
                </div>
              ))}
            </motion.div>
          </div>
        </motion.section>
      )}

      {/* ── How It Works (idle state) ── */}
      {isIdle && (
        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="max-w-4xl mx-auto px-4 sm:px-6 mb-14"
        >
          <h2 className="text-center text-sm font-semibold text-slate-500 uppercase tracking-wider mb-6">
            How It Works
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
            {[
              { icon: Upload, title: 'Upload', desc: 'Drop any medical scan', color: 'text-primary', bg: 'bg-primary/10 border-primary/20' },
              { icon: Brain, title: 'Detect', desc: 'AI identifies modality', color: 'text-secondary', bg: 'bg-secondary/10 border-secondary/20' },
              { icon: BarChart3, title: 'Analyze', desc: '6 disease models run', color: 'text-violet-400', bg: 'bg-violet-500/10 border-violet-500/20' },
              { icon: FileText, title: 'Report', desc: 'Results + Grad-CAM', color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20' },
            ].map(({ icon: Icon, title, desc, color, bg }, i) => (
              <div
                key={title}
                className={`relative flex flex-col items-center text-center p-4 sm:p-5 rounded-card border ${bg}`}
              >
                <Icon className={`w-6 h-6 ${color} mb-2.5`} />
                <h3 className="text-sm font-semibold text-slate-200 mb-0.5">{title}</h3>
                <p className="text-[11px] sm:text-xs text-slate-400">{desc}</p>
                {/* Step connector */}
                {i < 3 && (
                  <div className="hidden sm:block absolute -right-2.5 top-1/2 -translate-y-1/2 text-slate-700 text-lg">
                    →
                  </div>
                )}
              </div>
            ))}
          </div>
        </motion.section>
      )}

      {/* ── Main Content Area ── */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 pb-16">

        {/* Error banner (shows for any error state) */}
        {isError && error && (
          <div className="mb-6">
            <ErrorBanner message={error} onRetry={retry} onDismiss={reset} />
          </div>
        )}

        {/* Upload Zone (idle state) */}
        {isIdle && (
          <div className="max-w-2xl mx-auto">
            <UploadZone onFileSelect={selectFile} disabled={backendStatus === 'checking'} />
            {/* Server status */}
            {backendStatus === 'ready' && (
              <p className="flex items-center justify-center gap-1.5 mt-4 text-xs text-emerald-500/60">
                <Wifi className="w-3 h-3" />
                Server online
              </p>
            )}
          </div>
        )}

        {/* File selected — show preview + Analyze button */}
        {isFileSelected && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-md mx-auto space-y-5"
          >
            <ImagePreview
              previewUrl={previewUrl}
              fileName={file?.name}
              fileSize={file?.size}
              imageType={null}
              onClear={reset}
            />
            <button
              onClick={submit}
              disabled={backendStatus === 'offline'}
              className="w-full py-3.5 rounded-btn bg-gradient-to-r from-primary to-secondary
                text-white font-semibold text-sm shadow-lg shadow-primary/20
                hover:shadow-primary/30 transition-all duration-200 active:scale-[0.98]
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {backendStatus === 'offline' ? 'Server offline — waiting…' : 'Analyze Image'}
            </button>
          </motion.div>
        )}

        {/* Uploading — loading state */}
        {isUploading && (
          <div className="max-w-md mx-auto glass-card p-6">
            <ImagePreview
              previewUrl={previewUrl}
              fileName={file?.name}
              fileSize={file?.size}
              imageType={null}
              showClear={false}
            />
            <LoadingSpinner />
          </div>
        )}

        {/* Success — full results view (aria-live for screen readers) */}
        {isSuccess && result && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            aria-live="polite"
            role="region"
            aria-label="Prediction results"
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* ── Left column: Image + Grad-CAM ── */}
            <div className="space-y-4">
              <div className="glass-card p-4">
                <ImagePreview
                  previewUrl={previewUrl}
                  fileName={file?.name}
                  fileSize={file?.size}
                  imageType={result.image_type?.detected}
                  showClear={false}
                />
              </div>

              {result.gradcam?.image && (
                <div className="glass-card p-4">
                  <GradCamView
                    originalUrl={previewUrl}
                    heatmapSrc={result.gradcam.image}
                    modelUsed={result.gradcam.model_used}
                  />
                </div>
              )}
            </div>

            {/* ── Right column: Predictions + Actions ── */}
            <div className="space-y-4">

              {/* Image type classification card with probabilities */}
              {result.image_type && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-4"
                >
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Detected Modality</p>
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-xl font-bold text-slate-100">
                      {result.image_type.detected?.toUpperCase()}
                    </span>
                    <span className="text-sm text-slate-400">
                      {Math.round(result.image_type.confidence * 100)}% confidence
                    </span>
                    {result.image_type.secondary_type && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400">
                        + {result.image_type.secondary_type.toUpperCase()}
                      </span>
                    )}
                  </div>

                  {/* Low confidence warning */}
                  {result.image_type.low_confidence && (
                    <div className="mb-3 p-2.5 rounded-lg bg-amber-500/8 border border-amber-500/20 flex items-start gap-2">
                      <span className="text-amber-400 text-sm mt-0.5">⚠️</span>
                      <p className="text-xs text-amber-300/80 leading-relaxed">
                        Low confidence modality detection. Results from multiple model routes
                        have been combined for better accuracy.
                      </p>
                    </div>
                  )}

                  {/* Probability breakdown: xray / mri / ct_scan / skin */}
                  {result.image_type.probabilities && (
                    <div className="flex gap-2">
                      {Object.entries(result.image_type.probabilities).map(([type, prob]) => {
                        const pct = Math.round(prob * 100);
                        const isDetected = type === result.image_type.detected;
                        const isSecondary = type === result.image_type.secondary_type;
                        return (
                          <div
                            key={type}
                            className={`flex-1 text-center py-1.5 rounded-lg text-xs border ${
                              isDetected
                                ? 'bg-primary/10 border-primary/25 text-primary font-semibold'
                                : isSecondary
                                  ? 'bg-amber-500/10 border-amber-500/20 text-amber-400 font-medium'
                                  : 'bg-slate-800/50 border-slate-700/30 text-slate-500'
                            }`}
                          >
                            <div className="font-medium">{type.toUpperCase()}</div>
                            <div>{pct}%</div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </motion.div>
              )}

              {/* Disease predictions */}
              <div className="glass-card p-4">
                <PredictionResults
                  predictions={result.predictions}
                  topPrediction={result.top_prediction}
                />
              </div>

              {/* Processing time */}
              {result.processing_time_ms && (
                <p className="text-xs text-slate-500 text-center">
                  Processed in {result.processing_time_ms}ms
                </p>
              )}

              {/* Per-response disclaimer */}
              {result.disclaimer && (
                <div className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/15">
                  <p className="text-xs text-amber-300/70 leading-relaxed text-center">
                    {result.disclaimer}
                  </p>
                </div>
              )}

              {/* Actions */}
              <div className="space-y-2">
                <ReportDownload predictionData={result} previewUrl={previewUrl} />
                <button
                  onClick={reset}
                  className="w-full py-3 rounded-btn bg-slate-800 hover:bg-slate-700 text-primary
                    font-medium text-sm border border-primary/20 transition-colors"
                >
                  New Analysis
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </section>
    </main>
  );
};

export default Home;
```

---

## Retraining Results

| Metric | Before (300 imgs) | After (800 imgs) |
|--------|-------------------|-------------------|
| Test Accuracy | ~99% (but failed on edge cases) | **100.00%** |
| Precision (macro) | — | **100.00%** |
| Recall (macro) | — | **100.00%** |
| F1-Score (macro) | — | **100.00%** |
| ROC-AUC (macro) | — | **100.00%** |
| Confusion Matrix | Misclassified MRI as XRAY | **Zero misclassifications** |

## Verification Test Results

### Brain MRI (no_tumor) — **FIXED**
```
Detected:       MRI
Confidence:     99.6%
Top Prediction: No Tumor: 100.0%
```

### Chest X-ray (pneumonia) — **No Regression**
```
Detected:       XRAY
Confidence:     100.0%
Top Prediction: Pneumonia: 100.0%
```

> [!TIP]
> Both the backend (port 8000) and frontend (port 5173) are running. You can test by uploading the same `Tr-no_99.jpg` image through the UI now.
