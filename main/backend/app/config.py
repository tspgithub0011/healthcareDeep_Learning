import os
from dotenv import load_dotenv
from torchvision import transforms

load_dotenv()


class Settings:
    # Server
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    ENV: str = os.getenv("ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # CORS
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # Models
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./trained_models")
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"

    # Limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 10))
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

    # Allowed file types
    ALLOWED_MIME_TYPES: list = ["image/jpeg", "image/png", "image/webp"]
    MIN_IMAGE_DIMENSION: int = 32

    # Risk thresholds
    HIGH_RISK_THRESHOLD: float = 0.7
    MEDIUM_RISK_THRESHOLD: float = 0.4

    # App info
    VERSION: str = "1.0.0"
    DISCLAIMER: str = (
        "This is an AI-assisted screening tool. Results are NOT a medical diagnosis. "
        "Please consult a qualified healthcare professional."
    )


settings = Settings()

# ── ImageNet Normalization (shared across all models) ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference transforms (no augmentation — resize, center crop, normalize)
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Model Configuration ──
# Maps model_name → {architecture, num_classes, class_names}
MODEL_CONFIG = {
    # IMPORTANT: class_names MUST be in alphabetical order
    # because the training dataset.py sorts folder names alphabetically.
    "image_classifier": {
        "architecture": "efficientnet_b0",
        "num_classes": 4,
        "class_names": ["ct_scan", "mri", "skin", "xray"],
    },
    "brain_tumor": {
        "architecture": "resnet50",
        "num_classes": 4,
        "class_names": ["glioma", "meningioma", "no_tumor", "pituitary"],
    },
    "pneumonia": {
        "architecture": "efficientnet_b0",
        "num_classes": 2,
        "class_names": ["normal", "pneumonia"],
    },
    "covid": {
        "architecture": "efficientnet_b0",
        "num_classes": 3,
        "class_names": ["covid", "normal", "viral_pneumonia"],
    },
    "skin_lesion": {
        "architecture": "efficientnet_b0",
        "num_classes": 7,
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    },
    "lung_cancer": {
        "architecture": "resnet50",
        "num_classes": 3,
        "class_names": ["benign", "malignant", "normal"],
    },
    "cardiomegaly": {
        "architecture": "efficientnet_b0",
        "num_classes": 2,
        "class_names": ["cardiomegaly", "normal"],
    },
}

# ── Disease Display Names (for API response) ──
# Normal classes use "No [Disease]" naming to disambiguate across models
DISEASE_DISPLAY_NAMES = {
    "brain_tumor": {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary Tumor",
        "no_tumor": "No Tumor",
    },
    "pneumonia": {
        "normal": "No Pneumonia",
        "pneumonia": "Pneumonia",
    },
    "covid": {
        "normal": "No COVID-19",
        "covid": "COVID-19",
        "viral_pneumonia": "Viral Pneumonia",
    },
    "skin_lesion": {
        "akiec": "Actinic Keratosis",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevus",
        "vasc": "Vascular Lesion",
    },
    "lung_cancer": {
        "normal": "No Lung Cancer",
        "benign": "Benign Tumor",
        "malignant": "Malignant Tumor",
    },
    "cardiomegaly": {
        "normal": "No Cardiomegaly",
        "cardiomegaly": "Cardiomegaly",
    },
}

# ── Classes that represent "healthy/normal" — risk status is inverted ──
# High probability of these = LOW risk (good news), not high risk
NORMAL_CLASSES = {"normal", "no_tumor"}

# ── Image Type → Disease Models Routing ──
IMAGE_TYPE_ROUTES = {
    "xray": ["pneumonia", "covid", "cardiomegaly"],
    "mri": ["brain_tumor"],
    "ct_scan": ["lung_cancer"],
    "skin": ["skin_lesion"],
}
