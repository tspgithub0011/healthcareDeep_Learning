"""
Train Brain Tumor Model (4-class: glioma, meningioma, no_tumor, pituitary)
Architecture: ResNet50 with pretrained ImageNet weights
Dataset: Kaggle Brain Tumor MRI
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import DATASETS_DIR, set_seed, build_model
from dataset import create_data_loaders
from train import train_model
from evaluate import evaluate_model

MODEL_NAME = "brain_tumor"
DATASET_DIR = os.path.join(DATASETS_DIR, "brain_tumor")
ARCHITECTURE = "resnet50"
NUM_CLASSES = 4

def main():
    set_seed(42)

    train_loader, val_loader, test_loader, info = create_data_loaders(DATASET_DIR)
    print(f"   Classes: {info['classes']}")

    model = build_model(ARCHITECTURE, NUM_CLASSES, pretrained=True)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=MODEL_NAME,
    )

    evaluate_model(model, test_loader, info["classes"], MODEL_NAME)


if __name__ == "__main__":
    main()
