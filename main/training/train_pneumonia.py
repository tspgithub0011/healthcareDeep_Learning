"""
Train Pneumonia Model (2-class: normal, pneumonia)
Architecture: EfficientNet-B0 with pretrained ImageNet weights
Dataset: Kaggle Chest X-ray Pneumonia
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import DATASETS_DIR, set_seed, build_model
from dataset import create_data_loaders
from train import train_model
from evaluate import evaluate_model

MODEL_NAME = "pneumonia"
DATASET_DIR = os.path.join(DATASETS_DIR, "chest_xray")
ARCHITECTURE = "efficientnet_b0"
NUM_CLASSES = 2

def main():
    set_seed(42)

    train_loader, val_loader, test_loader, info = create_data_loaders(DATASET_DIR)
    print(f"   Classes: {info['classes']}")

    model = build_model(ARCHITECTURE, NUM_CLASSES, pretrained=True)

    # Pneumonia dataset is imbalanced — use class weights
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=MODEL_NAME,
        class_weights=info["class_weights"],
    )

    evaluate_model(model, test_loader, info["classes"], MODEL_NAME)


if __name__ == "__main__":
    main()
