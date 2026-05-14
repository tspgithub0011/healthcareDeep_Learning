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
