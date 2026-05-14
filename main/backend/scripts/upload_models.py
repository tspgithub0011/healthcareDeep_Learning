"""
Upload trained model weights to HuggingFace Hub.

Run this ONCE from your local machine to push all .pth files to your
HuggingFace repository. Then the production Dockerfile will download
them at startup.

Prerequisites:
    1. pip install huggingface_hub
    2. huggingface-cli login  (or set HF_TOKEN env var)
    3. Create a repo at https://huggingface.co/new

Usage:
    python scripts/upload_models.py

Environment Variables:
    HF_REPO_ID — Your HuggingFace repository ID (e.g., "AceFire09/healthcare-dl-models")
    MODEL_DIR  — Path to local trained_models/ directory
"""
import os
import sys

HF_REPO_ID = os.environ.get("HF_REPO_ID", "AceFire09/healthcare-dl-models")
MODEL_DIR = os.environ.get("MODEL_DIR", "./trained_models")

MODEL_FILES = [
    "image_classifier.pth",
    "brain_tumor.pth",
    "pneumonia.pth",
    "covid.pth",
    "skin_lesion.pth",
    "lung_cancer.pth",
    "cardiomegaly.pth",
]


def upload_models():
    """Upload all trained model weights to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ Please install huggingface_hub first:")
        print("   pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
        print(f"✓ Repository ready: https://huggingface.co/{HF_REPO_ID}\n")
    except Exception as e:
        print(f"⚠️  Could not create/verify repo: {e}")
        print("   Make sure you're logged in: huggingface-cli login\n")

    for filename in MODEL_FILES:
        local_path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(local_path):
            print(f"  ⚠️  {filename} — NOT FOUND, skipping")
            continue

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  ⬆️  Uploading {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
            print("✓ Done")
        except Exception as e:
            print(f"✗ FAILED: {e}")

    print(f"\n🎉 Upload complete! View at: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    print("=" * 50)
    print("📤 Healthcare DL — Model Weight Uploader")
    print("=" * 50)
    print(f"\nModel directory: {os.path.abspath(MODEL_DIR)}")
    print(f"Target HF repo: {HF_REPO_ID}\n")
    upload_models()
