"""
Model Weight Downloader — runs before server startup in production.

Downloads trained .pth model weights from HuggingFace Hub if they are not
already present in the local trained_models/ directory.

Usage:
    python scripts/download_models.py

Environment Variables:
    HF_REPO_ID   — HuggingFace repository ID (e.g., "AceFire09/healthcare-dl-models")
    MODEL_DIR    — Local directory for model weights (default: ./trained_models)
"""
import os
import sys

# ── Configuration ──
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


def download_models():
    """Download model weights from HuggingFace Hub if not present locally."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check which models are missing
    missing = []
    for filename in MODEL_FILES:
        local_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(local_path):
            missing.append(filename)
        else:
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  ✓ {filename} already exists ({size_mb:.1f} MB)")

    if not missing:
        print("\n✅ All model weights are present. Skipping download.")
        return

    print(f"\n⬇️  {len(missing)} model(s) need to be downloaded from HuggingFace Hub...")
    print(f"   Repository: {HF_REPO_ID}\n")

    # Import huggingface_hub only when needed
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ ERROR: huggingface_hub is not installed.")
        print("   Install it with: pip install huggingface_hub")
        print("   Or place model files manually in ./trained_models/")
        sys.exit(1)

    for filename in missing:
        local_path = os.path.join(MODEL_DIR, filename)
        print(f"  ⬇️  Downloading {filename}...", end=" ", flush=True)
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"✓ ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            print(f"\n❌ Could not download {filename}. Server will start with dummy weights.")
            # Don't exit — let the server start with random weights for that model


if __name__ == "__main__":
    print("=" * 50)
    print("📦 Healthcare DL — Model Weight Downloader")
    print("=" * 50)
    print(f"\nModel directory: {os.path.abspath(MODEL_DIR)}")
    print(f"HuggingFace repo: {HF_REPO_ID}\n")
    download_models()
    print("\n" + "=" * 50)
