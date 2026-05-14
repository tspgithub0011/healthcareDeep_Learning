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
