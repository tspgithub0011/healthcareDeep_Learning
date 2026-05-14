"""
Custom PyTorch Dataset + data loading utilities.
Handles directory-based image datasets with automatic train/val/test splitting.
"""
import os
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import TRAINING_CONFIG, IMAGENET_MEAN, IMAGENET_STD


# ── Transforms ──
def get_train_transforms(input_size: tuple = (224, 224)):
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(input_size[0], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(input_size: tuple = (224, 224)):
    """Validation/Test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MedicalImageDataset(Dataset):
    """
    Loads images from a directory structure:
        root_dir/
            class_1/
                img1.jpg
                img2.png
            class_2/
                ...
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []   # list of (image_path, label_index)
        self.classes = []   # class names sorted alphabetically
        self.class_to_idx = {}

        self._scan_directory()

    def _scan_directory(self):
        """Scan root_dir for class folders and images."""
        # Get sorted class names from directory names
        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
            and not d.startswith(".")
            and d != "archive"
            and d != "__pycache__"
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls],
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_labels(self):
        """Return all labels (used for stratified split)."""
        return [label for _, label in self.samples]

    def get_class_weights(self):
        """
        Compute class weights for imbalanced datasets.
        Returns a tensor of weights inversely proportional to class frequency.
        """
        label_counts = Counter(self.get_labels())
        total = len(self.samples)
        num_classes = len(self.classes)
        weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)
            weights.append(total / (num_classes * count))
        return torch.FloatTensor(weights)


def create_data_loaders(
    dataset_dir: str,
    batch_size: int = None,
    num_workers: int = None,
    input_size: tuple = None,
):
    """
    Create train/val/test data loaders with stratified splitting.

    Args:
        dataset_dir: path to dataset root (contains class subdirectories)
        batch_size: override default batch size
        num_workers: override default num_workers
        input_size: override default input size

    Returns:
        (train_loader, val_loader, test_loader, dataset_info)
    """
    cfg = TRAINING_CONFIG
    bs = batch_size or cfg["batch_size"]
    nw = num_workers or cfg["num_workers"]
    size = input_size or cfg["input_size"]
    split = cfg["data_split"]

    # Create full dataset (no transforms yet — we apply per-split)
    full_dataset = MedicalImageDataset(dataset_dir)
    labels = full_dataset.get_labels()

    print(f"\n📊 Dataset: {dataset_dir}")
    print(f"   Total images: {len(full_dataset)}")
    print(f"   Classes: {full_dataset.classes}")
    class_counts = Counter(labels)
    for cls_name, cls_idx in full_dataset.class_to_idx.items():
        print(f"   • {cls_name}: {class_counts[cls_idx]} images")

    # Stratified split: train / (val + test)
    train_indices, temp_indices = train_test_split(
        range(len(full_dataset)),
        test_size=(split["val"] + split["test"]),
        stratify=labels,
        random_state=cfg["seed"],
    )

    # Split temp into val / test
    temp_labels = [labels[i] for i in temp_indices]
    val_ratio = split["val"] / (split["val"] + split["test"])
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=cfg["seed"],
    )

    print(f"   Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # Create subset datasets with appropriate transforms
    train_transform = get_train_transforms(size)
    val_transform = get_val_transforms(size)

    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)
    test_dataset = TransformSubset(full_dataset, test_indices, val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=cfg["pin_memory"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=cfg["pin_memory"],
    )

    dataset_info = {
        "classes": full_dataset.classes,
        "class_to_idx": full_dataset.class_to_idx,
        "num_classes": len(full_dataset.classes),
        "class_weights": full_dataset.get_class_weights(),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
    }

    return train_loader, val_loader, test_loader, dataset_info


class TransformSubset(Dataset):
    """A subset of a dataset with a specific transform applied."""

    def __init__(self, dataset: MedicalImageDataset, indices: list, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
