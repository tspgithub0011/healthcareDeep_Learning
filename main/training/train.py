"""
Generic training loop with:
- Mixed precision (AMP) for RTX 3050
- ReduceLROnPlateau scheduler
- Early stopping
- Best model checkpoint saving
- Training history (loss/accuracy curves)
"""
import os
import time
import copy

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import DEVICE, TRAINING_CONFIG, TRAINED_MODELS_DIR


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    model_name: str,
    num_epochs: int = None,
    learning_rate: float = None,
    class_weights: torch.Tensor = None,
    use_amp: bool = None,
):
    """
    Train a model with early stopping, AMP, and scheduler.

    Args:
        model: PyTorch model (already on correct device)
        train_loader: training DataLoader
        val_loader: validation DataLoader
        model_name: name for saving weights (e.g., 'pneumonia')
        num_epochs: override max epochs
        learning_rate: override learning rate
        class_weights: optional class weights for imbalanced datasets
        use_amp: override mixed precision setting

    Returns:
        dict with training history and best model path
    """
    cfg = TRAINING_CONFIG
    epochs = num_epochs or cfg["epochs"]
    lr = learning_rate or cfg["learning_rate"]
    amp_enabled = use_amp if use_amp is not None else cfg["use_mixed_precision"]

    # Move model to device
    model = model.to(DEVICE)

    # Loss function (with optional class weights)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        print(f"⚖️  Using weighted loss (class weights: {class_weights.tolist()})")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg["scheduler_patience"],
        factor=cfg["scheduler_factor"],
        verbose=True,
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=amp_enabled)

    # Early stopping
    early_stopping = EarlyStopping(patience=cfg["early_stopping_patience"])

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }
    best_val_acc = 0.0
    best_model_state = None
    save_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}.pth")

    print(f"\n{'='*60}")
    print(f"🏋️ Training: {model_name}")
    print(f"   Device: {DEVICE}")
    print(f"   Epochs: {epochs} | LR: {lr} | Batch: {cfg['batch_size']}")
    print(f"   AMP: {'ON' if amp_enabled else 'OFF'}")
    print(f"   Save to: {save_path}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(epochs):
        # ── Training Phase ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [TRAIN]",
            leave=False,
        )

        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.*correct/total:.1f}%",
            )

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # ── Validation Phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                with autocast(enabled=amp_enabled):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{epochs} │ "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% │ "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:6.2f}% │ "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            print(f"  💾 Best model saved! (val_acc: {val_acc:.2f}%)")

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
            break

    # Final stats
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*60}")
    print(f"✅ Training complete: {model_name}")
    print(f"   Time: {minutes}m {seconds}s")
    print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {save_path}")
    print(f"{'='*60}\n")

    # Load best model weights back
    model.load_state_dict(best_model_state)

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "save_path": save_path,
        "elapsed_seconds": elapsed,
    }
