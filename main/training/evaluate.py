"""
Model evaluation on held-out test set.
Computes accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
"""
import os
import numpy as np
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from config import DEVICE, TRAINING_CONFIG

# Try importing plotting libraries
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    class_names: list,
    model_name: str,
    save_dir: str = None,
):
    """
    Evaluate a trained model on the test set.

    Args:
        model: trained PyTorch model
        test_loader: test DataLoader
        class_names: list of class name strings
        model_name: name for saving outputs
        save_dir: directory to save confusion matrix plot

    Returns:
        dict with all metrics
    """
    model = model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    amp_enabled = TRAINING_CONFIG["use_mixed_precision"]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with autocast(enabled=amp_enabled):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ── Metrics ──
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    # ROC-AUC (one-vs-rest)
    try:
        if len(class_names) == 2:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1]) * 100
        else:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro") * 100
    except ValueError:
        roc_auc = 0.0

    # Print results
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"   Accuracy:  {accuracy:.2f}%")
    print(f"   Precision: {precision:.2f}% (macro)")
    print(f"   Recall:    {recall:.2f}% (macro)")
    print(f"   F1-Score:  {f1:.2f}% (macro)")
    print(f"   ROC-AUC:   {roc_auc:.2f}% (macro)")
    print(f"\n{'─'*60}")

    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("🔢 Confusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    if PLOTTING_AVAILABLE:
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\n📈 Confusion matrix saved to: {plot_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }
