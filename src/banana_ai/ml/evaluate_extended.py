"""Extended evaluation: generates detailed metrics, confusion matrices,
and per-class reports for any checkpoint. Supports versioned output dirs.
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from banana_ai.ml.data import build_loaders
from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import configure_cpu_threads, get_device
from banana_ai.config import settings


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_root: str,
    output_dir: str,
    version_label: str = "v1",
):
    """Evaluate a checkpoint against the test set and save reports."""

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"No trained model found at {checkpoint_path}. Train first."
        )

    configure_cpu_threads()
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = checkpoint["class_names"]
    model = BananaCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    test_path = Path(dataset_root) / "test"
    if not test_path.exists():
        test_path = Path(dataset_root)
        
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    test_dataset = datasets.ImageFolder(str(test_path), transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {version_label}"):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(predictions)
            y_true.extend(labels.numpy())
            y_probs.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    # Normalized confusion matrix
    matrix_norm = matrix.astype(float)
    row_sums = matrix_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix_norm / row_sums

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics = {
        "version": version_label,
        "model_version": checkpoint.get("model_version", "unknown"),
        "checkpoint_path": checkpoint_path,
        "best_val_macro_f1": checkpoint.get("best_val_macro_f1", None),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": matrix_norm.tolist(),
    }

    (out / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Save text report
    (out / "classification_report.txt").write_text(
        f"Test Evaluation: {version_label}\n"
        f"Model: {checkpoint.get('model_version', 'unknown')}\n"
        f"Checkpoint: {checkpoint_path}\n\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n\n"
        f"{report_text}\n",
        encoding="utf-8",
    )

    # Confusion matrix plot
    _plot_confusion_matrix(
        matrix, class_names,
        f"Confusion Matrix — {version_label}",
        out / "confusion_matrix.png",
    )

    # Normalized confusion matrix
    _plot_confusion_matrix(
        matrix_norm, class_names,
        f"Normalized Confusion Matrix — {version_label}",
        out / "confusion_matrix_normalized.png",
        fmt=".2f",
    )

    # Probability distribution
    _plot_probability_distribution(
        y_probs, y_true, y_pred, class_names,
        out / "probability_distribution.png",
    )

    # Incorrect predictions
    incorrect_indices = np.where(y_true != y_pred)[0]
    incorrect_examples = []
    for idx in incorrect_indices[:20]:
        incorrect_examples.append({
            "index": int(idx),
            "true_class": class_names[y_true[idx]],
            "predicted_class": class_names[y_pred[idx]],
            "confidence": float(y_probs[idx].max()),
            "probabilities": {
                class_names[i]: float(y_probs[idx][i])
                for i in range(len(class_names))
            },
        })

    (out / "incorrect_predictions.json").write_text(
        json.dumps(incorrect_examples, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*50}")
    print(f"TEST EVALUATION: {version_label}")
    print(f"{'='*50}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")
    print(f"\n{report_text}")
    print(f"Reports saved to: {out}")

    return metrics


def _plot_confusion_matrix(matrix, class_names, title, save_path, fmt="d"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = f"{val:{fmt}}" if isinstance(val, (int, np.integer)) else f"{val:.2f}"
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=160)
    plt.close(fig)


def _plot_probability_distribution(y_probs, y_true, y_pred, class_names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confidence distribution for correct vs incorrect
    correct_mask = y_true == y_pred
    confidences = y_probs.max(axis=1)

    axes[0].hist(
        confidences[correct_mask],
        bins=30, alpha=0.7, label="Correct", color="green"
    )
    axes[0].hist(
        confidences[~correct_mask],
        bins=30, alpha=0.7, label="Incorrect", color="red"
    )
    axes[0].set_title("Prediction Confidence Distribution")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Per-class confidence
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.any():
            class_confs = y_probs[mask, i]
            axes[1].hist(class_confs, bins=20, alpha=0.5, label=name)

    axes[1].set_title("Per-Class Probability Distribution")
    axes[1].set_xlabel("Probability for True Class")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended evaluation.")
    parser.add_argument(
        "--checkpoint", default="models/banana_cnn_best.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--dataset-root", default=settings.dataset_root,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--output-dir", default="reports/v1",
        help="Directory to save reports.",
    )
    parser.add_argument(
        "--version-label", default="v1",
        help="Version label for the experiment.",
    )
    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint, args.dataset_root, args.output_dir, args.version_label
    )
