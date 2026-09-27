import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from banana_ai.ml.data import build_loaders
from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import configure_cpu_threads, get_device
from banana_ai.config import settings


def main(dataset_root: str = settings.dataset_root):
    checkpoint_path = Path("models/banana_cnn_best.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run: python -m banana_ai.ml.train"
        )

    configure_cpu_threads()
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = checkpoint["class_names"]
    model = BananaCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader, _ = build_loaders(
        dataset_root=dataset_root,
        batch_size=16,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(predictions)
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    Path("reports").mkdir(exist_ok=True)

    Path("reports/test_metrics.json").write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "classification_report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix)
    ax.set_title("Banana Ripeness Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30)
    ax.set_yticks(range(len(class_names)), class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the BananaAI classifier.")
    parser.add_argument(
        "--dataset-root",
        default=settings.dataset_root,
        help="Directory containing train/, valid/ (or val/), and test/ folders.",
    )
    args = parser.parse_args()
    main(args.dataset_root)
