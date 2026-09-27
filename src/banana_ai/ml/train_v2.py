"""V2 class-weight experiment.

This trains the SAME BananaCNN architecture on the SAME dataset/split
but adds class weights to CrossEntropyLoss to handle imbalance.

V2 must only run AFTER V1 has completed and been evaluated.
"""

import json
import random
import argparse
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from tqdm import tqdm

from banana_ai.ml.data import build_loaders
from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import configure_cpu_threads, get_device
from banana_ai.config import settings


SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
PATIENCE = 5


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = Counter()
    for _, label in dataset:
        if isinstance(label, int):
            counts[label] += 1
        else:
            counts[int(label)] += 1

    # Handle ImageFolder and Subset
    if hasattr(dataset, 'targets'):
        counts = Counter(dataset.targets)
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # For Subset
        counts = Counter()
        for idx in dataset.indices:
            counts[dataset.dataset.targets[idx]] += 1
    else:
        # Fallback: iterate (slow but correct)
        counts = Counter()
        for _, label in dataset:
            counts[int(label)] += 1

    total = sum(counts.values())
    num_classes = len(counts)
    weights = []
    for i in range(num_classes):
        weights.append(total / (num_classes * counts[i]))

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, loss_fn, optimizer, device, training=True):
    model.train(training)
    total_loss = 0.0
    all_true = []
    all_pred = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


def main(dataset_root: str = settings.dataset_root):
    seed_everything()

    Path("models").mkdir(exist_ok=True)
    Path("reports/v2").mkdir(parents=True, exist_ok=True)

    configure_cpu_threads()
    device = get_device()
    print(f"Using device: {device}")
    print("=" * 50)
    print("V2 EXPERIMENT: Class-Weight Balancing")
    print("=" * 50)

    train_loader, val_loader, _, class_names = build_loaders(
        dataset_root=dataset_root,
        batch_size=BATCH_SIZE,
    )

    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    # Compute class weights from training data
    class_weights = compute_class_weights(train_loader.dataset).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    model = BananaCNN(num_classes=len(class_names)).to(device)

    # V2 difference: weighted CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        epoch_started_at = time.perf_counter()

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, loss_fn, optimizer, device, training=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, loss_fn, optimizer, device, training=False
        )

        scheduler.step(val_loss)
        epoch_duration_seconds = time.perf_counter() - epoch_started_at

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_duration_seconds": epoch_duration_seconds,
        }
        history.append(record)

        print(
            f"train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
            f"duration={epoch_duration_seconds:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": 224,
                    "model_version": "banana-cnn-v2",
                    "best_val_macro_f1": best_f1,
                    "experiment": "class-weight-balancing",
                    "class_weights": class_weights.cpu().tolist(),
                },
                "models/banana_cnn_v2.pt",
            )
            print("Saved new best V2 checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation macro-F1 stopped improving.")
            break

    Path("reports/v2/training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print(f"\nBest V2 validation macro-F1: {best_f1:.4f}")
    print("Checkpoint: models/banana_cnn_v2.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V2 class-weight experiment.")
    parser.add_argument(
        "--dataset-root",
        default=settings.dataset_root,
        help="Directory containing train/, valid/ (or val/), and test/ folders.",
    )
    args = parser.parse_args()
    main(args.dataset_root)
