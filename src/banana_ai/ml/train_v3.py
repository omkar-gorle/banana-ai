import json
import argparse
import time
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from banana_ai.ml.data import build_loaders
from banana_ai.ml.device import configure_cpu_threads, get_device
from banana_ai.ml.train_v2 import compute_class_weights, seed_everything
from banana_ai.ml.model import BananaCNN

def run_epoch(model, loader, loss_fn, optimizer, device, training=True):
    model.train(training)
    total_loss = 0.0
    all_true = []
    all_pred = []

    # Limit batches if we just want a quick evaluation, but we need full for real metrics
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)

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


def main(dataset_root: str):
    seed_everything()
    Path("models").mkdir(exist_ok=True)
    Path("reports/v3").mkdir(parents=True, exist_ok=True)

    configure_cpu_threads()
    device = get_device()
    
    # Increase batch size to 64 for faster processing on CPU (fewer updates)
    train_loader, val_loader, _, class_names = build_loaders(
        dataset_root=dataset_root,
        batch_size=64,
    )

    class_weights = compute_class_weights(train_loader.dataset).to(device)
    
    model = BananaCNN(num_classes=len(class_names))
    # Load V2 weights as starting point (fine-tuning) to converge much faster
    checkpoint = torch.load("models/banana_cnn_v2.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # Just train for 3 epochs since it's already pre-trained on V2
    epochs = 3
    best_f1 = -1.0

    history = []
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        t0 = time.perf_counter()

        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, loss_fn, optimizer, device, False)

        dur = time.perf_counter() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "train_macro_f1": train_f1,
            "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_f1, "duration": dur
        })

        print(f"train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": 224,
                "model_version": "banana-cnn-v3",
                "best_val_macro_f1": best_f1,
                "experiment": "finetune_v2_with_archive2_and_weights"
            }, "models/banana_cnn_v3.pt")
            print("Saved new best V3 checkpoint.")

    with open("reports/v3/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main(r"D:\ff\banana\banana_classification_v3")
