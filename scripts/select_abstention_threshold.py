import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from banana_ai.ml.data import build_loaders
from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import get_device

def main():
    device = get_device()
    _, valid_loader, _, class_names = build_loaders(
        dataset_root="D:/ff/banana/banana_classification_v3",
        batch_size=64
    )
    
    model = BananaCNN(num_classes=len(class_names))
    checkpoint = torch.load("models/banana_cnn_v3.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    
    y_true = []
    y_prob = []
    
    print("Gathering predictions on INDEPENDENT VALIDATION SET (banana_classification_v3) for calibration...")
    print(f"Calibration Dataset Size: {len(valid_loader.dataset)} images")
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Analyze abstention
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    
    print("\nAbstention Analysis on Calibration Set:")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        accepted = confidences >= t
        if not np.any(accepted): continue
        coverage = accepted.mean()
        selective_acc = accuracy_score(y_true[accepted], predictions[accepted])
        print(f"Threshold: {t:.2f} | Coverage: {coverage:.2%} | Selective Accuracy: {selective_acc:.2%}")

if __name__ == "__main__":
    main()
