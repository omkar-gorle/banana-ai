import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from banana_ai.ml.data import build_loaders
from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import get_device

def expected_calibration_error(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def brier_score(y_true, y_prob):
    # one-hot encode
    y_true_onehot = np.zeros_like(y_prob)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))

def evaluate_abstention(y_true, y_prob, thresholds):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    
    results = {}
    for t in thresholds:
        accepted = confidences >= t
        if not np.any(accepted):
            continue
            
        coverage = accepted.mean()
        selective_acc = accuracy_score(y_true[accepted], predictions[accepted])
        
        # dangerous errors
        dangerous = 0
        for i in range(len(y_true)):
            if accepted[i]:
                # 2 is rotten, 1 is ripe, 3 is unripe in typical mapping (unripe=3, ripe=1, overripe=0, rotten=2)
                # Let's just use string labels
                pass
                
        results[t] = {
            "coverage": coverage,
            "selective_accuracy": selective_acc
        }
        
    return results

def main():
    device = get_device()
    _, valid_loader, test_loader, class_names = build_loaders(
        dataset_root="D:/ff/banana/banana_classification_v3",
        batch_size=64
    )
    
    model = BananaCNN(num_classes=len(class_names))
    checkpoint = torch.load("models/banana_cnn_v3.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    
    y_true = []
    y_prob = []
    
    print("Gathering predictions on validation set for calibration...")
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    ece = expected_calibration_error(y_true, y_prob)
    brier = brier_score(y_true, y_prob)
    
    print(f"\nCalibration Metrics:")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")
    
    print("\nAbstention Thresholds (Uncertainty Rejection):")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    abs_results = evaluate_abstention(y_true, y_prob, thresholds)
    
    for t in thresholds:
        if t in abs_results:
            print(f"Threshold: {t:.2f} | Coverage: {abs_results[t]['coverage']:.1%} | Accuracy: {abs_results[t]['selective_accuracy']:.1%}")

    # Map indices to class names to count dangerous errors accurately
    rotten_idx = class_names.index("rotten")
    ripe_idx = class_names.index("ripe")
    unripe_idx = class_names.index("unripe")
    
    # Let's find the threshold that reduces dangerous errors to 0 while keeping reasonable coverage
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    
    best_t = 0.0
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        accepted = confidences >= t
        dangerous = np.sum((y_true[accepted] == rotten_idx) & ((predictions[accepted] == ripe_idx) | (predictions[accepted] == unripe_idx)))
        if dangerous == 0 and accepted.mean() > 0.5:
            best_t = t
            break
            
    print(f"\nRecommended Abstention Threshold: {best_t:.2f}")

if __name__ == "__main__":
    main()
