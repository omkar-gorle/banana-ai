import torch
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from banana_ai.ml.data import build_loaders
from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import get_device

def main():
    device = get_device()
    _, valid_loader, _, class_names = build_loaders("D:/ff/banana/banana_classification_v3", batch_size=32)
    
    model = BananaCNN(num_classes=len(class_names))
    checkpoint = torch.load("models/banana_cnn_v3.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    
    y_true = []
    y_pred_base = []
    y_pred_tta = []
    
    print("Evaluating Test-Time Augmentation (TTA)...")
    
    # Timing
    t0 = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images = images.to(device)
            y_true.extend(labels.cpu().numpy())
            
            # Base prediction
            logits_base = model(images)
            probs_base = torch.softmax(logits_base, dim=1)
            y_pred_base.extend(probs_base.cpu().numpy())
            
            # TTA: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.softmax(logits_flipped, dim=1)
            
            # TTA: Add slight brightness variation by scaling tensor 
            # Note: Tensors are normalized, but we can simulate by shifting means slightly
            images_bright = images + 0.1
            logits_bright = model(images_bright)
            probs_bright = torch.softmax(logits_bright, dim=1)
            
            # Average probabilities
            probs_tta = (probs_base + probs_flipped + probs_bright) / 3.0
            y_pred_tta.extend(probs_tta.cpu().numpy())
            
    t1 = time.time()
    
    acc_base = accuracy_score(y_true, np.argmax(y_pred_base, axis=1))
    acc_tta = accuracy_score(y_true, np.argmax(y_pred_tta, axis=1))
    
    latency = (t1 - t0) / len(valid_loader.dataset)
    
    print("\nTTA Results:")
    print(f"Base Accuracy: {acc_base:.4f}")
    print(f"TTA Accuracy:  {acc_tta:.4f}")
    print(f"Improvement:   {acc_tta - acc_base:+.4f}")
    print(f"Average Inference Latency (batch=32): {latency*1000:.1f} ms / image")

if __name__ == "__main__":
    main()
