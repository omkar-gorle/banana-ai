import json
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from banana_ai.ml.model import BananaCNN
from torchvision.models import mobilenet_v2
from torch import nn
from banana_ai.ml.device import configure_cpu_threads, get_device

def evaluate_rw(checkpoint_path: str, rw_dir: str, output_dir: str, version_label: str):
    configure_cpu_threads()
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    class_names = checkpoint["class_names"]
    
    if "mobilenet" in checkpoint.get("model_version", ""):
        model = mobilenet_v2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    else:
        model = BananaCNN(num_classes=len(class_names))
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    ds = datasets.ImageFolder(rw_dir, transform=eval_transform)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    
    y_true = []
    y_pred = []
    y_probs = []
    
    high_conf_errors = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Eval RW {version_label}"):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()
            
            y_pred.extend(predictions)
            y_true.extend(labels.numpy())
            y_probs.extend(probs)
            
            for p, true_l, prb in zip(predictions, labels.numpy(), probs):
                total += 1
                # If predicted banana class is wrong and confidence is > 90%
                # Note: true_l maps to ds.classes! We need to map ds.classes to class_names if they match
                pass
                
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Actually, the real-world dataset has folders like banana_ripe, banana_unripe, hands, etc.
    # We should map 'banana_ripe' -> 'ripe'.
    # If the folder is 'hands' or 'yellow_objects', it's NOT a ripeness classification task!
    # But wait, the banana gate should block them! The ripeness model only sees bananas!
    # So for the ripeness model evaluation, we ONLY evaluate on banana_* folders.
    
    # So let's re-run only on banana folders
    pass

if __name__ == "__main__":
    # Simplified script just to get high-confidence error rate on the valid classes
    pass
