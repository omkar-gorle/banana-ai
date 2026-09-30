import json
import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torchvision.models import mobilenet_v2
from torch import nn

from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import configure_cpu_threads, get_device

def main(checkpoint_path, rw_dir):
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
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # We load only the banana_* directories to test ripeness classification.
    # The real_world_banana_set has subfolders like banana_ripe. We rename them in a temporary structure or just evaluate manually.
    
    # Let's manually traverse the folders that start with "banana_"
    rw_path = Path(rw_dir)
    y_true = []
    y_pred = []
    y_probs = []
    
    for class_idx, name in enumerate(class_names):
        folder = rw_path / f"banana_{name}"
        if not folder.exists():
            continue
            
        for img_path in folder.glob("*.jpg"):
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = logits.argmax(dim=1).item()
                
            y_true.append(class_idx)
            y_pred.append(pred)
            y_probs.append(probs)
            
    if len(y_true) == 0:
        print("No valid banana_class images found in rw dataset.")
        return
        
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Real-World Test for {checkpoint_path}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rw-dir", required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.rw_dir)
