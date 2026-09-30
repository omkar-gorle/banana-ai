import torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import get_device

def main():
    device = get_device()
    test_path = "D:/ff/banana/real_world_challenge_v2/test"
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = datasets.ImageFolder(test_path, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes
    
    model = BananaCNN(num_classes=len(class_names))
    checkpoint = torch.load("models/banana_cnn_v3.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    confidences = np.max(y_prob, axis=1)
    
    # Evaluate at chosen threshold (0.70)
    t = 0.70
    accepted = confidences >= t
    rejected = ~accepted
    
    total = len(y_true)
    n_accepted = accepted.sum()
    n_rejected = rejected.sum()
    
    overall_acc = accuracy_score(y_true, y_pred)
    selective_acc = accuracy_score(y_true[accepted], y_pred[accepted]) if n_accepted > 0 else 0
    coverage = n_accepted / total
    
    rotten_idx = class_names.index("rotten")
    ripe_idx = class_names.index("ripe")
    unripe_idx = class_names.index("unripe")
    overripe_idx = class_names.index("overripe")
    
    # Dangerous errors
    # Actual Rotten -> Predicted Ripe/Unripe/Overripe
    rotten_mask = (y_true == rotten_idx)
    def count_dangerous(mask, y_p):
        r_r = np.sum((mask) & (y_p == ripe_idx))
        r_u = np.sum((mask) & (y_p == unripe_idx))
        r_o = np.sum((mask) & (y_p == overripe_idx))
        return r_r, r_u, r_o
        
    r_r_base, r_u_base, r_o_base = count_dangerous(rotten_mask, y_pred)
    r_r_sel, r_u_sel, r_o_sel = count_dangerous(rotten_mask & accepted, y_pred[accepted] if n_accepted > 0 else np.array([]))
    
    print("UNTUNED CLEAN TEST EVALUATION (CHALLENGE V2)")
    print(f"Total Images: {total}")
    print(f"Base Accuracy: {overall_acc:.2%}")
    print(f"Threshold Applied: {t}")
    print(f"Accepted: {n_accepted} | Rejected: {n_rejected} | Coverage: {coverage:.2%}")
    print(f"Selective Accuracy: {selective_acc:.2%}")
    print(f"Dangerous Errors (Rotten -> Ripe): Base={r_r_base} | With Abstention={r_r_sel}")
    print(f"Dangerous Errors (Rotten -> Unripe): Base={r_r_base} | With Abstention={r_u_sel}")
    print(f"Dangerous Errors (Rotten -> Overripe): Base={r_o_base} | With Abstention={r_o_sel}")

if __name__ == "__main__":
    main()
