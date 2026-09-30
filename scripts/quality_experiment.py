import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from banana_ai.ml.data import build_loaders
from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import get_device

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def mean_brightness(image):
    return np.mean(image)

def main():
    device = get_device()
    _, valid_loader, _, class_names = build_loaders("D:/ff/banana/banana_classification_v3", batch_size=1)
    
    model = BananaCNN(num_classes=len(class_names))
    checkpoint = torch.load("models/banana_cnn_v3.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    
    correct_blur, correct_bright = [], []
    incorrect_blur, incorrect_bright = [], []
    
    print("Analyzing image quality vs accuracy...")
    dataset = valid_loader.dataset
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            img_tensor, label = dataset[i]
            path, _ = dataset.samples[i]
            
            # Run inference
            logits = model(img_tensor.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).item()
            is_correct = (pred == label)
            
            # Compute quality
            cv_img = cv2.imread(path)
            if cv_img is None: continue
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            blur = variance_of_laplacian(gray)
            bright = mean_brightness(gray)
            
            if is_correct:
                correct_blur.append(blur)
                correct_bright.append(bright)
            else:
                incorrect_blur.append(blur)
                incorrect_bright.append(bright)
                
    print("\nQuality Metrics Analysis:")
    print("--- BLUR (Laplacian Variance) ---")
    print(f"Correctly Classified: Mean = {np.mean(correct_blur):.1f}, Median = {np.median(correct_blur):.1f}")
    print(f"Incorrectly Classified: Mean = {np.mean(incorrect_blur):.1f}, Median = {np.median(incorrect_blur):.1f}")
    
    print("\n--- BRIGHTNESS (Mean Intensity) ---")
    print(f"Correctly Classified: Mean = {np.mean(correct_bright):.1f}, Median = {np.median(correct_bright):.1f}")
    print(f"Incorrectly Classified: Mean = {np.mean(incorrect_bright):.1f}, Median = {np.median(incorrect_bright):.1f}")
    
    blur_threshold = 50
    correct_rejected = sum(1 for b in correct_blur if b < blur_threshold) / len(correct_blur)
    incorrect_rejected = sum(1 for b in incorrect_blur if b < blur_threshold) / len(incorrect_blur)
    
    print(f"\nPotential Gate (Reject if blur < {blur_threshold}):")
    print(f"False Rejection Rate (Good images blocked): {correct_rejected:.1%}")
    print(f"Error Reduction (Bad images blocked): {incorrect_rejected:.1%}")
    
    if incorrect_rejected > correct_rejected * 3:
        print("Conclusion: Image Quality Gate IS useful.")
    else:
        print("Conclusion: Image Quality Gate is NOT sufficiently useful (too many false rejections).")

if __name__ == "__main__":
    main()
