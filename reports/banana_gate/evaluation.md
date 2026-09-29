# Banana Gate Binary Classifier Evaluation Report

Date: 2026-09-30
Model Architecture: `BananaGateMobileNetV3`
Checkpoint: `models/banana_gate_best.pt`
Checkpoint SHA-256: `a1416daae0ae22080af79436ae8c97b59ebcd301af28fddb8ce19dff5929789d`
Frozen Ripeness Model (`models/banana_cnn_v2.pt`) SHA-256: `cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d` (VERIFIED UNCHANGED)

---

## 1. Executive Summary

A dedicated binary classifier (**BANANA vs. NON-BANANA**) was trained to serve as the production gate for the Banana AI system.
The gate completely replaces the legacy COCO detector with a lightweight, deterministic neural network optimized for fast CPU inference (<15ms).

Any non-banana image is rejected immediately at the gate and **NEVER** reaches the frozen ripeness model (`banana_cnn_v2.pt`), Grad-CAM, shelf-life estimation, or PostgreSQL prediction storage.

---

## 2. Threshold Selection (Validation Set Only)

The production threshold was tuned strictly on **held-out validation data** (400 banana / 400 non-banana images) with the following hierarchy:
1. **High Banana Recall** (target >= 98.0%): Ensure genuine bananas of all 4 stages are rarely misclassified.
2. **Strong Non-Banana Rejection** (specificity target >= 95.0%): Ensure other fruits, objects, and scenes are rejected.
3. **F1 Optimization**: Balance precision and recall.

- **Selected Production Threshold:** `0.380`
- **Validation Accuracy:** `99.88%`
- **Validation Banana Recall:** `100.00%`
- **Validation Non-Banana Rejection (Specificity):** `99.75%`
- **Validation F1 Score:** `0.9988`
- **Validation ROC-AUC:** `1.0000`

---

## 3. Final Evaluation on Held-Out Test Set

The held-out test set (300 banana / 300 non-banana images) was evaluated **ONCE** using the selected threshold (`0.380`):

| Metric | Score |
|---|---|
| **Accuracy** | **99.67%** |
| **Banana Recall (TPR)** | **100.00%** |
| **Non-Banana Rejection Rate (TNR / Specificity)** | **99.33%** |
| **Precision** | **99.34%** |
| **F1 Score** | **0.9967** |
| **ROC-AUC** | **1.0000** |
| **False Positive Rate (FPR)** | **0.67%** |
| **False Negative Rate (FNR)** | **0.00%** |

### Confusion Matrix (Test Set: 600 Images)

| | Predicted Non-Banana (0) | Predicted Banana (1) | Total Actual |
|---|---|---|---|
| **Actual Non-Banana (0)** | **TN = 298** | **FP = 2** | 300 |
| **Actual Banana (1)** | **FN = 0** | **TP = 300** | 300 |

- **True Positives (TP):** `300` (correctly accepted bananas)
- **True Negatives (TN):** `298` (correctly rejected non-bananas)
- **False Positives (FP):** `2` (non-bananas falsely accepted)
- **False Negatives (FN):** `0` (bananas falsely rejected)

Confusion Matrix visualization saved to: `reports/banana_gate/confusion_matrix.png`

---

## 4. Inference Safety & Gating Pipeline

```
Camera / Upload / Batch
       ↓
Image File Validation (MIME / dimensions / decoding)
       ↓
  BANANA GATE (BananaGateMobileNetV3, threshold = 0.38)
   /          \
[Banana]    [Non-Banana]
   ↓             ↓
banana_cnn_v2  REJECT & STOP
   ↓          (No ripeness, No Grad-CAM, No shelf-life, No DB record)
Ripeness
   ↓
Grad-CAM
   ↓
Shelf-life
   ↓
Save / Analytics
```
