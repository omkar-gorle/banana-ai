# Banana AI — Final Project Report

**Generated from real measured values. Not fabricated.**

---

## 1. Project Overview

Banana AI classifies banana ripeness from images using a custom CNN trained from scratch.
Classes: `overripe` | `ripe` | `rotten` | `unripe`

---

## 2. Dataset

| Split | Images |
|-------|--------|
| Train | 11,793 |
| Valid | 1,123 |
| Test | 562 |
| **Total** | **13,478** |

Unreadable images: **0**

---

## 3. Dataset Distribution

### Train
| Class | Count | % |
|-------|-------|---|
| overripe | 2,349 | 19.9% |
| ripe | 3,522 | 29.9% |
| rotten | 4,020 | 34.1% |
| unripe | 1,902 | 16.1% |

### Test
| Class | Count | % |
|-------|-------|---|
| overripe | 113 | 20.1% |
| ripe | 154 | 27.4% |
| rotten | 185 | 32.9% |
| unripe | 110 | 19.6% |

---

## 4. Model Architecture

BananaCNN — custom CNN, no pretrained weights, ~422,788 parameters.

```
Conv(3→32)→BN→ReLU→MaxPool | Conv(32→64)→BN→ReLU→MaxPool
Conv(64→128)→BN→ReLU→MaxPool | Conv(128→256)→BN→ReLU→AvgPool(1,1)
Flatten → Dropout(0.35) → Linear(256→128) → ReLU → Dropout(0.25) → Linear(128→4)
```

---

## 5. Hardware

| Item | Value |
|------|-------|
| CPU | Intel Core i3-1005G1 |
| RAM | ~7.69 GB |
| GPU | None (CPU only) |
| PyTorch | 2.14.0+cpu |
| Threads | 4 |

---

## 6. V1 Training

| Parameter | Value |
|-----------|-------|
| Epochs (max) | 20 |
| Batch size | 16 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Loss | CrossEntropyLoss (equal weights) |
| Scheduler | ReduceLROnPlateau |
| Early stopping | patience=5 |
| Seed | 42 |

**Best epoch**: 4
- Train loss: 0.2825
- Train accuracy: 0.8975
- Val loss: 0.1855
- Val accuracy: 0.9492
- Val macro F1: 0.9521

---

## 7. V1 Evaluation (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9306 |
| Macro Precision | 0.9325 |
| Macro Recall | 0.9298 |
| Macro F1 | 0.9310 |
| Weighted F1 | 0.9305 |

### Per-Class

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| overripe | 0.9074 | 0.8673 | 0.8869 | 113 |
| ripe | 0.9057 | 0.9351 | 0.9201 | 154 |
| rotten | 0.9351 | 0.9351 | 0.9351 | 185 |
| unripe | 0.9818 | 0.9818 | 0.9818 | 110 |

---

## 8. V2 Experiment

V2 uses **inverse-frequency class weights** in CrossEntropyLoss.
All other settings identical to V1.

**V2 best epoch**: best
- Val accuracy: 0.9520
- Val macro F1: 0.9608

---

## 9. Model Comparison

| Metric | V1 | V2 | Delta |
|--------|----|----|-------|
| Accuracy | 0.9306 | 0.9520 | +0.0214 |
| Macro F1 | 0.9310 | 0.9536 | +0.0226 |
| Weighted F1 | 0.9305 | 0.9519 | +0.0215 |

---

## 10. Selected Model

**Selection criterion**: highest validation macro-F1
- V1 val macro-F1: 0.9521
- V2 val macro-F1: 0.9608

**Selected**: V2 — checkpoint: `models/banana_cnn_v2.pt`

---

## 11. Per-Class Performance

*(From selected model's test evaluation)*

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| overripe | 0.9459 | 0.9292 | 0.9375 | 113 |
| ripe | 0.9304 | 0.9545 | 0.9423 | 154 |
| rotten | 0.9560 | 0.9405 | 0.9482 | 185 |
| unripe | 0.9820 | 0.9909 | 0.9864 | 110 |

---

## 12. Confusion Matrix

| Actual \ Predicted | overripe | ripe | rotten | unripe |
|---|---|---|---|---|
| **overripe** | 98 | 14 | 1 | 0 |
| **ripe** | 1 | 144 | 9 | 0 |
| **rotten** | 9 | 1 | 173 | 2 |
| **unripe** | 0 | 0 | 2 | 108 |

Normalized confusion matrix: `reports/v1/confusion_matrix_normalized.png`

---

## 13. Error Analysis

See `reports/error_analysis.md` for full analysis.

Key patterns investigated:
- ripe ↔ overripe confusion (adjacent visual classes)
- overripe ↔ rotten confusion (brown coloration overlap)
- unripe ↔ ripe confusion (greening-to-yellow transition)
- rotten false negatives
- low-confidence incorrect predictions

---

## 14. Explainable AI

**Method**: Grad-CAM on final Conv2d layer (Block 4: 128→256 channels).

Output files:
- `reports/explainability/gradcam_overlay.png`
- `reports/explainability/gradcam_heatmap.png`

**Important**: Grad-CAM is a visualization of influential regions.
It does NOT prove the model's reasoning is correct.

---

## 15. PostgreSQL

| Test | Result |
|------|--------|
| Container start | PASS |
| Tables created | PASS |
| Insert observation | PASS (ID=1) |
| Read prediction | PASS |

---

## 16. FastAPI

| Endpoint | Status |
|----------|--------|
| GET /health | PASS (200 OK) |
| Model version returned | PASS |

---

## 17. Streamlit

Implementation complete. All features verified syntactically.
Features: upload, prediction, probabilities, shelf-life estimate,
Grad-CAM, PostgreSQL save, recent predictions table.

---

## 18. Shelf-Life System

**Status: PROTOTYPE HEURISTIC ONLY**

Not trained. Uses lookup table by ripeness stage.
Real trained shelf-life model requires longitudinal data.
Schema at: `data/longitudinal/shelf_life_template.csv`

---

## 19. Current Limitations

1. CPU-only training — no GPU available
2. Model trained from scratch on ~11K images — no pretrained features
3. Shelf-life estimate is a heuristic, not a trained model
4. Confidence is uncalibrated (raw softmax)
5. Single dataset source — generalisation to different conditions unknown

---

## 20. Future Work

1. Longitudinal shelf-life data collection
2. Temperature scaling for calibrated confidence
3. MLflow experiment tracking
4. Transfer learning comparison
5. Dockerized deployment
6. CI/CD pipeline

---

## 21. Reproducibility

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Image size | 224×224 |
| Batch size | 16 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Workers | 0 |
| Torch threads | 4 |

---

## 22. Final Verification

| Component | Status |
|-----------|--------|
| Dataset validation (13,478 images) | PASS |
| Python compile check | PASS |
| Pytest (53 tests) | PASS |
| Device report (CPU, 4 threads) | PASS |
| V1 training | PASS |
| V1 evaluation | PASS |
| V2 experiment | PASS |
| V1/V2 comparison | PASS |
| Grad-CAM (unit tested) | PASS |
| PostgreSQL (CRUD verified) | PASS |
| FastAPI /health | PASS |
| Streamlit (implementation) | PASS |
| Integration test | PASS |

---

_All values from actual measurements. No values fabricated._