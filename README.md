# Banana AI — From-Scratch Ripeness Classifier

A complete ML pipeline that classifies banana ripeness from images using a **custom CNN trained from scratch** (no pretrained weights).

## Classes

```text
overripe | ripe | rotten | unripe
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| ML | PyTorch 2.14 (CPU) |
| Computer Vision | torchvision, OpenCV, Pillow |
| Data | NumPy, Pandas, scikit-learn |
| Database | PostgreSQL 18 + SQLAlchemy 2.1 + Psycopg 3 |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Testing | pytest |
| Infrastructure | Docker Compose |

---

## Dataset

```text
D:/ff/banana/banana_classification/
├── train/   — 11,793 images
├── valid/   —  1,123 images
└── test/    —    562 images
Total:        13,478 images
```

**Class distribution (train):**

| Class | Count | % |
|-------|-------|---|
| overripe | 2,349 | 19.9% |
| ripe | 3,522 | 29.9% |
| rotten | 4,020 | 34.1% |
| unripe | 1,902 | 16.1% |

Notable imbalance: rotten (~34%) vs unripe (~16%). This motivated the V2 class-weight experiment.

---

## Model Architecture — BananaCNN

Custom CNN trained from scratch. No ImageNet weights.

```text
Conv(3→32) → BN → ReLU → MaxPool
Conv(32→64) → BN → ReLU → MaxPool
Conv(64→128) → BN → ReLU → MaxPool
Conv(128→256) → BN → ReLU → AdaptiveAvgPool(1,1)
Flatten → Dropout(0.35) → Linear(256→128) → ReLU → Dropout(0.25) → Linear(128→4)
Parameters: ~422,788
```

---

## Experiments & Results

| Version | Loss function | Best Val Macro-F1 | Test Accuracy | Test Macro F1 | Checkpoint |
|---------|--------------|-------------------|---------------|---------------|------------|
| V1 | CrossEntropy (equal weights) | 0.9521 | 93.06% | 0.9310 | `models/banana_cnn_best.pt` |
| **V2** | CrossEntropy (class weights) | **0.9608** | **95.20%** | **0.9536** | `models/banana_cnn_v2.pt` |

**Selected model: V2** — chosen by highest validation macro-F1 (not test-set F1).

### V2 Per-Class Test Performance (Selected Model)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| overripe | 0.946 | 0.929 | 0.938 | 113 |
| ripe | 0.930 | 0.955 | 0.942 | 154 |
| rotten | 0.956 | 0.941 | 0.948 | 185 |
| unripe | 0.982 | 0.991 | 0.986 | 110 |

Class weights applied in V2: overripe=1.255, ripe=0.837, rotten=0.733, unripe=1.550.

---

## Hardware

```text
CPU: Intel Core i3-1005G1
RAM: ~7.69 GB
GPU: None (CPU training)
PyTorch: 2.14.0+cpu
Torch threads: 4
DataLoader workers: 0
```

Training V1: ~9 epochs (~4.5 hours on CPU). V2: same configuration.

---

## Quick Start

### 1. Create environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure

```powershell
copy .env.example .env
# Edit .env — set DATASET_ROOT to your banana_classification path
```

### 3. Check device

```powershell
python -m banana_ai.ml.device
```

### 4. Validate dataset

```powershell
python -m banana_ai.ml.dataset_report --dataset-root ..\banana_classification
```

### 5. Train V1 baseline

```powershell
python -m banana_ai.ml.train --dataset-root ..\banana_classification
```

### 6. Evaluate V1 on test set

```powershell
python -m banana_ai.ml.evaluate_extended `
    --checkpoint models/banana_cnn_best.pt `
    --dataset-root ..\banana_classification `
    --output-dir reports/v1 --version-label v1
```

### 7. Train V2 (class-weight experiment)

```powershell
python -m banana_ai.ml.train_v2 --dataset-root ..\banana_classification
```

### 8. Evaluate V2

```powershell
python -m banana_ai.ml.evaluate_extended `
    --checkpoint models/banana_cnn_v2.pt `
    --dataset-root ..\banana_classification `
    --output-dir reports/v2 --version-label v2
```

### 9. Compare models

```powershell
python -m banana_ai.ml.analysis --v1-dir reports/v1 --v2-dir reports/v2
```

### 10. Run prediction on one image (production default: V2)

```powershell
python -m banana_ai.ml.predict --image "path\to\banana.jpg" --model models/banana_cnn_v2.pt
```

### 11. Generate Grad-CAM

```powershell
python -m banana_ai.ml.explainability `
    --image "path\to\banana.jpg" `
    --model models/banana_cnn_v2.pt `
    --output-dir reports/explainability
```

### 12. Run the full post-training pipeline (steps 6–22 in one command)

```powershell
python scripts/post_training_pipeline.py
```

---

## PostgreSQL Setup

Docker Desktop must be running.

```powershell
docker compose up -d db
python -m banana_ai.db.init_db
```

Connection details:
```
Host:     localhost
Port:     5432
Database: banana_ai
User:     banana_user
Password: banana_password
```

**Change the password before any production use.**

---

## FastAPI

```powershell
$env:PYTHONPATH="src"
uvicorn banana_ai.api.main:app --reload
```

Endpoints:
- `GET  /health` — application health check
- `POST /predict` — classify a banana image

The API and Streamlit UI default to `models/banana_cnn_v2.pt`
(`banana-cnn-v2`). V1 remains available for historical training and
evaluation by passing `models/banana_cnn_best.pt` explicitly. In the UI,
analysis has no database side effect; use **Save prediction** to persist
exactly once. Softmax scores are raw, uncalibrated model outputs, and the
shelf-life value is a prototype heuristic.

API docs: http://127.0.0.1:8000/docs

Example response:
```json
{
  "predicted_stage": "ripe",
  "confidence": 0.91,
  "probabilities": {
    "overripe": 0.03,
    "ripe": 0.91,
    "rotten": 0.01,
    "unripe": 0.05
  },
  "model_version": "banana-cnn-v2",
  "estimated_days_left": "2-4 days",
  "shelf_life_warning": "estimated_days_left is a PROTOTYPE heuristic estimate based on the predicted ripeness stage. It is NOT a trained shelf-life model."
}
```

---

## Streamlit UI

```powershell
$env:PYTHONPATH="src"
streamlit run src/banana_ai/app.py
```

Features:
1. Upload banana image
2. Run CNN classification
3. View predicted ripeness with colour coding
4. View confidence score
5. View all class probabilities
6. View prototype shelf-life estimate (clearly labelled as heuristic)
7. Generate Grad-CAM explanation
8. Save observation to PostgreSQL
9. Browse recent prediction history

---

## Grad-CAM Explainability

Grad-CAM highlights which image regions most influenced the model's prediction. It uses the final Conv2d layer (Block 4: 128→256 channels).

```powershell
python -m banana_ai.ml.explainability `
    --image banana.jpg `
    --model models/banana_cnn_v2.pt `
    --output-dir reports/explainability
```

Output:
- `reports/explainability/gradcam_overlay.png` — original blended with heatmap
- `reports/explainability/gradcam_heatmap.png` — raw heatmap

**Grad-CAM is a visualisation tool, not proof of reasoning.**

---

## Shelf-Life System

### Current status: PROTOTYPE HEURISTIC

The dataset contains **ripeness class labels only** — not longitudinal `days_left` values.

The current estimate is a **lookup table only**:
```python
STAGE_RANGES = {
    "unripe":   (3, 7),   # days — HEURISTIC
    "ripe":     (2, 4),   # days — HEURISTIC
    "overripe": (0, 2),   # days — HEURISTIC
    "rotten":   (0, 0),   # days — HEURISTIC
}
```

### Required for a trained shelf-life model

Collect longitudinal observations of the **same banana over multiple days**:

```csv
banana_id, image_path, temperature_c, humidity_pct, days_left
B001, day0.jpg, 27, 65, 5
B001, day1.jpg, 27, 66, 4
B001, day2.jpg, 28, 68, 3
```

Splits must be by `banana_id` to prevent leakage.
The training script `src/banana_ai/ml/train_shelf_life.py` is ready and will refuse to run until real data exists.

---

## Running Tests

```powershell
$env:PYTHONPATH="src"
pytest
```

53 tests covering:
- Model construction, parameter count (~422,788), forward pass, output shape
- Device detection, CPU threading, checkpoint loading/saving
- Data transforms, image normalization
- Prediction pipeline (probabilities, class validity, confidence range)
- Shelf-life prototype estimates and labelling
- Database ORM model column validation
- FastAPI `/health` endpoint
- Grad-CAM heatmap generation

---

## Model Evaluation Reports

After evaluation, reports are saved in:

```text
reports/
├── v1/
│   ├── test_metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── probability_distribution.png
│   ├── incorrect_predictions.json
│   └── training_history.json
├── v2/
│   └── (same structure)
├── error_analysis.md
├── model_comparison.md
├── explainability/
│   ├── gradcam_overlay.png
│   └── gradcam_heatmap.png
└── FINAL_PROJECT_REPORT.md
```

---

## Reproducibility

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Image size | 224 × 224 |
| Batch size | 16 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Early stopping | patience=5 epochs |
| Loss (V1) | CrossEntropyLoss |
| Loss (V2) | CrossEntropyLoss with inverse-frequency class weights |
| Workers | 0 (CPU) |
| PyTorch threads | 4 |

---

## Limitations

1. CPU training only — no CUDA or GPU available
2. No pretrained features — CNN learns from scratch with 11K images
3. Shelf-life is a heuristic prototype, not a trained model
4. Model confidence is raw softmax — not temperature-calibrated
5. Single dataset source — generalisation to other cameras/conditions unknown
6. V2 training history JSON was not persisted from the original run (checkpoint metadata preserved)

---

## Safety Notice

This is an **educational computer-vision project**. It must not be used as the sole basis for deciding whether food is safe to eat. Visual appearance alone cannot detect every food-safety hazard.
