# Banana AI — Banana Quality & Shelf-Life Intelligence System

A complete ML pipeline that classifies banana ripeness from images using a **custom CNN trained from scratch** (no pretrained weights), enhanced with environment-aware shelf-life estimation, batch analysis, feedback collection, and a mobile-friendly multi-page Streamlit UI.

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

## Banana Gate Binary Classifier (Production Input Guard)

To eliminate false positive ripeness predictions on non-banana images, the pipeline utilizes a dedicated binary gate model:

- **Checkpoint**: `models/banana_gate_best.pt`
- **Architecture**: `BananaGateMobileNetV3` (MobileNetV3-Small backbone with custom classifier head)
- **Classes**: `non_banana` (0), `banana` (1)
- **Negative Training Sources (2,700 images)**:
  - Fruit-Images-Dataset (`Horea94/Fruit-Images-Dataset`, MIT License): 1,300 images across 10 fruit/vegetable classes
  - OpenCV Official Samples (`opencv/opencv`, Apache 2.0 License): 90 images of people/hands, tableware, architecture, objects
  - CIFAR-10 Dataset (Open Academic Research): 1,310 images of animals and vehicles
- **Validation-selected threshold**: `0.380` (tuned strictly on held-out validation set)
- **Test Set Accuracy**: **99.67%** (600 held-out images: 300 banana, 300 non-banana)
- **Banana Recall**: **100.00%** (300 / 300 bananas accepted across all 4 stages)
- **Non-Banana Rejection**: **99.33%** (298 / 300 non-bananas blocked)
- **Inference Latency**: **< 15ms** on CPU

```text
Camera / Upload / Batch
       ↓
Image File Validation (MIME / size / dimensions)
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

**Frozen Ripeness Model Invariant**: The production ripeness model (`models/banana_cnn_v2.pt`) remains completely frozen and immutable (SHA-256: `cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d`). Non-banana images are stopped dead at the gate and never reach `banana_cnn_v2.pt`.

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

Example response (with optional `temperature_c=27&humidity_pct=65` query params):
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
  "estimated_days_left": "1-3 days",
  "estimated_min_days": 1,
  "estimated_max_days": 3,
  "shelf_life_method": "heuristic",
  "shelf_life_warning": "estimated_days_left is a PROTOTYPE heuristic estimate. It is NOT a trained shelf-life model and NOT a food-safety guarantee.",
  "prediction_id": 42
}
```

---

## Streamlit UI

```powershell
$env:PYTHONPATH="src"
streamlit run src/banana_ai/app.py
```

### Free deployment on Streamlit Community Cloud

This repository is ready to deploy at no cost on
[Streamlit Community Cloud](https://share.streamlit.io/):

1. Push the repository to GitHub.
2. Sign in to Streamlit Community Cloud with the GitHub account that owns
   `omkar-gorle/banana-ai`.
3. Select **New app**, choose `omkar-gorle/banana-ai`, and set the branch to
   `main`.
4. Set **Main file path** to `src/banana_ai/app.py`.
5. Deploy. Streamlit installs `requirements.txt` automatically.

The V2 checkpoint is included in `models/banana_cnn_v2.pt`, so the deployed
app can perform inference without downloading a private artifact. PostgreSQL
features are optional; image analysis and reports work without a database.
If database persistence is needed, add a `DATABASE_URL` secret in the
Streamlit app settings. Never commit `.env` or database credentials.

Multi-page app (sidebar navigation):

**Analyze page:**
1. Camera capture (mobile browser) OR file upload — both always available
2. Image security validation before ML pipeline
3. Run V2 CNN classification
4. Styled prediction card with stage, confidence, model version
5. Quality assessment per stage
6. Eat-first priority label
7. All 4 class probabilities as progress bars
8. Environment-aware shelf-life estimate (prototype heuristic)
9. What-if environment simulator
10. Optional Grad-CAM explanation (expander)
11. HTML + CSV report download
12. Explicit save to PostgreSQL (duplicate-save protected)
13. Human feedback (correct/incorrect + optional correction)

**Other pages:** Batch Analysis, Prediction History, Analytics, Waste Insights, About AI

See `docs/FEATURES.md` for full feature documentation.

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

The shelf-life estimator (`src/banana_ai/services/shelf_life.py`) uses a bounded linear formula:
- **Baseline stage ranges**: unripe 4-7, ripe 2-4, overripe 0-2, rotten 0 days
- **Temperature adjustment**: 0.08 days/°C deviation from 22°C reference
- **Humidity adjustment**: 0.05 days/% deviation from 60% reference
- **Storage multipliers**: room ×1.0, cool ×1.3, refrigerator ×1.6
- **Output clamped**: [0, 14] days

All coefficients are **prototype assumptions, not experimentally validated constants**.

See `docs/SHELF_LIFE.md` for full methodology.

### Architecture for future trained model

`ShelfLifeEstimator` (abstract) → `HeuristicShelfLifeEstimator` (current) → `MLShelfLifeEstimator` (future)

Required longitudinal dataset (split by `banana_id` to prevent leakage):
```csv
banana_id, image_path, temperature_c, humidity_pct, storage_condition, days_since_start, days_left, observed_stage
```

---

## Running Tests

```powershell
$env:PYTHONPATH="src"
pytest
```

**126 tests** (55 original + 71 new) covering:
- Model construction, parameter count (~422,788), forward pass, output shape
- Device detection, CPU threading, checkpoint loading/saving
- Data transforms, image normalization
- Prediction pipeline (probabilities, class validity, confidence range)
- Enhanced shelf-life service (environment-aware, all stages, boundaries)
- Temperature and humidity boundary clamping
- What-if simulation determinism
- Image validation and security
- Batch analysis helpers
- HTML/CSV report generation
- Extended database model columns
- PredictionFeedback table
- FastAPI backward compatibility
- Model version consistency

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
