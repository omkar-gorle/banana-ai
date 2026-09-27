# Banana AI — Architecture

## System Overview

```text
                         USER
                           |
              +------------+------------+
              |                         |
      Streamlit UI                 FastAPI
      (app.py)                    (/health, /predict)
              |                         |
              +----------+--------------+
                         |
                 Prediction Service
                  (ml/predict.py)
                         |
               +----+----+----+
               |              |
         Preprocessing      Grad-CAM
         (224×224, norm)   (ml/explainability.py)
               |
           BananaCNN
           (ml/model.py)
           422,788 params
           Trained from scratch
               |
        Softmax Probabilities
               |
       +---------+----------+
       |                    |
 PostgreSQL            Shelf-life
 (observations +       prototype
  predictions)        (heuristic only)
```

## Component Map

| Component | File | Purpose |
|-----------|------|---------|
| CNN model | `src/banana_ai/ml/model.py` | 4-class ripeness classifier |
| Data loading | `src/banana_ai/ml/data.py` | Train/val/test loaders |
| V1 training | `src/banana_ai/ml/train.py` | Baseline training |
| V2 training | `src/banana_ai/ml/train_v2.py` | Class-weight experiment |
| Evaluation | `src/banana_ai/ml/evaluate.py` | Basic test evaluation |
| Extended eval | `src/banana_ai/ml/evaluate_extended.py` | Versioned reports |
| Prediction | `src/banana_ai/ml/predict.py` | Single-image inference |
| Grad-CAM | `src/banana_ai/ml/explainability.py` | Region visualisation |
| Analysis | `src/banana_ai/ml/analysis.py` | Error + comparison |
| Dataset report | `src/banana_ai/ml/dataset_report.py` | Validation counts |
| Device | `src/banana_ai/ml/device.py` | CPU/CUDA/XPU detection |
| Shelf-life | `src/banana_ai/services/shelf_life.py` | Prototype heuristic |
| DB models | `src/banana_ai/db/models.py` | SQLAlchemy ORM |
| DB session | `src/banana_ai/db/session.py` | Connection pool |
| DB CRUD | `src/banana_ai/db/crud.py` | Save/query predictions |
| DB init | `src/banana_ai/db/init_db.py` | Create tables |
| FastAPI | `src/banana_ai/api/main.py` | REST API |
| Streamlit | `src/banana_ai/app.py` | Web UI |
| Config | `src/banana_ai/config.py` | Pydantic settings |

## BananaCNN Architecture

```text
Input: (N, 3, 224, 224)

Block 1:  Conv2d(3, 32, 3)  → BN → ReLU → MaxPool(2)   → (N, 32, 112, 112)
Block 2:  Conv2d(32, 64, 3) → BN → ReLU → MaxPool(2)   → (N, 64,  56,  56)
Block 3:  Conv2d(64, 128, 3)→ BN → ReLU → MaxPool(2)   → (N, 128, 28,  28)
Block 4:  Conv2d(128, 256, 3)→BN → ReLU → AdaptiveAvgPool(1,1) → (N, 256, 1, 1)

Flatten                                                  → (N, 256)
Dropout(0.35)
Linear(256, 128) → ReLU
Dropout(0.25)
Linear(128, 4)                                           → (N, 4) logits

Softmax (at inference)                                   → (N, 4) probabilities

Total parameters: ~422,788
```

## Prediction Pipeline

```text
image file
    |
PIL.Image.open → .convert("RGB")
    |
Resize(224, 224)
    |
ToTensor → Normalize(ImageNet stats)
    |
BananaCNN.forward()
    |
torch.softmax()
    |
argmax → predicted class index
    |
class_names[index] → "ripe" | "overripe" | "rotten" | "unripe"
    |
prototype_estimate() → "2-4 days" (HEURISTIC ONLY)
```

## Grad-CAM Pipeline

```text
input tensor (1, 3, 224, 224)
    |
Forward hook on final Conv2d layer (Block 4)
    |
model.forward() → save activation maps
    |
target_class_score.backward() → save gradients
    |
weights = mean(gradients over spatial dims)
    |
cam = relu(sum(weights × activations))
    |
normalize to [0, 1]
    |
resize to (224, 224)
    |
applyColorMap(JET) → heatmap
    |
blend(heatmap, original, alpha=0.5) → overlay
```

## Database Schema

```sql
TABLE observations (
    id           SERIAL PRIMARY KEY,
    image_path   TEXT NOT NULL,
    temperature_c FLOAT,
    humidity_pct  FLOAT,
    observed_at  TIMESTAMPTZ DEFAULT now()
);

TABLE predictions (
    id                SERIAL PRIMARY KEY,
    observation_id    INT NOT NULL REFERENCES observations(id),
    predicted_stage   VARCHAR(32) NOT NULL,
    confidence        FLOAT NOT NULL,
    estimated_days_left VARCHAR(32) NOT NULL,
    model_version     VARCHAR(64) NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT now()
);
```

**Model weights are stored on disk, never in PostgreSQL.**

## Experiment Versions

| Version | Description | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Checkpoint |
|---------|-------------|-------------|---------------|---------------|------------|
| V1 | Baseline: equal class weights | 0.9521 | 93.06% | 0.9310 | `models/banana_cnn_best.pt` |
| **V2** | Class-weight balancing | **0.9608** | **95.20%** | **0.9536** | `models/banana_cnn_v2.pt` |

## Model Selection

Production model = V2 (`models/banana_cnn_v2.pt`), selected using the
highest **validation** macro-F1 (not test macro-F1). V1 remains available
only for historical training/evaluation.
The test set is evaluated exactly once, after model selection.

**Result**: V2 selected — val macro-F1 0.9608 vs V1's 0.9521.

The Streamlit analysis path is side-effect free. A prediction is inserted
into PostgreSQL only after the explicit **Save prediction** action, with a
rerun guard preventing duplicate saves. Softmax outputs are raw and
uncalibrated; shelf-life estimates remain prototype heuristics.

## Why PostgreSQL?

- Stores application data: observations, predictions, confidence, timestamps
- NOT used for neural network weights
- Enables prediction history and future longitudinal shelf-life tracking

## Shelf-Life Status

The current system provides a **PROTOTYPE HEURISTIC** only:

```python
STAGE_RANGES = {
    "unripe":   (3, 7),   # days
    "ripe":     (2, 4),
    "overripe": (0, 2),
    "rotten":   (0, 0),
}
```

A **trained** shelf-life model requires longitudinal data:
```csv
banana_id, image_path, temperature_c, humidity_pct, days_left
B001, day0.jpg, 27, 65, 5
B001, day1.jpg, 27, 66, 4
...
```

Splits must be by `banana_id` to prevent data leakage.

## Future Upgrades

- Alembic database migrations
- MLflow experiment tracking
- S3/MinIO image storage
- Calibrated confidence scores (temperature scaling)
- Dockerized API + UI
- CI/CD pipeline
- Real longitudinal shelf-life dataset
- Transfer learning comparison
