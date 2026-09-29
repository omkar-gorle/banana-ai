# Banana AI Features Documentation

## Overview

Banana AI is a **Banana Quality & Shelf-Life Intelligence System** built around the production-ready `banana-cnn-v2` ripeness classifier.

---

## Features

### Feature 0 — Dedicated Banana vs. Non-Banana Gate Classifier

A lightweight, dedicated neural network gate (`BananaGateMobileNetV3`) guards the production pipeline against non-banana images:

- **Model**: `models/banana_gate_best.pt`
- **Architecture**: MobileNetV3-Small transfer learning with custom binary classifier head
- **Operating Threshold**: `0.380` (selected strictly on validation data to maximize recall and specificity)
- **Held-Out Test Accuracy**: `99.67%` (300 banana, 300 non-banana)
- **Banana Recall**: `100.00%` (0 false negatives across unripe, ripe, overripe, and rotten stages)
- **Non-Banana Rejection Rate**: `99.33%` (298/300 non-banana items blocked)
- **Inference Speed**: `< 15ms` on CPU
- **Strict Downstream Invariant**: Any rejected image immediately stops at the gate. It is NEVER passed to the frozen ripeness model (`models/banana_cnn_v2.pt`), Grad-CAM visualizer, shelf-life estimator, or PostgreSQL database.

---

### Feature 1 — Dual Image Input

The UI provides two input methods, always both available:

**Take Photo (Camera)**
- Uses `st.camera_input()` (Streamlit built-in)
- Works in modern mobile browsers with camera access permission
- Graceful fallback message if camera is unavailable — Upload Photo remains active
- No native app installation required

**Upload Photo**
- `st.file_uploader` accepting JPG, JPEG, PNG, WEBP
- Maximum 20 MB, minimum 32×32 px, maximum 16,000 px
- Validated by `src/banana_ai/services/image_validation.py` before reaching the ML pipeline

---

### Feature 2 — Prediction Result Card

After analysis, the UI displays:
- Predicted stage (UNRIPE / RIPE / OVERRIPE / ROTTEN) in a styled card
- Model confidence (labeled as "raw softmax score — not a calibrated probability")
- Model version identifier
- Input method (camera / upload)
- All four class probabilities as visual progress bars

---

### Feature 3 — Quality Assessment

A descriptive assessment is shown per stage:
- **Ripe**: "Likely ready to eat."
- **Overripe**: "Very soft and sweet — consume soon."
- **Rotten**: Warning not to treat the model as a food-safety guarantee.
- **Unripe**: "Allow to ripen."

No fake numerical quality score is used.

---

### Feature 4 — Environment-Aware Shelf-Life Estimation

User inputs:
- Temperature (°C) — sidebar slider
- Humidity (%) — sidebar slider

Output: "Estimated good-quality window: ~X–Y days"

**IMPORTANT:** This is a PROTOTYPE HEURISTIC, not a trained shelf-life model. It is not a food-safety guarantee.

---

### Feature 5 — Environmental Estimation Engine

**Module:** `src/banana_ai/services/shelf_life.py`

The `HeuristicShelfLifeEstimator` class:
- Is deterministic: same inputs → same output
- Is bounded: results always in [0, MAX_DAYS_CAP=14]
- Is transparent: all coefficients documented as prototype assumptions

Coefficients (prototype assumptions, NOT experimentally validated):
- Baseline ranges: unripe 4–7, ripe 2–4, overripe 0–2, rotten 0
- Temperature: 0.08 days per °C deviation from 22 °C
- Humidity: 0.05 days per % deviation from 60%
- Storage multipliers: room ×1.0, cool ×1.3, refrigerator ×1.6

---

### Feature 6 — What-If Environment Simulator

Section in the Analyze page that lets users adjust hypothetical temperature, humidity, and storage condition. Uses the same `estimate_shelf_life()` function — no formula duplication. Clearly labeled "Prototype simulation."

---

### Feature 7 — Storage Condition

Optional dropdown: Room / Cool Storage / Refrigerator / Other.
Used as a multiplier in the heuristic. Assumptions are documented.

---

### Feature 8 — Grad-CAM Explanation

Shows:
- Original image
- Grad-CAM heatmap overlay
- Caption: "The highlighted regions show image areas that contributed to the model's prediction."

If Grad-CAM fails: prediction continues, friendly message shown, error logged.

---

### Feature 9 — Prediction History

History dashboard (requires PostgreSQL):
- Total scans
- Stage distribution
- Filterable table: stage, confidence, temperature, humidity, estimated window, model version

---

### Feature 10 — Analytics Dashboard

Charts (require ≥3 records):
- Ripeness distribution
- Confidence distribution
- Scans over time
- Temperature vs stage

If insufficient data: "Not enough historical data yet."

---

### Feature 11 — Environmental History

Observations store: temperature, humidity, timestamp, stage, confidence, estimated days, model version. Clearly noted: correlation ≠ causation.

---

### Feature 12 — Human Feedback

After prediction:
- 👍 Yes (correct) / 👎 No (incorrect)
- Optional stage correction if incorrect
- Stored in `prediction_feedback` table
- **Does NOT retrain the model — ever.**

---

### Feature 13 — Batch Analysis

Upload 5–20 images. Each analyzed by V2. Summary table + CSV export. Results NOT saved automatically — user must click "Save Batch Results."

---

### Feature 14 — Food-Waste Insights

Dashboard showing real counts of ripe/overripe/rotten from stored data. No fabricated environmental impact numbers. Requires ≥5 records.

---

### Feature 15 — Eat-First Priority

Priority shown per prediction:
- Overripe → High priority
- Ripe → Medium priority
- Unripe → Low priority
- Rotten → Inspect/discard

This is a prioritization aid, NOT a food-safety determination.

---

### Feature 16 — Report Generation

Downloads available after each analysis:
- **HTML Report** — human-readable, can be printed to PDF from browser
- **CSV Report** — machine-readable, includes all fields + disclaimer

Both always include the mandatory disclaimer about prototype heuristic status.

---

### Feature 17 — Mobile-First UI

- Tabs for camera/upload
- Large buttons
- Expandable sections for advanced info
- Simple first screen
- No excessively wide tables

---

### Feature 18 — Camera/Upload Security

All images validated by `src/banana_ai/services/image_validation.py`:
- Extension check
- PIL format check
- Empty file rejection
- Size limits (max 20 MB)
- Dimension limits (min 32 px, max 16,000 px)
- Safe UUID-based temp filenames (no path traversal)
- Images converted to RGB — no executable code can reach the ML pipeline

---

### Feature 19 — Database Design

New columns (additive):
- `observations.storage_condition` VARCHAR(64)
- `observations.input_method` VARCHAR(32)
- `predictions.estimated_min_days` INTEGER
- `predictions.estimated_max_days` INTEGER
- `predictions.shelf_life_method` VARCHAR(64)

New table: `prediction_feedback`

Migration: `python -m banana_ai.db.migrate` (idempotent, never drops)

---

### Feature 20 — API Support

`POST /predict` now accepts optional query params:
- `temperature_c` (float)
- `humidity_pct` (float)
- `storage_condition` (str)

Response includes: `estimated_min_days`, `estimated_max_days`, `shelf_life_method`, `shelf_life_warning`.

---

### Feature 21 — About the AI

Dedicated page in Streamlit showing model architecture, metrics, per-class F1, honest disclaimers, and future shelf-life model plan.

---

### Feature 22 — Future ML Shelf-Life Architecture

`ShelfLifeEstimator` abstract base class allows future `MLShelfLifeEstimator` to replace `HeuristicShelfLifeEstimator`. Longitudinal dataset schema documented.

---

### Feature 23 — Testing

New test file: `tests/test_features_v2.py`
- 70+ new tests covering all new features
- All 55 original tests continue to pass

---

### Feature 24 — Performance

- Model loaded once via `@st.cache_resource`
- Grad-CAM only runs when requested
- Batch reuses cached model
- Image size stays at 224×224

---

### Feature 25 — Documentation

- `README.md` — updated
- `ARCHITECTURE.md` — updated
- `docs/FEATURES.md` — this file
- `docs/SHELF_LIFE.md` — shelf-life methodology
- `reports/feature_upgrade_report.md` — final report

---

### Feature 26 — No Fake Science

All heuristic estimates labeled:
- "Estimated"
- "Prototype"
- "Heuristic"
- "Not a food-safety guarantee"

No scientifically validated claims made for the shelf-life estimator.

---

### Feature 27 — Regression Protection

V2 checkpoint SHA-256 verified before and after all changes:
`cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d`

---

### Feature 28 — Git Safety

No automatic pushes. `.env`, secrets, model weights, dataset, `.venv` excluded from commits via `.gitignore`.
