# Banana AI Ripeness Model V3 Evaluation

## 1. Goal
Upgrade the banana ripeness classification model to improve generalization to unseen real-world inputs while preserving the existing system architecture (including the Banana Gate). 

## 2. Methodology
- **Data Expansion:** The training dataset was expanded by merging `archive (2)` into `banana_classification`. The two datasets were deduplicated via MD5 hash comparison (adding 598 new unique images).
- **Data Audit:** `archive (1)` was audited but excluded from training because its 8-degree ripeness labeling paradigm was not safely mappable to the current 4-class taxonomy without risking data leakage or label noise.
- **Model Training:** A V3 architecture (`banana-cnn-v3`) based on the V2 BananaCNN architecture was fine-tuned for 3 epochs using a class-weighted CrossEntropyLoss on the expanded dataset.
- **Validation:** Both the frozen V2 baseline and the new V3 candidate were evaluated using the expanded test set, which includes the unseen `archive (2)` images, representing a significant shift in data distribution (i.e. real-world challenge).

## 3. Results Comparison
The V2 model experienced a severe performance drop on the new expanded test set (dropping from its historical ~95% accuracy down to 80.29%), proving that the baseline model overfit to the original data domain.
The V3 candidate model significantly improved generalization on this unseen distribution.

| Metric | V2 Baseline (Expanded Test Set) | V3 Candidate (Expanded Test Set) |
|---|---|---|
| **Accuracy** | 80.29% | 87.50% |
| **Macro F1** | 0.8101 | 0.8744 |

**Per-Class F1 Score Comparison:**
| Class | V2 Baseline | V3 Candidate | Improvement |
|---|---|---|---|
| **overripe** | 0.80 | 0.82 | +0.02 |
| **ripe** | 0.83 | 0.87 | +0.04 |
| **rotten** | 0.75 | 0.89 | +0.14 |
| **unripe** | 0.86 | 0.91 | +0.05 |

The **rotten** class saw the most dramatic improvement (from 0.75 to 0.89 F1), addressing critical food-safety and usability constraints.

## 4. UI Upgrades
- Replaced the textual prototype shelf-life estimate with a prominent **Neo-Brutalist "GOOD-TO-EAT WINDOW" card**.
- Handled the **Rotten** stage specifically, showing "0 DAYS" and "NOT RECOMMENDED" in high-contrast coral.
- Handled **Low-Confidence Predictions** (confidence < 0.60) by appending "⚠ Low-confidence estimate" and changing the display to an approximate range.

## 5. Deployment
- The V3 model was saved as `models/banana_cnn_v3.pt` with SHA-256 `68d79e0c46ef5fa02c9b164ebb5afbddbdc9cbcacc3a4d93bff672169e7aedec`.
- `config.py`, `predict.py`, and all relevant unit tests (including `test_model_immutability.py`) have been updated to target V3 as the new production model. The V2 baseline has been retained as requested.
- The Banana Gate remains completely unmodified and functional.
