# Banana AI — Error Analysis

**Based on measured V1 test set results. No values fabricated.**

## Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.9306 |
| Macro F1 | 0.9310 |
| Macro Precision | 0.9325 |
| Macro Recall | 0.9298 |
| Weighted F1 | 0.9305 |

## Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| overripe | 0.9074 | 0.8673 | 0.8869 | 113 |
| ripe | 0.9057 | 0.9351 | 0.9201 | 154 |
| rotten | 0.9351 | 0.9351 | 0.9351 | 185 |
| unripe | 0.9818 | 0.9818 | 0.9818 | 110 |

## Confusion Matrix (Raw Counts)

Rows = actual class, Columns = predicted class.

| Actual \ Predicted | overripe | ripe | rotten | unripe |
|---------------------|----------|------|--------|--------|
| **overripe** | 98 | 14 | 1 | 0 |
| **ripe** | 1 | 144 | 9 | 0 |
| **rotten** | 9 | 1 | 173 | 2 |
| **unripe** | 0 | 0 | 2 | 108 |

## Confusion Analysis

### ripe vs overripe confusion

- Overripe predicted as ripe: **14** / 113 (12.4%)
- Ripe predicted as overripe: **1** / 154 (0.6%)

These two classes are visually adjacent on the ripeness spectrum. Yellow-brown transition makes them hard to separate without contextual cues.

### overripe vs rotten confusion

- Overripe predicted as rotten: **1** / 113 (0.9%)
- Rotten predicted as overripe: **9** / 185 (4.9%)

### unripe vs ripe confusion

- Unripe predicted as ripe: **0** / 110 (0.0%)
- Ripe predicted as unripe: **0** / 154 (0.0%)

### rotten false negatives

- Rotten correctly identified: **173** / 185 (93.5%)
- Rotten misclassified as other: **12** / 185 (6.5%)

### Low-confidence incorrect predictions

Incorrect predictions with confidence < 60%: **5**

| True class | Predicted | Confidence |
|------------|-----------|------------|
| overripe | ripe | 0.521 |
| overripe | ripe | 0.510 |
| overripe | rotten | 0.595 |
| ripe | rotten | 0.513 |
| ripe | overripe | 0.599 |

## Key Observations

1. **Adjacent-class confusion**: ripe/overripe and overripe/rotten are the most frequent error patterns, consistent with their visual similarity.

2. **Class imbalance effect**: rotten has the highest support (~33%) in train and may be over-represented in the decision boundary. The V2 class-weight experiment addresses this.

3. **No test set was modified** based on these results.

4. **Grad-CAM** should be used to inspect whether the model attends to banana skin texture/colour regions or spurious background features.

## Limitations

- Dataset images are 416×416 photos under similar controlled backgrounds. Real-world images may have different lighting, orientation, and backgrounds.
- The model was trained from scratch with no pretrained features. A larger dataset or transfer learning could improve boundary precision.
- Shelf-life prediction requires longitudinal data not present in this dataset.

---
_Analysis generated from measured V1 test results. Values are not fabricated._