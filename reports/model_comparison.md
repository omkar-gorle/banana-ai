# V1 vs V2 Model Comparison

## Overall Metrics

| Metric | V1 (baseline) | V2 (class weights) | Delta |
|--------|--------------|---------------------|-------|
| Accuracy | 0.9306 | 0.9520 | +0.0214 |
| Macro F1 | 0.9310 | 0.9536 | +0.0226 |
| Macro Precision | 0.9325 | 0.9536 | +0.0211 |
| Macro Recall | 0.9298 | 0.9538 | +0.0240 |
| Weighted F1 | 0.9305 | 0.9519 | +0.0215 |

## Per-Class F1 Comparison

| Class | V1 F1 | V2 F1 | Delta |
|-------|-------|-------|-------|
| overripe | 0.8869 | 0.9375 | +0.0506 |
| ripe | 0.9201 | 0.9423 | +0.0222 |
| rotten | 0.9351 | 0.9482 | +0.0131 |
| unripe | 0.9818 | 0.9864 | +0.0046 |

## Model Selection

**Selection criterion**: best validation macro-F1 (validation set, not test set).

| Model | Best Val Macro-F1 |
|-------|------------------|
| V1 | 0.9521 |
| V2 | 0.9608 |

**Selected model: V2** (higher validation macro-F1)

Note: This selection was made using the validation set only, not by inspecting test-set metrics.

## Important Notes

- V1 uses equal class weights (vanilla CrossEntropyLoss).
- V2 uses inverse-frequency class weights to reduce majority-class bias.
- Both models use identical architecture, dataset, and evaluation protocol.
- No universal 'winner' is declared. V2 may improve minority-class recall at the cost of overall accuracy. Examine per-class F1 differences.

---
_Values from measured test evaluations. Not fabricated._