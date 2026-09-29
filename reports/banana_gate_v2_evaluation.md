# Banana Gate v2 Hard-Negative Evaluation

Date: 2026-09-30
Checkpoint: `models/banana_gate_v2.pt`
Production alias: `models/banana_gate_best.pt`
Checkpoint SHA-256: `e5315286e7ff279cc5da04660a129f465d55853dd426e2466409aabc9ca6cd7b`
Previous checkpoint SHA-256: `a1416daae0ae22080af79436ae8c97b59ebcd301af28fddb8ce19dff5929789d`
Frozen V2 SHA-256: `cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d`

## Root Cause Evidence

The exact production fist image is not present in the repository, so the reported
91% production failure was not claimed as reproduced. The original gate was
evaluated against a licensed HaGRIDv2 `no_gesture` hard-negative source. At the
previous production threshold of `0.50`, it accepted 240 of 2,164 hand-negative
images (11.09% false-positive rate). At the checkpoint threshold of `0.38`, it
accepted 341 of 2,164 (15.76%). This demonstrates that threshold calibration
alone could not solve the hand/fist failure.

## Dataset and Provenance

- Positive data: existing Banana Classification Dataset, four ripeness stages.
- Existing negative data: Fruits-360 (MIT), OpenCV samples (Apache 2.0), and
  CIFAR-10 (open academic research use).
- Hard negatives: HaGRIDv2 `no_gesture`, CC BY 4.0.
- Source: <https://github.com/hukenovs/hagrid>
- Archive: <https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagrid_v2_zip/no_gesture.zip>
- Hard-negative images evaluated: 2,164.
- Deterministic filename-hash split: 1,531 train, 309 validation, 324 test.
- The original v1 checkpoint remains available as `models/banana_gate_v1.pt`.

The HaGRID archive did not expose subject identifiers in the extracted sample,
so the hard-negative split is by deterministic filename hash, not by person.
This limitation is recorded rather than presented as a subject-disjoint result.

## Selected Threshold

| Item | Value |
|---|---:|
| Previous application threshold | 0.500 |
| Previous checkpoint operating point | 0.380 |
| Candidate checkpoint validation operating point | 0.350 |
| Production application threshold | 0.500 |

The v2 checkpoint threshold was selected on the training script's validation
split using the existing high-recall/high-specificity/F1 policy. The
application remains at 0.500 to preserve the established BANANA/UNCERTAIN
boundary; the v2 model also rejected every held-out HaGRIDv2 hard negative at
that production threshold. Threshold calibration was not selected from the
original fist report.

## Held-Out Metrics

The combined held-out set contains 300 banana positives and 624 non-banana
images (300 existing negatives plus 324 HaGRIDv2 hard negatives).

| Metric | Result |
|---|---:|
| Accuracy | 99.89% |
| Precision | 99.67% |
| Banana recall | 100.00% |
| Specificity | 99.84% |
| F1 | 0.9983 |
| ROC-AUC | 1.0000 |
| False-positive rate | 0.16% |
| False-negative rate | 0.00% |
| Confusion matrix | TN=623, FP=1, FN=0, TP=300 |

### Hard-Negative Category Results

| Category | Images | False positives | False-positive rate |
|---|---:|---:|---:|
| HaGRIDv2 `no_gesture` hands | 324 held-out test | 0 | 0.00% |

The separate HaGRIDv2 validation split also had 0/309 false positives. The
previous v1 checkpoint had 341/2,164 false positives on the full hard-negative
source at threshold 0.38.

## Acceptance Images

All four banana stages were accepted by v2, and the representative local hand,
apple, orange, cup, and background images were rejected. The exact production
fist remains unavailable.

## Pipeline Invariants

The Banana Gate remains upstream of the frozen ripeness model:

```text
NOT_BANANA -> stop
BANANA -> V2 ripeness -> Grad-CAM / shelf-life / persistence
```

Rejected images do not reach V2, Grad-CAM, shelf-life estimation, or prediction
persistence. The existing regression tests cover these blocking invariants.

## Model Changes

- V2 ripeness model modified: **NO**
- Banana Gate modified: **YES**, new `banana_gate_v2.pt`
- Previous gate retained: `banana_gate_v1.pt`
- Production alias: `banana_gate_best.pt` points to v2
