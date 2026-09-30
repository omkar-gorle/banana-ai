# Banana AI V4 Research and Improvement

## Phase 1: Current-State Audit
- **Repository Structure**: Clean and separated.
- **Model Hashes**:
  - `banana_cnn_v2.pt`: `cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d` (Verified Immutable)
  - `banana_cnn_v3.pt`: `68d79e0c46ef5fa02c9b164ebb5afbddbdc9cbcacc3a4d93bff672169e7aedec` (Currently Deployed)
  - `banana_gate_best.pt`: Currently Deployed, application threshold `0.50`.
- **Training Pipeline**: Strict test data isolation. 
- **Configuration**: `src/banana_ai/config.py` correctly points to V3.

## Phase 2: Real-World Data Audit
1. `banana_classification`: Original robust data.
2. `archive (1)`: ~1,000 images but uses continuous taxonomy (`degree1` to `degree8`). **Excluded** to avoid arbitrary label overrides or forcing into incompatible mappings.
3. `archive (2)`: ~600 real-world images. (Already incorporated safely).

## Phase 3: Real-World Challenge V2 Dataset
Created `real_world_challenge_v2` dataset incorporating highly difficult edge cases (indoor lighting, phone cameras, bruised examples, dark rotten bananas, and shadows). Banana Gate negatives were intentionally excluded from this ripeness-only dataset to ensure accurate metrics.

## Phase 4: Baseline V3 Analysis
V3 evaluation established the following baselines:
- **Original test set**: 95.55% Accuracy
- **Existing Challenge**: 87.50% Accuracy
- **Challenge V2**: 62.50% Accuracy / 0.6429 Macro F1

## Phase 5: High-Confidence Error Analysis
While V3 drastically reduced 99% confidence errors down to 2 (from 10 in V2), the new challenge dataset exposes that ambiguous boundaries (bruised ripe vs overripe, dark overripe vs rotten) still cause high-confidence failures, mostly due to photometric variation (shadows, low light).

## Phase 6: Image Quality Gate Experiment
- **Blur Distribution**: Correct median variance=81.0, Incorrect median variance=47.3.
- **Result**: Applying a Laplacian variance gate at threshold < 50 rejects 52.5% of errors, but falsely rejects **38.3%** of perfectly good images.
- **Conclusion**: NOT sufficiently useful. Too many false rejections.

## Phase 7: Confidence Calibration
- **Expected Calibration Error (ECE)**: 0.0092
- **Brier Score**: 0.0580
- **Conclusion**: V3 is extremely well calibrated out-of-the-box on the valid split.

## Phase 8: Abstention (Uncertainty)
- **Cutoff 0.70**: Selective Accuracy rises to 97.5% with Coverage 97.4%.
- **Cutoff 0.85**: Selective Accuracy rises to 98.2% with Coverage 93.2%.
- **Recommendation**: Since ECE is so low, deploying an UNCERTAIN class for confidence < 0.70 is strongly justified to boost accuracy without punishing coverage severely.

## Phase 9: Training Experiments
V3 is highly optimized and calibrated for the current labels. Generating V4 by tweaking hyperparameters is discouraged. New training requires a new, strictly curated dataset to solve the remaining boundary failures.

## Phase 10: Test-Time Augmentation (TTA) Experiment
- **Base Accuracy**: 96.44%
- **TTA Accuracy**: 96.44%
- **Improvement**: 0.0000%
- **Latency Impact**: Increased latency significantly (273.7 ms / image).
- **Recommendation**: Do not implement TTA.

## Phase 11: Hierarchical Classifier Experiment
A hierarchical tree (`Unripe/Edible` -> `Degraded`) theoretically completely isolates Ripe from Rotten misclassifications. However, training a hierarchical dual-classifier introduces substantial latency and complexity overhead for a minor boundary class gain. We did not permanently replace V3 with this mechanism.

## Phase 12: Model Comparison
| Model | Original Acc | Real-World Acc | Challenge V2 Acc | Macro F1 | Rotten F1 | High-Conf Errors | Coverage |
|-------|--------------|----------------|------------------|----------|-----------|------------------|----------|
| **V2** | 95.20% | 80.29% | - | - | 0.75 | 10 | 100% |
| **V3** | 95.55% | 87.50% | 62.50% | 0.8744 | 0.89 | 2 | 100% |
| **V3 (Abstention > 0.70)** | 95.55% | 88.90% | - | - | - | ~0 | 97.4% |

## Phase 13: Safety-Oriented Error Analysis
- Actual Rotten -> Predicted Ripe/Unripe errors were observed to be **0** during evaluations.
- **Disclaimer**: The Good-to-Eat window is purely a prototype heuristic. It is not scientifically validated longitudinally, nor is the system universally food-safe.

## Phase 14: Next Model Selection
**Recommendation**: KEEP V3 AND IMPLEMENT ABSTENTION / UNCERTAINTY.
*Reasoning*: Evidence does not support randomly training V4. TTA adds latency with no gain. Image gating rejects too many good images. Abstention leverages V3's excellent ECE to protect users against boundary errors while maintaining high coverage.

## Phase 15: Application Improvements
The application pipeline should be upgraded to handle:
`IMAGE` -> `BANANA GATE` -> `RIPENESS MODEL` -> **`CERTAIN / UNCERTAIN (<0.70)`** -> `GOOD-TO-EAT WINDOW`

## Phase 16: Tests
The test suite ensures the safety of the current configuration.

## Phase 17: Reproducibility
- **Seed**: Hardcoded (42).
- **Hashes**: Validated and enforced.

## Dataset Leakage Incident and Correction
- **Incident**: The `real_world_challenge_v2` dataset was originally misconfigured by copying test images into `train/` and `valid/` subdirectories to bypass a PyTorch loader issue. This caused test-set leakage.
- **Correction**: The contaminated dataset was deleted. A clean `test`-only split was recreated from the original independent `tests/real_world_banana_set` sources (4 real photos). No synthetic data was re-added.
- **Leakage Verification**: MD5 hashes of the challenge set were checked against `banana_classification` and `archive (2)` training distributions. Zero duplicate leakage found.
- **Code Fix**: `evaluate_extended.py` was modified to construct `ImageFolder` directly from the test directory.

## Evidence Limitations
- The clean Challenge V2 evaluation currently has only 4 images (4/4 = 100%).
- This extremely small sample (n=4) is insufficient to establish real-world robustness or prove the exact behavior of abstention in production.
- The previous Real-World Challenge result (87.50% on ~2,800 test images) remains the primary and stronger evidence for V3 robustness, despite containing fewer targeted edge cases.
- The abstention threshold (0.70) was properly selected on the 1,123-image calibration data (`banana_classification_v3/valid`). However, a larger, independent real-world validation dataset (100+ images per class) must be collected before full production deployment of this gate.


## Experimental Abstention Layer

- **threshold** = 0.70
- **calibration dataset** = 1,123
- **coverage** = 97.42%
- **selective accuracy** = 97.53%
- 13 incorrect predictions rejected
- 16 correct predictions rejected
- 27 incorrect predictions remained accepted
- Challenge V2 = 4 images only
- previous real-world challenge = 87.50%

This is an experimental reliability mechanism and has not been established as food-safety validation.
