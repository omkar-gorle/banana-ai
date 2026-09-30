# Phase 1: Current-State Audit

## 1. Repository Structure & Configuration
- **Project Root**: `D:\ff\banana\banana_ai_from_scratch`
- **Configuration**: `src/banana_ai/config.py` points to V3 (`models/banana_cnn_v3.pt`) and Banana Gate (`models/banana_gate_best.pt` with threshold `0.50`). The `.env` file matches.
- **Pipeline**: The prediction pipeline (`predict.py`) and UI (`app.py`) route images through the Gate first, then to the Ripeness model, and then to the prototype Good-to-Eat estimator based on the ripeness string.
- **Grad-CAM**: Enabled and triggers on the accepted banana image.

## 2. Models & Immutability
- **V2 Checkpoint (`banana_cnn_v2.pt`)**: Hash is `cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d`. Confirmed frozen and immutable.
- **V3 Checkpoint (`banana_cnn_v3.pt`)**: Hash is `68d79e0c46ef5fa02c9b164ebb5afbddbdc9cbcacc3a4d93bff672169e7aedec`. Correctly deployed.
- **Banana Gate (`banana_gate_best.pt`)**: Immutable. Application threshold is `0.50`.

## 3. Data & Splits
- **Dataset (`banana_classification_v3`)**: 
  - Contains `train`, `valid`, `test` splits.
  - Built by deduplicating and merging `archive (2)` into `banana_classification`.
  - No duplicate leakage was found previously (MD5 deduplication applied).
  - Train set is entirely isolated from validation/test.
- **Augmentation**: Standard `torchvision.transforms` (resize to 224, normalize with ImageNet stats). No extreme test-time augmentation or advanced photometric distortion is currently running in production.
- **Class Mapping**: Strict 4-class taxonomy (`unripe`, `ripe`, `overripe`, `rotten`). No arbitrary rules or overrides found.

## 4. Confidence & Shelf-Life
- **Confidence**: Calculated via straightforward softmax over logits (`probabilities = torch.softmax(model(tensor), dim=1)[0]`). Currently uncalibrated.
- **Good-to-Eat Window**: Prototype heuristic mapping (Unripe -> 5-7 days, Ripe -> 2-4 days, Overripe -> 1-2 days, Rotten -> 0 days). No longitudinal data backs this scientifically.
- **Post-processing**: No artificial accuracy inflation or arbitrary overrides are forcing predictions.

## Conclusion
The repository is fundamentally healthy. The codebase is well-structured, the checkpoints are secured via SHA-256 constraints, and test data isolation is preserved. The gap between original and real-world performance is purely a model generalization limitation, not a code bug.
