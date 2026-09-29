# Banana Gate Real-Image Acceptance Test Report

Date: 2026-09-30
Model: BananaGateMobileNetV3 (`models/banana_gate_best.pt`)
Application Gate Threshold: `0.500`
Overall Status: **PASSED**

---

## 1. Banana Acceptance Results (Target: All Accepted)

| Image File | Description | Banana Probability | Threshold | Result | Status |
|---|---|---|---|---|---|
| `banana_overripe.jpg` | Overripe banana | 100.00% | 0.50 | Accepted as Banana | **PASS** |
| `banana_ripe.jpg` | Ripe banana | 100.00% | 0.50 | Accepted as Banana | **PASS** |
| `banana_rotten.jpg` | Rotten banana | 100.00% | 0.50 | Accepted as Banana | **PASS** |
| `banana_unripe.jpg` | Unripe banana | 100.00% | 0.50 | Accepted as Banana | **PASS** |

---

## 2. Non-Banana Rejection Results (Target: All Rejected)

| Image File | Category | Banana Probability | Threshold | Result | Status |
|---|---|---|---|---|---|
| `nonbanana_apple.jpg` | Fresh Apple | 0.00% | 0.50 | Rejected as Non-Banana | **PASS** |
| `nonbanana_orange.jpg` | Fresh Orange | 0.00% | 0.50 | Rejected as Non-Banana | **PASS** |
| `nonbanana_person_hand.jpg` | Person / Hand | 0.00% | 0.50 | Rejected as Non-Banana | **PASS** |
| `nonbanana_household_cup.jpg` | Household Object / Plate | 0.01% | 0.50 | Rejected as Non-Banana | **PASS** |
| `nonbanana_background_scene.jpg` | Architecture / Scene | 0.00% | 0.50 | Rejected as Non-Banana | **PASS** |

---

## 3. Invariant Verification

1. All 4 banana stages (unripe, ripe, overripe, rotten) were confirmed as bananas.
2. All 5 realistic non-banana categories were successfully rejected before reaching the ripeness pipeline.
3. Zero rejected images reached `banana_cnn_v2.pt`, Grad-CAM, shelf-life estimation, or the database.
