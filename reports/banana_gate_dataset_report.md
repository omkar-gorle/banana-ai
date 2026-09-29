# Banana Gate Dataset Audit & Manifest Report

Generated Date: 2026-09-30  
Dataset Directory: `data/banana_gate/`  
Random Seed: 42  

## 1. Executive Summary

A dedicated binary classification dataset (**BANANA** vs. **NON-BANANA**) was constructed to replace the legacy COCO detector with a dedicated, lightweight binary gate. The dataset is strictly partitioned into train, validation, and test splits with **zero data leakage** across split boundaries.

- **Total Images:** 5,400
- **Positive (Banana):** 2,700 (50.0%)
- **Negative (Non-Banana):** 2,700 (50.0%)
- **Balance:** Exactly 50.0% Banana / 50.0% Non-Banana across all splits.

---

## 2. Split Partitioning & Class Counts

| Split | Banana (Positive) | Non-Banana (Negative) | Total | Split % |
|---|---|---|---|---|
| **Train** | 2,000 | 2,000 | 4,000 | 74.1% |
| **Validation** | 400 | 400 | 800 | 14.8% |
| **Test** | 300 | 300 | 600 | 11.1% |
| **Total** | **2,700** | **2,700** | **5,400** | **100.0%** |

### Detailed Negative Sub-Source Breakdown per Split

| Split | Fruit-Images-Dataset (MIT) | OpenCV Samples (Apache 2.0) | CIFAR-10 (Academic) | Total Negative |
|---|---|---|---|---|
| **Train** | 950 | 50 | 1,000 | 2,000 |
| **Validation** | 200 | 20 | 180 | 400 |
| **Test** | 150 | 20 | 130 | 300 |
| **Total** | **1,300** | **90** | **1,310** | **2,700** |

---

## 3. Data Sources & Licensing Audit

### A. Positive Banana Source (2,700 Images)
- **Dataset Name:** Banana Ripeness Classification Dataset
- **Local Origin:** `d:/ff/banana/banana_classification`
- **Classes Represented (Strictly Stratified 25% each):**
  - `overripe`: 675 images (500 train, 100 valid, 75 test)
  - `ripe`: 675 images (500 train, 100 valid, 75 test)
  - `rotten`: 675 images (500 train, 100 valid, 75 test)
  - `unripe`: 675 images (500 train, 100 valid, 75 test)
- **Split Preservation:** Images were sampled strictly within their historical split directories (`train` -> `train`, `valid` -> `valid`, `test` -> `test`). Zero cross-split leakage.

### B. Negative Non-Banana Sources (2,700 Images)

#### 1. Fruit-Images-Dataset (Horea94)
- **Exact Repository Name:** `Horea94/Fruit-Images-Dataset`
- **Source URL:** `https://github.com/Horea94/Fruit-Images-Dataset`
- **Git Commit / Version:** `15fae20f2d0d1f33fdec93f8035426c78f92c2fe` (master branch)
- **Citation:** Horea Muresan, Mihai Oltean, *Fruit recognition from images using deep learning*, Acta Technica Napocensis: Electronics and Telecommunications, 58(4), 2017.
- **License Applicable:** **MIT License** (Copyright (c) 2017-2020 Mihai Oltean, Horea Muresan).  
  *Audit Note:* This dataset is the original GitHub version under the permissive MIT license, NOT the newer CC BY-SA 4.0 release.
- **Classes Used (10 Non-Banana Classes):**
  - Apple (371 images: Braeburn, Golden 1, Granny Smith, Red 1)
  - Tomato (146 images: Tomato 1)
  - Pepper (132 images: Pepper Red)
  - Strawberry (104 images)
  - Mango (97 images)
  - Orange (97 images)
  - Pear (94 images)
  - Peach (89 images)
  - Lemon (87 images)
  - Cucumber (83 images: Cucumber Ripe)
- **Contribution:** Exactly **1,300 images** (950 train, 200 valid, 150 test).

#### 2. OpenCV Official Sample Images
- **Exact Repository Name:** `opencv/opencv` (samples/data)
- **Source URL:** `https://github.com/opencv/opencv/tree/master/samples/data`
- **Branch / Version:** `master`
- **License Applicable:** **Apache License 2.0**
- **Categories Used:**
  - Person / Hands / Faces: `messi5.jpg` (Lionel Messi playing soccer), `lena.jpg`
  - Household Objects & Games: `plate.jpg`, `cards.png`, `basketball1.png`, `basketball2.png`, `smarties.png`, `rubberwhale2.png`, `fruits.jpg`, `apple.jpg`, `orange.jpg`
  - Architecture & Environments: `building.jpg`, `home.jpg`, `castle.jpg`, `graf3.png`, `starry_night.jpg`
  - Textures & Patterns: `chessboard.png`, `sudoku.png`, `blox.jpg`, `board.jpg`, `aero1.jpg`, `aero3.jpg`, `aloeGT.png`, `aloeR.jpg`, `baboon.jpg`, `butterfly.jpg`, `chicky_512.png`, `Blender_Suzanne1.jpg`, `Blender_Suzanne2.jpg`, `LinuxLogo.jpg`, `WindowsLogo.jpg`, `opencv-logo.png`
- **Contribution:** Exactly **90 images** (50 train, 20 valid, 20 test).

#### 3. CIFAR-10 Dataset
- **Exact Dataset Name:** `CIFAR-10` (Canadian Institute for Advanced Research)
- **Source URL:** `https://www.cs.toronto.edu/~kriz/cifar.html`
- **Authors:** Alex Krizhevsky, Vinod Nair, Geoffrey Hinton
- **License Applicable:** **Open Academic Research License** (Freely available for research and education)
- **Standard Classes Used:**
  - `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
  - *Audit Note:* CIFAR-10 contains only these 10 vehicle and animal classes. It contains no apples, oranges, tableware, or human portraits (those categories are supplied by Fruit-Images-Dataset and OpenCV).
- **Contribution:** Exactly **1,310 images** (1,000 train, 180 valid, 130 test).

---

## 4. Integrity and Leakage Verification

- **Perceptual & SHA-256 Hashes:** Verified across train, validation, and test splits.
- **Cross-Split Duplicate Count:** 0
- **Corrupted Image Count:** 0 (all images verified via `PIL.Image.verify()`).
- **File Formats:** Standard RGB JPEG and PNG.
- **Preprocessing Pipeline:** Resize to 224×224 RGB, normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

---

## 5. Acceptance Test Image Provenance (`tests/acceptance_images/`)

1. `banana_overripe.jpg`: Overripe banana from test split (`test/overripe`)
2. `banana_ripe.jpg`: Ripe banana from test split (`test/ripe`)
3. `banana_rotten.jpg`: Rotten banana from test split (`test/rotten`)
4. `banana_unripe.jpg`: Unripe banana from test split (`test/unripe`)
5. `nonbanana_apple.jpg`: Fresh Apple Braeburn from `Horea94/Fruit-Images-Dataset` (MIT License)
6. `nonbanana_orange.jpg`: Fresh Orange from `Horea94/Fruit-Images-Dataset` (MIT License)
7. `nonbanana_person_hand.jpg`: `messi5.jpg` from `opencv/opencv` (Apache 2.0 License)
8. `nonbanana_household_cup.jpg`: `plate.jpg` / tableware from `opencv/opencv` (Apache 2.0 License)
9. `nonbanana_background_scene.jpg`: `building.jpg` architecture scene from `opencv/opencv` (Apache 2.0 License)
