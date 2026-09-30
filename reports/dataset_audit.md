# Dataset Audit Report

## 1. Existing Dataset (`banana_classification`)
- **Structure:** Separated into `train`, `valid`, and `test` directories, each containing `overripe`, `ripe`, `rotten`, and `unripe` classes.
- **Image Count:** 
  - Train: 11,793 images
  - Valid: 1,123 images
  - Test: 562 images
  - Total: 13,478 images
- **Formats:** JPEG, 416x416.
- **Corrupted:** 0 images.
- **Duplicates:** None detected within or across the dataset based on hash comparison.

## 2. New Archive 1 (`archive (1)`)
- **Structure:** COCO format JSON (`_annotations.coco.json`) and a single `labeled banana images` directory.
- **Classes:** Labeled with `degree1` through `degree8` rather than categorical ripeness (e.g., unripe, ripe).
- **Image Count:** 1,000 images, 1,002 annotations.
  - Distribution: `degree1`: 159, `degree2`: 72, `degree3`: 53, `degree4`: 89, `degree5`: 102, `degree6`: 247, `degree7`: 188, `degree8`: 92.
- **Formats:** JPEG.
- **Corrupted:** 0 images.
- **Duplicates:** 13 internal duplicates found. No cross-dataset duplicates with `banana_classification` or `archive (2)`.
- **Conclusion for usage:** Because the labels (`degree1` to `degree8`) cannot be scientifically and safely mapped to the `unripe/ripe/overripe/rotten` paradigm without making strong unverified assumptions, this dataset will **not** be merged into the general model training to prevent data leakage and incorrect labeling. 

## 3. New Archive 2 (`archive (2)`)
- **Structure:** `dataset/train` and `dataset/test` directories containing `Overripe`, `Ripe`, and `Unripe` classes. It is missing the `Rotten` class.
- **Image Count:**
  - Train: 480 (160 overripe, 160 ripe, 160 unripe)
  - Test: 120 (40 overripe, 40 ripe, 40 unripe)
  - Total: 600 images
- **Formats:** JPEG, 640x480.
- **Corrupted:** 0 images.
- **Duplicates:** 2 internal duplicates. No cross-dataset duplicates.
- **Conclusion for usage:** These images are directly compatible with our class paradigm. We will carefully deduplicate them and merge them into the training and validation sets. The missing `rotten` class is not a major issue as the existing dataset has 4,000+ rotten images.

## 4. Shelf-Life Modeling Analysis
- **Finding:** Neither of the new archives nor the existing dataset contain true longitudinal data (the same banana tracked over multiple days) or explicit annotations like `remaining_days`, `temperature`, or `humidity`. 
- **Recommendation:** We cannot scientifically train a true shelf-life regression model. Instead, we must rely on a heuristic-based "Estimated Good-to-Eat Window" linked to the predicted ripeness category, as outlined in the requirements.

## 5. Merging & Preparation Plan
- `archive (2)` data will be parsed, hashes compared, and distinct non-duplicate images will be injected into our existing `banana_classification` training data pool. We will ensure the `test` split remains pristine.
