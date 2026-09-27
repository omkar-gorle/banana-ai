# Dataset Audit Report

**Audit date:** 2026-09-27. **Dataset:** `D:\ff\banana\banana_classification`. **Model:** `D:\ff\banana\banana_ai_from_scratch\models\banana_cnn_v2.pt`.

## Actual dataset inventory

| Split | overripe | ripe | rotten | unripe | total |
|---|---:|---:|---:|---:|---:|
| train | 2349 | 3522 | 4020 | 1902 | 11793 |
| valid | 229 | 339 | 388 | 167 | 1123 |
| test | 113 | 154 | 185 | 110 | 562 |

Total image files: **13478**. Classes are exactly `['overripe', 'ripe', 'rotten', 'unripe']`. No `val` split exists at this dataset path; the repository loader accepts `valid`.

## Production checkpoint

SHA-256: `cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d`; size 1706343 bytes.
Metadata: class_names=['overripe', 'ripe', 'rotten', 'unripe'], image_size=224, model_version=banana-cnn-v2, experiment=class-weight-balancing, best_val_macro_f1=0.9608173664337731.

## Deterministic 40-image audit

Sampling: each test class sorted by filename (case-insensitive), `random.Random(42).sample(..., 10)`. Predictions use the production V2 checkpoint, RGB conversion, resize 224x224, tensor conversion, and ImageNet normalization. Full rows are in `dataset_audit_results.csv`; contact sheet: `test_40_contact_sheet.jpg`.
- overripe: 10 audited; correct 9/10; predictions Counter({'overripe': 9, 'ripe': 1})
- ripe: 10 audited; correct 9/10; predictions Counter({'ripe': 9, 'rotten': 1})
- rotten: 10 audited; correct 10/10; predictions Counter({'rotten': 10})
- unripe: 10 audited; correct 10/10; predictions Counter({'unripe': 10})

## Rotten 20 probability audit

A separate seeded-42 sample of 20 sorted test/rotten images was scored. All four probabilities are recorded below and the contact sheet is `rotten_20_contact_sheet.jpg`.

| rank | filename | predicted | confidence | overripe | ripe | rotten | unripe |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `musa-acuminata-rotten-829716e2-2653-11ec-97b3-d8c4975e38aa_jpg.rf.a68db800b7403e39f4d4365a76b8c056.jpg` | rotten | 0.999378 | 0.000398 | 0.000057 | 0.999378 | 0.000167 |
| 2 | `musa-acuminata-banana-a95acdfd-394a-11ec-b543-d8c4975e38aa_jpg.rf.ebdb5a7996bc548cb01d93046fd2406f.jpg` | rotten | 0.999923 | 0.000000 | 0.000077 | 0.999923 | 0.000000 |
| 3 | `musa-acuminata-ripe-966b5e21-1d0a-11ec-ab19-d8c4975e38aa_jpg.rf.55c5a39035dcb40a154c1eaa54dafc4b.jpg` | rotten | 0.999698 | 0.000009 | 0.000293 | 0.999698 | 0.000000 |
| 4 | `musa-acuminata-rotten-87043990-2653-11ec-aa07-d8c4975e38aa_jpg.rf.601bb875ad9ac92a330a390f25a28cf9.jpg` | rotten | 0.980493 | 0.001648 | 0.017858 | 0.980493 | 0.000001 |
| 5 | `musa-acuminata-banana-9ebb3756-394a-11ec-a9a8-d8c4975e38aa_jpg.rf.6c18a51e74bcc0940628a11283e5d4a2.jpg` | rotten | 0.999382 | 0.000000 | 0.000618 | 0.999382 | 0.000000 |
| 6 | `musa-acuminata-banana-9beb52ba-394a-11ec-97e9-d8c4975e38aa_jpg.rf.fc713c69733ebeb7a19bfe6353de777d.jpg` | rotten | 0.973056 | 0.000000 | 0.026944 | 0.973056 | 0.000000 |
| 7 | `musa-acuminata-rotten-8a7850bd-2653-11ec-af1f-d8c4975e38aa_jpg.rf.4eb5576de82523bc03fc08d83e93a953.jpg` | rotten | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| 8 | `musa-acuminata-banana-9df5450b-394a-11ec-bd44-d8c4975e38aa_jpg.rf.754d58dad21a5ef7ab0566d88555d130.jpg` | rotten | 0.990460 | 0.000000 | 0.009540 | 0.990460 | 0.000000 |
| 9 | `musa-acuminata-rotten-88635b8d-2653-11ec-b12c-d8c4975e38aa_jpg.rf.06d0607f215eb4e3e848a3b9df5ac5d6.jpg` | rotten | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| 10 | `musa-acuminata-rotten-87d295b9-2653-11ec-9dba-d8c4975e38aa_jpg.rf.77813f9192428e6eecc2ce86d5f6bcbf.jpg` | rotten | 0.916625 | 0.016436 | 0.046654 | 0.916625 | 0.020286 |
| 11 | `musa-acuminata-unripe-5d01c29b-2653-11ec-b9cf-d8c4975e38aa---Copy_jpg.rf.c800884215eae640819e500e15939484.jpg` | rotten | 0.948754 | 0.000952 | 0.000241 | 0.948754 | 0.050053 |
| 12 | `musa-acuminata-rotten-8204b6e4-2653-11ec-88a5-d8c4975e38aa_jpg.rf.5853255d392c37e01f38778d2d0cc919.jpg` | rotten | 0.960355 | 0.000279 | 0.000046 | 0.960355 | 0.039320 |
| 13 | `musa-acuminata-banana-967cdc40-394a-11ec-b8d8-d8c4975e38aa_jpg.rf.99affac3588307bb1a13af9713cd36db.jpg` | rotten | 0.645714 | 0.000002 | 0.354284 | 0.645714 | 0.000000 |
| 14 | `musa-acuminata-rotten-8f68ed3d-2653-11ec-b039-d8c4975e38aa_jpg.rf.87438385fadce9752493e88639a3a4a0.jpg` | rotten | 0.650402 | 0.000066 | 0.348612 | 0.650402 | 0.000920 |
| 15 | `musa-acuminata-rotten-b6d3c51d-1d0a-11ec-9f4a-d8c4975e38aa_jpg.rf.a1715f6d638a6adf681d54298e3ddd3f.jpg` | rotten | 0.999729 | 0.000087 | 0.000183 | 0.999729 | 0.000001 |
| 16 | `musa-acuminata-banana-a33bd066-394a-11ec-98de-d8c4975e38aa_jpg.rf.1eb1b91188935907f6450edbcd01303d.jpg` | rotten | 0.994426 | 0.000000 | 0.005574 | 0.994426 | 0.000000 |
| 17 | `musa-acuminata-rotten-89d9466c-2653-11ec-88e8-d8c4975e38aa_jpg.rf.cbc17111a4b86e3ed3b5c94e94da90df.jpg` | rotten | 0.999170 | 0.000016 | 0.000814 | 0.999170 | 0.000000 |
| 18 | `musa-acuminata-banana-9b59211a-394a-11ec-84e7-d8c4975e38aa_jpg.rf.4123a744568414ec66e4f90afab0f42e.jpg` | rotten | 0.871868 | 0.000000 | 0.128132 | 0.871868 | 0.000000 |
| 19 | `musa-acuminata-rotten-b9d4305e-1d0a-11ec-a8ec-d8c4975e38aa_jpg.rf.f67389b4204bb7d429604690757d686c.jpg` | rotten | 0.999939 | 0.000061 | 0.000000 | 0.999939 | 0.000000 |
| 20 | `musa-acuminata-rotten-8301c283-2653-11ec-b15d-d8c4975e38aa_jpg.rf.ecc601d31c199161d8a9601463b30f9a.jpg` | rotten | 0.996684 | 0.000121 | 0.000537 | 0.996684 | 0.002658 |

## Cross-split hash audit

Hashed all 13478 images with SHA-256. Cross-split duplicate hashes: **0**.
No identical image content was found across train/valid/test.


## Known demo V2 inference

Three found class-specific candidate demo filenames were selected from `test/<class>` by case-insensitive lexical order among filenames containing the class token. These are dataset evidence candidates; no separate screenshot/upload asset is present. Inference used `banana_cnn_v2.pt`, RGB conversion, 224x224 resize, ToTensor, and ImageNet normalization. Dimensions are original `(width x height)` pixels.

| Known class | Exact filename | Original dimensions | Predicted | Confidence | overripe | ripe | rotten | unripe |
|---|---|---:|---|---:|---:|---:|---:|---:|
| overripe | `musa-acuminata-overripe-74d28e6f-2653-11ec-a1c3-d8c4975e38aa_jpg.rf.c494b2c758d1430121a55d0607fd5f29.jpg` | 416 x 416 | ripe | 0.7961139679 | 0.0027513984 | 0.7961139679 | 0.2011030763 | 0.0000315718 |
| rotten | `musa-acuminata-rotten-8125534a-2653-11ec-bf1f-d8c4975e38aa_jpg.rf.1cf1a631554ba70a0254cebba9b37482.jpg` | 416 x 416 | overripe | 0.9417306781 | 0.9417306781 | 0.0000851924 | 0.0581820309 | 0.0000020850 |
| unripe | `musa-acuminata-freshunripe-1cdf07a1-2653-11ec-a4b4-d8c4975e38aa_jpg.rf.9500aa90b98dc6f34859be314aa61dc2.jpg` | 416 x 416 | unripe | 0.9992239475 | 0.0000105049 | 0.0000152846 | 0.0007502646 | 0.9992239475 |

The known ripe filename is **absent/not identified** as a separate known-demo asset. Ripe dataset images exist, but no known ripe demo filename was supplied or discoverable in repository documentation. A search of these three outputs for approximately 99.3% rotten found **no such result**; the rotten candidate above is reported exactly rather than rounded.

## Dataset counts

See Actual dataset inventory above: train 11,793; valid 1,123; test 562; total 13,478.

## Class distribution

Train: overripe 2,349; ripe 3,522; rotten 4,020; unripe 1,902. Valid: 229/339/388/167. Test: 113/154/185/110, in class order overripe/ripe/rotten/unripe.

## Exact screenshot image investigation

No exact screenshot or original upload asset is stored. Three class-specific dataset candidates were inferenced above. The known ripe filename is absent/not identified. No candidate yielded approximately 99.3% rotten; generated Grad-CAM/report PNGs are not source screenshots.

## 40-image test audit

The deterministic seed-42 audit contains 40 rows, 10 per class, in `dataset_audit_results.csv`, with `test_40_contact_sheet.jpg`; observed accuracy was 38/40.

## Rotten-class audit

Twenty separate test/rotten images were scored with all four probabilities and are tabulated above; `rotten_20_contact_sheet.jpg` is provided.

## Other-class audit

The 10-image audits for overripe, ripe, and unripe are included in the 40-row CSV and contact sheet.

## Duplicate/leakage check

All 13,478 image SHA-256 hashes were compared across train/valid/test; zero cross-split duplicates were found.

## Preprocessing and class mapping checks

V2 training/evaluation (`src/banana_ai/ml/data.py`) uses ImageFolder, training resize 224, augmentation (horizontal flip, rotation +/-12 degrees, ColorJitter), ToTensor, and ImageNet normalization mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. Evaluation removes augmentation and retains resize/ToTensor/normalization.
Prediction (`src/banana_ai/ml/predict.py`) uses checkpoint image_size=224, RGB conversion, deterministic resize, ToTensor, and exactly the same normalization as evaluation. Match: **YES**.
ImageFolder alphabetical mapping observed in the dataset: ['overripe', 'ripe', 'rotten', 'unripe']; checkpoint mapping: ['overripe', 'ripe', 'rotten', 'unripe']; exact match: **True**. Evaluation, prediction, Streamlit and FastAPI consume checkpoint class_names for labels/probability keys. `.env` selects V2 (`MODEL_PATH=models/banana_cnn_v2.pt`, `MODEL_VERSION=banana-cnn-v2`), while standalone code defaults still name V1.

## Caveats

The 40-image CSV contains exactly 40 rows. Rotten probability details are intentionally reported in this markdown because the requested CSV is the 40-image audit; the 20 rotten contact sheet and complete four-class probability table provide the detailed rotten audit. No files outside reports/dataset_audit were modified.

## Findings

The production V2 checkpoint and dataset labels agree on the deterministic samples: 38/40 in the broad audit and 20/20 in the separate rotten audit. Two broad-audit errors were boundary cases (overripe -> ripe and ripe -> rotten). The rotten sample includes filenames containing ripe/unripe tokens, but filenames are not labels and no visual relabeling was performed; a dataset-labeling concern therefore remains **UNCERTAIN**, not established. No screenshot-level identity or 99.3% rotten demo match was established.

- Dataset labeling concern: **UNCERTAIN** (requires the original screenshot and/or human visual review; this audit did not alter or reinterpret labels).
- Inference bug: **NO EVIDENCE**. Preprocessing and class-index checks passed; exact screenshot provenance is unavailable.
- Model retraining: **not decided automatically**.

## Recommended next action

If exact screenshot provenance is required, obtain the original upload/screenshot or its filename/hash from the demo owner and rerun this same inference record; otherwise retain the dataset candidates as explicitly labelled evidence, not exact identification.
