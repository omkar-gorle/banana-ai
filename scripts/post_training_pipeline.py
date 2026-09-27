#!/usr/bin/env python
"""Post-training pipeline: runs all post-V1-training steps automatically.

Run this after V1 training completes:
    python scripts/post_training_pipeline.py

Steps:
  1. Evaluate V1 on test set (extended)
  2. Train V2 (class-weight experiment)
  3. Evaluate V2 on test set (extended)
  4. Compare V1 vs V2
  5. Generate error analysis
  6. Generate Grad-CAM example
  7. Run integration test
  8. Generate final report
"""

import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
DATASET_ROOT = "D:/ff/banana/banana_classification"
SRC = str(Path(__file__).parents[1] / "src")


def run(cmd: list, label: str):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    env = {**__import__('os').environ, "PYTHONPATH": SRC}
    result = subprocess.run(cmd, capture_output=False, env=env)
    if result.returncode != 0:
        print(f"FAILED with code {result.returncode}")
    return result.returncode == 0


def main():
    steps = []

    # 1. Evaluate V1
    ok = run([
        PYTHON, "-m", "banana_ai.ml.evaluate_extended",
        "--checkpoint", "models/banana_cnn_best.pt",
        "--dataset-root", DATASET_ROOT,
        "--output-dir", "reports/v1",
        "--version-label", "v1",
    ], "Step 1: Evaluate V1 on test set")
    steps.append(("V1 Evaluation", ok))

    # 2. Train V2
    ok = run([
        PYTHON, "-m", "banana_ai.ml.train_v2",
        "--dataset-root", DATASET_ROOT,
    ], "Step 2: Train V2 (class weights)")
    steps.append(("V2 Training", ok))

    # 3. Evaluate V2
    v2_ckpt = Path("models/banana_cnn_v2.pt")
    if v2_ckpt.exists():
        ok = run([
            PYTHON, "-m", "banana_ai.ml.evaluate_extended",
            "--checkpoint", str(v2_ckpt),
            "--dataset-root", DATASET_ROOT,
            "--output-dir", "reports/v2",
            "--version-label", "v2",
        ], "Step 3: Evaluate V2 on test set")
        steps.append(("V2 Evaluation", ok))
    else:
        print("V2 checkpoint not found — skipping V2 evaluation")
        steps.append(("V2 Evaluation", False))

    # 4. Compare V1 vs V2
    ok = run([
        PYTHON, "-m", "banana_ai.ml.analysis",
        "--v1-dir", "reports/v1",
        "--v2-dir", "reports/v2",
        "--output", "reports/model_comparison.md",
    ], "Step 4: V1 vs V2 comparison")
    steps.append(("V1/V2 Comparison", ok))

    # 5. Error analysis
    try:
        sys.path.insert(0, SRC)
        from banana_ai.ml.analysis import generate_error_analysis
        generate_error_analysis(
            v1_metrics_path="reports/v1/test_metrics.json",
            output_path="reports/error_analysis.md",
        )
        steps.append(("Error Analysis", True))
        print("Error analysis generated.")
    except Exception as exc:
        print(f"Error analysis failed: {exc}")
        steps.append(("Error Analysis", False))

    # 6. Grad-CAM example
    import os
    test_root = Path(DATASET_ROOT) / "test"
    test_image = None
    for cls_dir in sorted(test_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                test_image = img
                break
        if test_image:
            break

    if test_image:
        ok = run([
            PYTHON, "-m", "banana_ai.ml.explainability",
            "--image", str(test_image),
            "--model", "models/banana_cnn_best.pt",
            "--output-dir", "reports/explainability",
        ], "Step 6: Generate Grad-CAM example")
        steps.append(("Grad-CAM", ok))
    else:
        print("No test image found — skipping Grad-CAM")
        steps.append(("Grad-CAM", False))

    # 7. Integration test
    ok = run([
        PYTHON, "scripts/integration_test.py",
    ], "Step 7: Integration test")
    steps.append(("Integration Test", ok))

    # 8. Update final report with real values
    ok = run([
        PYTHON, "scripts/update_final_report.py",
    ], "Step 8: Update FINAL_PROJECT_REPORT.md")
    steps.append(("Final Report", ok))

    # Summary
    print(f"\n{'='*55}")
    print("  POST-TRAINING PIPELINE SUMMARY")
    print(f"{'='*55}")
    for label, success in steps:
        icon = "[PASS]" if success else "[FAIL]"
        print(f"  {icon} {label}")

    all_ok = all(ok for _, ok in steps)
    print(f"\n  Overall: {'COMPLETE' if all_ok else 'INCOMPLETE'}")
    print(f"{'='*55}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
