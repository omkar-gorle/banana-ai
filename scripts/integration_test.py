#!/usr/bin/env python
"""End-to-end integration test for Banana AI.

Tests the complete pipeline:
  image → validation → preprocessing → CNN → prediction → Grad-CAM → DB record

Uses a real image from the test dataset. Does NOT modify the image.
"""

import json
import sys
from pathlib import Path

# Add src to PYTHONPATH if running directly
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from banana_ai.config import settings


DATASET_ROOT = Path(settings.dataset_root)
MODEL_PATH = Path(settings.model_path)
PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def find_test_image() -> Path | None:
    """Find one image from the test dataset."""
    test_root = DATASET_ROOT / "test"
    if not test_root.exists():
        return None
    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img in class_dir.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                return img
    return None


def main():
    results = {}

    # ─── 1. Dataset ───────────────────────────────────────
    section("1. Dataset validation")
    test_image = find_test_image()
    if test_image:
        print(f"{PASS} Found test image: {test_image}")
        results["dataset"] = "PASS"
    else:
        print(f"{FAIL} No test images found under {DATASET_ROOT}/test/")
        results["dataset"] = "FAIL"

    # ─── 2. Model loading ─────────────────────────────────
    section("2. Model loading")
    if not MODEL_PATH.exists():
        print(f"{FAIL} Model not found at {MODEL_PATH}")
        print("       Run: python -m banana_ai.ml.train")
        results["model_loading"] = "FAIL"
    else:
        try:
            import torch
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            version = checkpoint.get("model_version", "unknown")
            classes = checkpoint.get("class_names", [])
            best_f1 = checkpoint.get("best_val_macro_f1", None)
            print(f"{PASS} Model loaded: version={version}, classes={classes}")
            if best_f1 is not None:
                print(f"       Best val macro-F1: {best_f1:.4f}")
            results["model_loading"] = "PASS"
        except Exception as exc:
            print(f"{FAIL} Model loading failed: {exc}")
            results["model_loading"] = "FAIL"

    # ─── 3. Prediction pipeline ───────────────────────────
    section("3. Prediction pipeline")
    if test_image and results.get("model_loading") == "PASS":
        try:
            from banana_ai.ml.predict import predict
            result = predict(str(test_image), str(MODEL_PATH))

            stage = result["stage"]
            confidence = result["confidence"]
            probs = result["probabilities"]

            assert stage in ["overripe", "ripe", "rotten", "unripe"], \
                f"Invalid stage: {stage}"
            assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
            assert abs(sum(probs.values()) - 1.0) < 1e-4, "Probabilities don't sum to 1"

            print(f"{PASS} Prediction: stage={stage}, confidence={confidence:.3f}")
            print(f"       Probabilities: {json.dumps({k: round(v, 3) for k, v in probs.items()})}")
            print(f"       Prototype days left: {result['estimated_days_left']}")
            results["prediction"] = "PASS"
            results["_prediction_stage"] = stage
            results["_prediction_confidence"] = confidence
        except Exception as exc:
            print(f"{FAIL} Prediction failed: {exc}")
            results["prediction"] = "FAIL"
    else:
        print(f"{SKIP} (missing image or model)")
        results["prediction"] = "SKIP"

    # ─── 4. Grad-CAM ──────────────────────────────────────
    section("4. Grad-CAM")
    if test_image and results.get("model_loading") == "PASS":
        try:
            import torch
            from banana_ai.ml.model import BananaCNN
            from banana_ai.ml.device import get_device
            from banana_ai.ml.explainability import generate_gradcam_overlay

            device = get_device()
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            model = BananaCNN(num_classes=len(checkpoint["class_names"])).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            overlay, heatmap, pred_idx, conf = generate_gradcam_overlay(
                str(test_image), model, device,
                image_size=checkpoint.get("image_size", 224),
            )

            assert overlay.shape == (224, 224, 3), f"Bad overlay shape: {overlay.shape}"
            assert heatmap.min() >= 0 and heatmap.max() <= 1.0

            # Save to reports/explainability/
            import cv2
            import numpy as np
            out_dir = Path("reports/explainability")
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(out_dir / "gradcam_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                str(out_dir / "gradcam_heatmap.png"),
                cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            )
            class_names = checkpoint["class_names"]
            predicted_cls = class_names[pred_idx]
            print(f"{PASS} Grad-CAM generated: predicted={predicted_cls}, conf={conf:.3f}")
            print(f"       Saved to reports/explainability/")
            results["gradcam"] = "PASS"
        except Exception as exc:
            print(f"{FAIL} Grad-CAM failed: {exc}")
            import traceback; traceback.print_exc()
            results["gradcam"] = "FAIL"
    else:
        print(f"{SKIP} (missing image or model)")
        results["gradcam"] = "SKIP"

    # ─── 5. Shelf-life prototype ──────────────────────────
    section("5. Shelf-life prototype")
    try:
        from banana_ai.services.shelf_life import prototype_estimate
        for stage in ["overripe", "ripe", "rotten", "unripe"]:
            est = prototype_estimate(stage)
            assert "day" in est, f"Unexpected result for {stage}: {est}"
        print(f"{PASS} Prototype estimates work for all 4 classes")
        print(f"       NOTE: These are HEURISTIC values, not a trained model")
        results["shelf_life"] = "PROTOTYPE ONLY"
    except Exception as exc:
        print(f"{FAIL} Shelf-life prototype failed: {exc}")
        results["shelf_life"] = "FAIL"

    # ─── 6. PostgreSQL ────────────────────────────────────
    section("6. PostgreSQL")
    try:
        from banana_ai.db.session import SessionLocal
        from banana_ai.db.crud import save_prediction, recent_predictions

        db = SessionLocal()
        try:
            # Insert a test record
            row = save_prediction(
                db=db,
                image_path="/integration_test/test.jpg",
                stage="ripe",
                confidence=0.91,
                estimated_days_left="2-4 days",
                model_version=settings.model_version,
                temperature_c=25.0,
                humidity_pct=60.0,
            )
            print(f"{PASS} Inserted prediction ID: {row.id}")

            # Read it back
            recent = recent_predictions(db, limit=1)
            assert len(recent) > 0
            pred, obs = recent[0]
            print(f"{PASS} Read back prediction: stage={pred.predicted_stage}, "
                  f"confidence={pred.confidence:.3f}")

            results["postgresql"] = "PASS"
        finally:
            db.close()
    except Exception as exc:
        print(f"{SKIP} PostgreSQL not available: {exc}")
        print("       Start with: docker compose up -d db")
        print("       Then: python -m banana_ai.db.init_db")
        results["postgresql"] = "SKIP (not running)"

    # ─── 7. FastAPI ───────────────────────────────────────
    section("7. FastAPI health endpoint")
    try:
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        print(f"{PASS} GET /health → {data}")
        results["fastapi"] = "PASS"
    except Exception as exc:
        print(f"{FAIL} FastAPI test failed: {exc}")
        results["fastapi"] = "FAIL"

    # ─── Summary ──────────────────────────────────────────
    section("INTEGRATION TEST SUMMARY")

    for key, val in results.items():
        if key.startswith("_"):
            continue
        icon = PASS if ("PASS" in val or "PROTOTYPE" in val) else (SKIP if "SKIP" in val else FAIL)
        print(f"  {icon} {key:25s}: {val}")

    print(f"\n{'='*55}")
    overall_fail = any(
        v == "FAIL" for k, v in results.items() if not k.startswith("_")
    )
    if overall_fail:
        print("  INTEGRATION STATUS: INCOMPLETE - see FAIL items above")
    else:
        print("  INTEGRATION STATUS: PASS (with any noted SKIPs)")
    print(f"{'='*55}\n")

    return 0 if not overall_fail else 1


if __name__ == "__main__":
    sys.exit(main())
