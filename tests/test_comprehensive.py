"""Comprehensive test suite for BananaAI.

Tests are organized by component:
- Model construction, parameters, forward pass
- Device detection, CPU fallback, threading
- Dataset class discovery
- Prediction pipeline
- Shelf-life prototype
- Database models
- API endpoints
- Explainability (Grad-CAM)
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from banana_ai.ml.model import BananaCNN, CLASS_NAMES
from banana_ai.ml.device import (
    configure_cpu_threads,
    get_device,
    get_device_name,
    get_device_report,
)
from banana_ai.services.shelf_life import (
    prototype_estimate,
    STAGE_RANGES,
    ShelfLifeEstimate,
)


# ============================================================
# MODEL TESTS
# ============================================================


class TestBananaCNN:
    """Test the BananaCNN model architecture."""

    def test_model_construction(self):
        model = BananaCNN(num_classes=4)
        assert isinstance(model, torch.nn.Module)

    def test_model_output_shape(self):
        model = BananaCNN(num_classes=4)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 4)

    def test_model_single_image(self):
        model = BananaCNN(num_classes=4)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 4)

    def test_parameter_count_sanity(self):
        """Parameter count should be approximately 422,788."""
        model = BananaCNN(num_classes=4)
        total = sum(p.numel() for p in model.parameters())
        assert 400_000 < total < 500_000, f"Unexpected parameter count: {total}"

    def test_trainable_parameters(self):
        model = BananaCNN(num_classes=4)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == total, "All parameters should be trainable"

    def test_forward_pass_no_nan(self):
        model = BananaCNN(num_classes=4)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert not torch.isnan(output).any()

    def test_softmax_sums_to_one(self):
        model = BananaCNN(num_classes=4)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_different_num_classes(self):
        model = BananaCNN(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 10)

    def test_class_names_defined(self):
        assert CLASS_NAMES == ["overripe", "ripe", "rotten", "unripe"]
        assert len(CLASS_NAMES) == 4


# ============================================================
# DEVICE TESTS
# ============================================================


class TestDevice:
    """Test device detection and CPU configuration."""

    def test_device_detection_does_not_crash(self):
        assert get_device().type in {"cpu", "cuda", "xpu"}

    def test_cpu_fallback_is_available(self):
        assert torch.device("cpu").type == "cpu"

    def test_forward_pass_on_selected_device(self):
        device = get_device()
        model = BananaCNN(num_classes=4).to(device)
        output = model(torch.randn(1, 3, 224, 224, device=device))
        assert output.shape == (1, 4)

    def test_checkpoint_save_load_with_map_location(self, tmp_path):
        path = tmp_path / "checkpoint.pt"
        model = BananaCNN(num_classes=4)
        torch.save({"model_state_dict": model.state_dict()}, path)
        checkpoint = torch.load(
            path, map_location=torch.device("cpu"), weights_only=False
        )
        restored = BananaCNN(num_classes=4)
        restored.load_state_dict(checkpoint["model_state_dict"])

    def test_device_report_has_valid_information(self):
        report = get_device_report()
        assert report["selected_device"] in {"cpu", "cuda", "xpu"}
        assert isinstance(report["cuda_available"], bool)
        assert isinstance(report["xpu_available"], bool)
        assert report["torch_cpu_threads"] >= 1
        assert get_device_name()

    def test_cpu_thread_configuration(self, monkeypatch):
        monkeypatch.setenv("BANANA_TORCH_THREADS", "4")
        assert configure_cpu_threads() == 4

    def test_cpu_thread_default(self, monkeypatch):
        monkeypatch.delenv("BANANA_TORCH_THREADS", raising=False)
        threads = configure_cpu_threads()
        assert 1 <= threads <= 4

    def test_invalid_thread_count(self, monkeypatch):
        monkeypatch.setenv("BANANA_TORCH_THREADS", "abc")
        with pytest.raises(ValueError):
            configure_cpu_threads()

    def test_negative_thread_count(self, monkeypatch):
        monkeypatch.setenv("BANANA_TORCH_THREADS", "-1")
        with pytest.raises(ValueError):
            configure_cpu_threads()

    def test_checkpoint_with_full_metadata(self, tmp_path):
        """Checkpoint with full metadata can be loaded correctly."""
        path = tmp_path / "full_checkpoint.pt"
        model = BananaCNN(num_classes=4)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": 224,
                "model_version": "banana-cnn-test",
                "best_val_macro_f1": 0.85,
            },
            path,
        )
        checkpoint = torch.load(
            path, map_location=torch.device("cpu"), weights_only=False
        )
        assert checkpoint["class_names"] == CLASS_NAMES
        assert checkpoint["image_size"] == 224
        assert checkpoint["model_version"] == "banana-cnn-test"


# ============================================================
# DATASET TESTS
# ============================================================


class TestDataset:
    """Test dataset loading and preprocessing."""

    def test_class_names_match_expected(self):
        expected = ["overripe", "ripe", "rotten", "unripe"]
        assert CLASS_NAMES == expected

    def test_transforms_produce_correct_shape(self):
        from banana_ai.ml.data import build_transforms

        train_tf, eval_tf = build_transforms(224)

        # Create a dummy PIL image
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

        train_tensor = train_tf(img)
        eval_tensor = eval_tf(img)

        assert train_tensor.shape == (3, 224, 224)
        assert eval_tensor.shape == (3, 224, 224)

    def test_transforms_normalize_range(self):
        from banana_ai.ml.data import build_transforms

        _, eval_tf = build_transforms(224)
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tensor = eval_tf(img)

        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert tensor.min() > -5
        assert tensor.max() < 5


# ============================================================
# PREDICTION TESTS
# ============================================================


class TestPrediction:
    """Test the prediction pipeline."""

    @pytest.fixture
    def dummy_checkpoint(self, tmp_path):
        """Create a temporary model checkpoint."""
        model = BananaCNN(num_classes=4)
        path = tmp_path / "test_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": 224,
                "model_version": "banana-cnn-test",
                "best_val_macro_f1": 0.0,
            },
            path,
        )
        return str(path)

    @pytest.fixture
    def dummy_image(self, tmp_path):
        """Create a temporary test image."""
        from PIL import Image
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        path = tmp_path / "test_banana.jpg"
        img.save(str(path))
        return str(path)

    def test_predict_valid_image(self, dummy_checkpoint, dummy_image):
        from banana_ai.ml.predict import predict

        result = predict(dummy_image, dummy_checkpoint)
        assert "stage" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_class_in_valid_classes(self, dummy_checkpoint, dummy_image):
        from banana_ai.ml.predict import predict

        result = predict(dummy_image, dummy_checkpoint)
        assert result["stage"] in CLASS_NAMES

    def test_predict_probabilities_sum_to_one(self, dummy_checkpoint, dummy_image):
        from banana_ai.ml.predict import predict

        result = predict(dummy_image, dummy_checkpoint)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_predict_confidence_range(self, dummy_checkpoint, dummy_image):
        from banana_ai.ml.predict import predict

        result = predict(dummy_image, dummy_checkpoint)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_all_classes_in_probabilities(self, dummy_checkpoint, dummy_image):
        from banana_ai.ml.predict import predict

        result = predict(dummy_image, dummy_checkpoint)
        for name in CLASS_NAMES:
            assert name in result["probabilities"]

    def test_predict_includes_estimated_days(self, dummy_checkpoint, dummy_image):
        from banana_ai.ml.predict import predict

        result = predict(dummy_image, dummy_checkpoint)
        assert "estimated_days_left" in result


# ============================================================
# SHELF-LIFE TESTS
# ============================================================


class TestShelfLife:
    """Test the prototype shelf-life estimation."""

    def test_prototype_estimate_rotten(self):
        assert prototype_estimate("rotten") == "0 day(s)"

    def test_prototype_estimate_unripe(self):
        assert "days" in prototype_estimate("unripe")

    def test_prototype_estimate_ripe(self):
        result = prototype_estimate("ripe")
        assert "days" in result

    def test_prototype_estimate_overripe(self):
        result = prototype_estimate("overripe")
        assert "day" in result

    def test_prototype_estimate_unknown_stage(self):
        result = prototype_estimate("unknown_stage")
        assert "0 day(s)" == result

    def test_all_stages_covered(self):
        for stage in CLASS_NAMES:
            assert stage in STAGE_RANGES

    def test_shelf_life_estimate_display(self):
        est = ShelfLifeEstimate(minimum_days=2, maximum_days=4, method="test")
        assert est.display() == "2-4 days"

    def test_shelf_life_estimate_single_day(self):
        est = ShelfLifeEstimate(minimum_days=0, maximum_days=0, method="test")
        assert est.display() == "0 day(s)"

    def test_prototype_is_not_trained_model(self):
        """Verify the prototype method is clearly labelled."""
        est = ShelfLifeEstimate(
            minimum_days=2, maximum_days=4, method="prototype_stage_rule"
        )
        assert "prototype" in est.method


# ============================================================
# DATABASE MODEL TESTS
# ============================================================


class TestDatabaseModels:
    """Test database model definitions (no database connection required)."""

    def test_observation_model_exists(self):
        from banana_ai.db.models import Observation
        assert hasattr(Observation, "__tablename__")
        assert Observation.__tablename__ == "observations"

    def test_prediction_model_exists(self):
        from banana_ai.db.models import Prediction
        assert hasattr(Prediction, "__tablename__")
        assert Prediction.__tablename__ == "predictions"

    def test_observation_has_required_columns(self):
        from banana_ai.db.models import Observation
        columns = {c.name for c in Observation.__table__.columns}
        assert "id" in columns
        assert "image_path" in columns
        assert "temperature_c" in columns
        assert "humidity_pct" in columns
        assert "observed_at" in columns

    def test_prediction_has_required_columns(self):
        from banana_ai.db.models import Prediction
        columns = {c.name for c in Prediction.__table__.columns}
        assert "id" in columns
        assert "observation_id" in columns
        assert "predicted_stage" in columns
        assert "confidence" in columns
        assert "estimated_days_left" in columns
        assert "model_version" in columns
        assert "created_at" in columns


# ============================================================
# API TESTS
# ============================================================


class TestAPI:
    """Test FastAPI endpoints (without database)."""

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ============================================================
# EXPLAINABILITY TESTS
# ============================================================


class TestExplainability:
    """Test Grad-CAM implementation."""

    def test_gradcam_generates_heatmap(self):
        from banana_ai.ml.explainability import GradCAM

        model = BananaCNN(num_classes=4)
        model.eval()

        grad_cam = GradCAM(model)
        try:
            input_tensor = torch.randn(1, 3, 224, 224)
            heatmap = grad_cam.generate(input_tensor)

            assert isinstance(heatmap, np.ndarray)
            assert heatmap.min() >= 0
            assert heatmap.max() <= 1.0
        finally:
            grad_cam.cleanup()

    def test_gradcam_with_specific_class(self):
        from banana_ai.ml.explainability import GradCAM

        model = BananaCNN(num_classes=4)
        model.eval()

        grad_cam = GradCAM(model)
        try:
            input_tensor = torch.randn(1, 3, 224, 224)
            heatmap = grad_cam.generate(input_tensor, class_idx=2)

            assert isinstance(heatmap, np.ndarray)
            assert heatmap.shape[0] > 0
            assert heatmap.shape[1] > 0
        finally:
            grad_cam.cleanup()

    def test_gradcam_overlay_on_dummy_image(self, tmp_path):
        from banana_ai.ml.explainability import generate_gradcam_overlay
        from PIL import Image

        # Create a dummy image
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        img_path = tmp_path / "test_banana.jpg"
        img.save(str(img_path))

        model = BananaCNN(num_classes=4)
        model.eval()
        device = torch.device("cpu")

        overlay, heatmap, pred_idx, confidence = generate_gradcam_overlay(
            str(img_path), model, device
        )

        assert overlay.shape == (224, 224, 3)
        assert 0 <= pred_idx < 4
        assert 0.0 <= confidence <= 1.0
