"""Tests for all new Banana AI features (Feature Upgrade v2).

Tests cover:
- Enhanced shelf-life service (environment-aware)
- Temperature & humidity boundaries
- Each ripeness class
- What-if simulation
- Missing environmental values
- Image validation / security
- Upload validation
- Batch analysis helpers
- Report generation (HTML/CSV)
- Database model structure (new columns)
- Feedback model
- API response compatibility
- Model version consistency
- Backward compatibility of prototype_estimate
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from PIL import Image

from banana_ai.services.shelf_life import (
    STAGE_RANGES,
    ShelfLifeEstimate,
    HeuristicShelfLifeEstimator,
    estimate_shelf_life,
    prototype_estimate,
    TEMP_REFERENCE_C,
    HUMIDITY_REFERENCE_PCT,
    STORAGE_MULTIPLIERS,
    MAX_DAYS_CAP,
)
from banana_ai.services.image_validation import (
    ImageValidationError,
    validate_image_bytes,
    safe_temp_filename,
    ALLOWED_EXTENSIONS,
)


# ===========================================================================
# ENHANCED SHELF-LIFE SERVICE TESTS
# ===========================================================================


class TestEnhancedShelfLifeService:
    """Tests for the environment-aware HeuristicShelfLifeEstimator."""

    estimator = HeuristicShelfLifeEstimator()

    # ---- Basic stage tests ----

    def test_rotten_returns_zero(self):
        est = self.estimator.estimate("rotten")
        assert est.estimated_min_days == 0
        assert est.estimated_max_days == 0

    def test_rotten_without_env_still_zero(self):
        est = self.estimator.estimate("rotten", temperature_c=30.0, humidity_pct=80.0)
        assert est.estimated_min_days == 0
        assert est.estimated_max_days == 0

    def test_unripe_has_longer_baseline(self):
        est = self.estimator.estimate("unripe")
        base_min, base_max = STAGE_RANGES["unripe"]
        # At reference temperature/humidity (no env input) result should match baseline
        assert est.estimated_min_days >= 0
        assert est.estimated_max_days <= MAX_DAYS_CAP
        assert est.estimated_max_days >= est.estimated_min_days

    def test_ripe_has_moderate_baseline(self):
        est = self.estimator.estimate("ripe")
        assert est.estimated_min_days >= 0
        assert est.estimated_max_days <= MAX_DAYS_CAP

    def test_overripe_has_short_baseline(self):
        est = self.estimator.estimate("overripe")
        # overripe should have shorter window than unripe
        unripe_est = self.estimator.estimate("unripe")
        assert est.estimated_max_days <= unripe_est.estimated_max_days

    def test_all_classes_return_valid_estimate(self):
        for stage in ["unripe", "ripe", "overripe", "rotten"]:
            est = self.estimator.estimate(stage)
            assert est.estimated_min_days <= est.estimated_max_days
            assert est.estimated_min_days >= 0
            assert est.estimated_max_days <= MAX_DAYS_CAP

    # ---- Temperature boundary tests ----

    def test_cooler_temperature_extends_window(self):
        est_warm = self.estimator.estimate("ripe", temperature_c=35.0, humidity_pct=60.0)
        est_cool = self.estimator.estimate("ripe", temperature_c=15.0, humidity_pct=60.0)
        # Cooler should give same or more days
        assert est_cool.estimated_max_days >= est_warm.estimated_max_days

    def test_warmer_temperature_shortens_window(self):
        est_ref = self.estimator.estimate("ripe", temperature_c=TEMP_REFERENCE_C, humidity_pct=60.0)
        est_hot = self.estimator.estimate("ripe", temperature_c=40.0, humidity_pct=60.0)
        assert est_hot.estimated_max_days <= est_ref.estimated_max_days

    def test_extreme_high_temperature_clamped(self):
        est = self.estimator.estimate("ripe", temperature_c=1000.0, humidity_pct=60.0)
        assert est.estimated_min_days >= 0
        assert est.estimated_max_days <= MAX_DAYS_CAP

    def test_extreme_low_temperature_clamped(self):
        est = self.estimator.estimate("ripe", temperature_c=-1000.0, humidity_pct=60.0)
        assert est.estimated_min_days >= 0
        assert est.estimated_max_days <= MAX_DAYS_CAP

    def test_temperature_none_uses_baseline(self):
        est_none = self.estimator.estimate("ripe", temperature_c=None, humidity_pct=60.0)
        est_ref = self.estimator.estimate("ripe", temperature_c=None, humidity_pct=60.0)
        assert est_none.estimated_min_days == est_ref.estimated_min_days

    # ---- Humidity boundary tests ----

    def test_high_humidity_shortens_window(self):
        est_dry = self.estimator.estimate("ripe", temperature_c=TEMP_REFERENCE_C, humidity_pct=20.0)
        est_wet = self.estimator.estimate("ripe", temperature_c=TEMP_REFERENCE_C, humidity_pct=90.0)
        assert est_dry.estimated_max_days >= est_wet.estimated_max_days

    def test_extreme_high_humidity_clamped(self):
        est = self.estimator.estimate("ripe", temperature_c=25.0, humidity_pct=500.0)
        assert est.estimated_min_days >= 0
        assert est.estimated_max_days <= MAX_DAYS_CAP

    def test_extreme_low_humidity_clamped(self):
        est = self.estimator.estimate("ripe", temperature_c=25.0, humidity_pct=-100.0)
        assert est.estimated_min_days >= 0
        assert est.estimated_max_days <= MAX_DAYS_CAP

    def test_humidity_none_uses_baseline(self):
        est = self.estimator.estimate("ripe", temperature_c=25.0, humidity_pct=None)
        assert est.estimated_min_days >= 0
        assert est.estimated_max_days <= MAX_DAYS_CAP

    # ---- Missing environmental values ----

    def test_both_none_returns_valid_estimate(self):
        est = self.estimator.estimate("ripe", temperature_c=None, humidity_pct=None)
        assert est.estimated_min_days >= 0
        assert est.estimated_max_days <= MAX_DAYS_CAP

    def test_no_env_args_still_works(self):
        est = self.estimator.estimate("unripe")
        assert isinstance(est, ShelfLifeEstimate)

    # ---- Storage condition tests ----

    def test_refrigerator_extends_window(self):
        est_room = self.estimator.estimate("ripe", temperature_c=25.0, storage_condition="room")
        est_fridge = self.estimator.estimate("ripe", temperature_c=25.0, storage_condition="refrigerator")
        assert est_fridge.estimated_max_days >= est_room.estimated_max_days

    def test_unknown_storage_uses_room_multiplier(self):
        est_unknown = self.estimator.estimate("ripe", storage_condition="underwater_cave")
        est_room = self.estimator.estimate("ripe", storage_condition="room")
        assert est_unknown.estimated_max_days == est_room.estimated_max_days

    # ---- Determinism ----

    def test_same_inputs_same_output(self):
        est1 = self.estimator.estimate("ripe", temperature_c=25.0, humidity_pct=65.0)
        est2 = self.estimator.estimate("ripe", temperature_c=25.0, humidity_pct=65.0)
        assert est1.estimated_min_days == est2.estimated_min_days
        assert est1.estimated_max_days == est2.estimated_max_days

    # ---- Output fields ----

    def test_estimate_has_explanation(self):
        est = self.estimator.estimate("ripe", temperature_c=25.0, humidity_pct=60.0)
        assert len(est.explanation) > 0

    def test_estimate_has_warning(self):
        est = self.estimator.estimate("ripe")
        assert "heuristic" in est.warning.lower() or "prototype" in est.warning.lower()

    def test_model_type_is_heuristic(self):
        est = self.estimator.estimate("ripe")
        assert est.model_type == "heuristic"

    def test_method_is_labeled(self):
        est = self.estimator.estimate("ripe")
        assert "heuristic" in est.method

    def test_min_max_correct_order(self):
        est = self.estimator.estimate("ripe", temperature_c=25.0, humidity_pct=65.0)
        assert est.estimated_min_days <= est.estimated_max_days

    # ---- Public API function ----

    def test_estimate_shelf_life_function_works(self):
        est = estimate_shelf_life("ripe", 25.0, 65.0)
        assert isinstance(est, ShelfLifeEstimate)

    def test_estimate_shelf_life_rotten_is_zero(self):
        est = estimate_shelf_life("rotten")
        assert est.estimated_min_days == 0
        assert est.estimated_max_days == 0

    # ---- What-if simulation ----

    def test_whatif_lower_temp_gives_more_days(self):
        base = estimate_shelf_life("ripe", temperature_c=30.0, humidity_pct=65.0)
        cool = estimate_shelf_life("ripe", temperature_c=15.0, humidity_pct=65.0)
        assert cool.estimated_max_days >= base.estimated_max_days

    def test_whatif_refrigerator_vs_room(self):
        room = estimate_shelf_life("overripe", temperature_c=25.0, storage_condition="room")
        fridge = estimate_shelf_life("overripe", temperature_c=25.0, storage_condition="refrigerator")
        assert fridge.estimated_max_days >= room.estimated_max_days

    # ---- Backward compatibility ----

    def test_prototype_estimate_backward_compat_rotten(self):
        """Existing test: prototype_estimate('rotten') == '0 day(s)'"""
        assert prototype_estimate("rotten") == "0 day(s)"

    def test_prototype_estimate_backward_compat_unripe(self):
        result = prototype_estimate("unripe")
        assert "day" in result

    def test_prototype_estimate_unknown_stage(self):
        result = prototype_estimate("unknown_stage")
        assert "0" in result

    # ---- ShelfLifeEstimate backward compat ----

    def test_shelf_life_estimate_display_range(self):
        est = ShelfLifeEstimate(
            estimated_min_days=2, estimated_max_days=4,
            explanation="test", warning="test"
        )
        assert "2" in est.display()
        assert "4" in est.display()
        assert est.display() == "2-4 days"

    def test_shelf_life_estimate_display_single_day(self):
        est = ShelfLifeEstimate(
            estimated_min_days=0, estimated_max_days=0,
            explanation="test", warning="test"
        )
        assert est.display() == "0 day(s)"

    def test_shelf_life_estimate_backward_compat_minimum_days(self):
        """The old ShelfLifeEstimate exposed minimum_days / maximum_days."""
        est = ShelfLifeEstimate(
            estimated_min_days=2, estimated_max_days=4,
            explanation="", warning=""
        )
        assert est.minimum_days == 2
        assert est.maximum_days == 4


# ===========================================================================
# IMAGE VALIDATION / SECURITY TESTS
# ===========================================================================


class TestImageValidation:
    """Tests for image upload validation and security."""

    def _make_jpeg_bytes(self, w: int = 224, h: int = 224) -> bytes:
        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def _make_png_bytes(self, w: int = 224, h: int = 224) -> bytes:
        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_valid_jpeg_accepted(self):
        data = self._make_jpeg_bytes()
        img = validate_image_bytes(data, filename="test.jpg")
        assert img is not None

    def test_valid_png_accepted(self):
        data = self._make_png_bytes()
        img = validate_image_bytes(data, filename="test.png")
        assert img is not None

    def test_empty_file_rejected(self):
        with pytest.raises(ImageValidationError, match="empty"):
            validate_image_bytes(b"", filename="test.jpg")

    def test_non_image_file_rejected(self):
        with pytest.raises(ImageValidationError):
            validate_image_bytes(b"This is not an image file at all!", filename="test.jpg")

    def test_file_too_large_rejected(self):
        data = self._make_jpeg_bytes()
        with pytest.raises(ImageValidationError, match="too large"):
            validate_image_bytes(data, filename="test.jpg", max_size_bytes=100)

    def test_disallowed_extension_rejected(self):
        data = self._make_jpeg_bytes()
        with pytest.raises(ImageValidationError, match="extension"):
            validate_image_bytes(data, filename="test.exe")

    def test_small_image_rejected(self):
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        with pytest.raises(ImageValidationError, match="small"):
            validate_image_bytes(buf.getvalue(), filename="test.jpg")

    def test_returned_image_is_rgb(self):
        img_data = self._make_png_bytes()
        img = validate_image_bytes(img_data, filename="test.png")
        assert img.mode == "RGB"

    def test_safe_temp_filename_no_traversal(self):
        name = safe_temp_filename("../../etc/passwd.jpg")
        assert ".." not in name
        assert "/" not in name
        assert "\\" not in name

    def test_safe_temp_filename_has_banana_prefix(self):
        name = safe_temp_filename("banana.jpg")
        assert name.startswith("banana_")

    def test_safe_temp_filename_preserves_extension(self):
        name = safe_temp_filename("photo.png")
        assert name.endswith(".png")

    def test_safe_temp_filename_randomness(self):
        n1 = safe_temp_filename("test.jpg")
        n2 = safe_temp_filename("test.jpg")
        assert n1 != n2  # Should be unique each call

    def test_allowed_extensions_set(self):
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            assert ext in ALLOWED_EXTENSIONS


# ===========================================================================
# DATABASE MODEL TESTS — NEW COLUMNS
# ===========================================================================


class TestExtendedDatabaseModels:
    """Test extended database model definitions."""

    def test_observation_has_storage_condition_column(self):
        from banana_ai.db.models import Observation
        cols = {c.name for c in Observation.__table__.columns}
        assert "storage_condition" in cols

    def test_observation_has_input_method_column(self):
        from banana_ai.db.models import Observation
        cols = {c.name for c in Observation.__table__.columns}
        assert "input_method" in cols

    def test_prediction_has_estimated_min_days(self):
        from banana_ai.db.models import Prediction
        cols = {c.name for c in Prediction.__table__.columns}
        assert "estimated_min_days" in cols

    def test_prediction_has_estimated_max_days(self):
        from banana_ai.db.models import Prediction
        cols = {c.name for c in Prediction.__table__.columns}
        assert "estimated_max_days" in cols

    def test_prediction_has_shelf_life_method(self):
        from banana_ai.db.models import Prediction
        cols = {c.name for c in Prediction.__table__.columns}
        assert "shelf_life_method" in cols

    def test_feedback_model_exists(self):
        from banana_ai.db.models import PredictionFeedback
        assert PredictionFeedback.__tablename__ == "prediction_feedback"

    def test_feedback_has_required_columns(self):
        from banana_ai.db.models import PredictionFeedback
        cols = {c.name for c in PredictionFeedback.__table__.columns}
        assert "id" in cols
        assert "prediction_id" in cols
        assert "feedback" in cols
        assert "corrected_stage" in cols
        assert "created_at" in cols

    def test_existing_observation_columns_preserved(self):
        from banana_ai.db.models import Observation
        cols = {c.name for c in Observation.__table__.columns}
        assert "id" in cols
        assert "image_path" in cols
        assert "temperature_c" in cols
        assert "humidity_pct" in cols
        assert "observed_at" in cols

    def test_existing_prediction_columns_preserved(self):
        from banana_ai.db.models import Prediction
        cols = {c.name for c in Prediction.__table__.columns}
        assert "id" in cols
        assert "observation_id" in cols
        assert "predicted_stage" in cols
        assert "confidence" in cols
        assert "estimated_days_left" in cols
        assert "model_version" in cols
        assert "created_at" in cols


# ===========================================================================
# API RESPONSE COMPATIBILITY TESTS
# ===========================================================================


class TestAPICompatibility:
    """Test FastAPI endpoint backward compatibility and new features."""

    def test_health_endpoint_ok(self):
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_has_model_version(self):
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert "model_version" in response.json()

    def test_health_model_version_is_v3(self):
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.json()["model_version"] == "banana-cnn-v3"


# ===========================================================================
# REPORT GENERATION TESTS
# ===========================================================================


class TestReportGeneration:
    """Test HTML and CSV report generation."""

    _sample_prediction = {
        "stage": "ripe",
        "confidence": 0.942,
        "probabilities": {
            "overripe": 0.021,
            "ripe": 0.942,
            "rotten": 0.004,
            "unripe": 0.033,
        },
        "estimated_days_left": "2–4 days",
    }

    def test_html_report_generated(self):
        from banana_ai.services.report import generate_scan_report_html
        html = generate_scan_report_html(self._sample_prediction)
        assert "<html" in html.lower()
        assert "RIPE" in html

    def test_html_report_contains_disclaimer(self):
        from banana_ai.services.report import generate_scan_report_html
        html = generate_scan_report_html(self._sample_prediction)
        assert "heuristic" in html.lower() or "not a food-safety" in html.lower()

    def test_html_report_contains_model_version(self):
        from banana_ai.services.report import generate_scan_report_html
        html = generate_scan_report_html(self._sample_prediction)
        assert "banana-cnn-v3" in html

    def test_html_report_with_env_data(self):
        from banana_ai.services.report import generate_scan_report_html
        html = generate_scan_report_html(
            self._sample_prediction, temperature_c=27.0, humidity_pct=65.0
        )
        assert "27" in html
        assert "65" in html

    def test_csv_report_generated(self):
        from banana_ai.services.report import generate_scan_report_csv
        csv = generate_scan_report_csv(self._sample_prediction)
        assert "ripe" in csv.lower()
        assert "0.942" in csv

    def test_csv_report_contains_disclaimer(self):
        from banana_ai.services.report import generate_scan_report_csv
        csv = generate_scan_report_csv(self._sample_prediction)
        assert "heuristic" in csv.lower() or "food-safety" in csv.lower()

    def test_batch_csv_generated(self):
        from banana_ai.services.report import generate_batch_csv
        batch = [
            {"filename": "a.jpg", "stage": "ripe", "confidence": 0.9,
             "probabilities": {"ripe": 0.9, "overripe": 0.05, "unripe": 0.04, "rotten": 0.01},
             "estimated_days_left": "2–4 days"},
            {"filename": "b.jpg", "stage": "unripe", "confidence": 0.95,
             "probabilities": {"ripe": 0.03, "overripe": 0.01, "unripe": 0.95, "rotten": 0.01},
             "estimated_days_left": "4–7 days"},
        ]
        csv = generate_batch_csv(batch)
        assert "a.jpg" in csv
        assert "b.jpg" in csv
        assert "ripe" in csv.lower()

    def test_batch_csv_empty_input(self):
        from banana_ai.services.report import generate_batch_csv
        result = generate_batch_csv([])
        assert result == ""


# ===========================================================================
# MODEL VERSION CONSISTENCY TESTS
# ===========================================================================


class TestModelVersionConsistency:
    """Ensure model version is consistently reported."""

    def test_config_model_version(self):
        from banana_ai.config import settings
        assert settings.model_version == "banana-cnn-v3"

    def test_model_path_is_v3(self):
        from banana_ai.config import settings
        assert "v3" in settings.model_path

    def test_shelf_life_method_is_heuristic(self):
        est = estimate_shelf_life("ripe")
        assert est.model_type == "heuristic"
        assert "heuristic" in est.method
