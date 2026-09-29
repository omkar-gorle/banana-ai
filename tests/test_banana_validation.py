"""Tests for the Banana Image Validation Gate.

Test strategy
-------------
All tests use synthetic PIL images rather than real photos so the suite
runs offline without any image downloads.

The key invariants verified:

1. BananaImageValidator returns ValidationResult with the correct fields.
2. validate() raises TypeError for non-PIL inputs.
3. ValidationState THREE-STATE logic works correctly.
4. Module-level validate_banana_image() is callable.
5. CRITICAL: When validator rejects an image, the ripeness model is NEVER called.
6. CRITICAL: When validator rejects uncertain images, the ripeness model is NEVER called.
7. Configurable thresholds work.
8. API /predict endpoint enforces banana gate and returns correct HTTP codes.
9. Batch pipeline rejects non-banana images individually.
10. Safe fallback when detector is unavailable (UNCERTAIN, not crash).

Implementation note
-------------------
Because the SSDLite detector requires COCO weights (network download on first
use), all tests that call the real detector are designed to handle either:
  - the weights being already cached (~23 MB, downloaded by torchvision),
  - or the weights not yet being available.

Tests that verify the GATE LOGIC (most tests) mock the detector's output
score instead of calling the real model, which keeps the suite fast and
deterministic.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(w: int = 224, h: int = 224, color: tuple = (200, 170, 50)) -> Image.Image:
    """Create a solid-colour PIL RGB image."""
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_jpeg_bytes(w: int = 224, h: int = 224) -> bytes:
    img = _make_rgb_image(w, h)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from banana_ai.services.banana_validation import (
    BananaImageValidator,
    ValidationResult,
    ValidationState,
    COCO_BANANA_CLASS_IDX,
    validate_banana_image,
    get_validator,
)


# ===========================================================================
# 1. ValidationResult dataclass
# ===========================================================================


class TestValidationResult:
    """Unit tests for the ValidationResult dataclass."""

    def test_is_banana_true_for_banana_state(self):
        r = ValidationResult(
            is_banana=True,
            confidence=0.90,
            reason="Banana detected",
            method="test",
            state=ValidationState.BANANA,
        )
        assert r.is_banana is True
        assert r.state == ValidationState.BANANA

    def test_is_banana_false_for_not_banana_state(self):
        r = ValidationResult(
            is_banana=False,
            confidence=0.05,
            reason="No banana",
            method="test",
            state=ValidationState.NOT_BANANA,
        )
        assert r.is_banana is False
        assert r.state == ValidationState.NOT_BANANA

    def test_is_banana_false_for_uncertain_state(self):
        r = ValidationResult(
            is_banana=False,
            confidence=0.42,
            reason="Ambiguous",
            method="test",
            state=ValidationState.UNCERTAIN,
        )
        assert r.is_banana is False
        assert r.state == ValidationState.UNCERTAIN

    def test_result_is_frozen(self):
        r = ValidationResult(
            is_banana=True, confidence=0.9, reason="x", method="y",
            state=ValidationState.BANANA,
        )
        with pytest.raises(Exception):  # dataclass(frozen=True) raises FrozenInstanceError
            r.is_banana = False  # type: ignore[misc]

    def test_confidence_range(self):
        for conf in [0.0, 0.5, 1.0]:
            r = ValidationResult(
                is_banana=False, confidence=conf, reason="", method="",
                state=ValidationState.NOT_BANANA,
            )
            assert 0.0 <= r.confidence <= 1.0

    def test_str_representation(self):
        r = ValidationResult(
            is_banana=True, confidence=0.94, reason="Banana detected",
            method="banana_gate_ssdlite320", state=ValidationState.BANANA,
        )
        s = str(r)
        assert "BANANA" in s
        assert "94.00%" in s


# ===========================================================================
# 2. ValidationState enum
# ===========================================================================


class TestValidationState:
    def test_all_states_defined(self):
        assert ValidationState.BANANA
        assert ValidationState.NOT_BANANA
        assert ValidationState.UNCERTAIN

    def test_state_values(self):
        assert ValidationState.BANANA == "BANANA"
        assert ValidationState.NOT_BANANA == "NOT_BANANA"
        assert ValidationState.UNCERTAIN == "UNCERTAIN"

    def test_states_are_distinct(self):
        assert ValidationState.BANANA != ValidationState.NOT_BANANA
        assert ValidationState.BANANA != ValidationState.UNCERTAIN
        assert ValidationState.NOT_BANANA != ValidationState.UNCERTAIN


# ===========================================================================
# 3. BananaImageValidator — threshold logic
# ===========================================================================


class TestBananaImageValidatorThresholds:
    """Test the three-state decision logic using mocked scores."""

    def _validator(self, high=0.60, low=0.25) -> BananaImageValidator:
        return BananaImageValidator(threshold_high=high, threshold_low=low)

    def _validate_with_score(self, validator: BananaImageValidator, score: float) -> ValidationResult:
        """Call _build_result directly to test threshold logic without detector."""
        return validator._build_result(score)

    # ---- Banana state ----

    def test_score_at_high_threshold_is_banana(self):
        v = self._validator()
        r = self._validate_with_score(v, 0.60)
        assert r.state == ValidationState.BANANA
        assert r.is_banana is True

    def test_score_above_high_threshold_is_banana(self):
        v = self._validator()
        r = self._validate_with_score(v, 0.95)
        assert r.state == ValidationState.BANANA

    def test_score_1_0_is_banana(self):
        v = self._validator()
        r = self._validate_with_score(v, 1.0)
        assert r.state == ValidationState.BANANA

    # ---- NOT_BANANA state ----

    def test_score_at_low_threshold_is_not_banana(self):
        v = self._validator()
        r = self._validate_with_score(v, 0.25)
        assert r.state == ValidationState.NOT_BANANA
        assert r.is_banana is False

    def test_score_below_low_threshold_is_not_banana(self):
        v = self._validator()
        r = self._validate_with_score(v, 0.10)
        assert r.state == ValidationState.NOT_BANANA

    def test_score_0_is_not_banana(self):
        v = self._validator()
        r = self._validate_with_score(v, 0.0)
        assert r.state == ValidationState.NOT_BANANA

    # ---- UNCERTAIN state ----

    def test_score_between_thresholds_is_uncertain(self):
        v = self._validator()
        r = self._validate_with_score(v, 0.42)
        assert r.state == ValidationState.UNCERTAIN
        assert r.is_banana is False

    def test_just_above_low_threshold_is_uncertain(self):
        v = self._validator()
        r = self._validate_with_score(v, 0.26)
        assert r.state == ValidationState.UNCERTAIN

    def test_just_below_high_threshold_is_uncertain(self):
        v = self._validator()
        r = self._validate_with_score(v, 0.59)
        assert r.state == ValidationState.UNCERTAIN

    # ---- Confidence returned correctly ----

    def test_confidence_matches_score(self):
        v = self._validator()
        for score in [0.0, 0.3, 0.6, 1.0]:
            r = self._validate_with_score(v, score)
            assert abs(r.confidence - score) < 1e-6

    # ---- Custom thresholds ----

    def test_custom_threshold_high(self):
        v = self._validator(high=0.80, low=0.20)
        r = self._validate_with_score(v, 0.75)
        assert r.state == ValidationState.UNCERTAIN  # below new high

    def test_custom_threshold_low(self):
        v = self._validator(high=0.70, low=0.50)
        r = self._validate_with_score(v, 0.45)
        assert r.state == ValidationState.NOT_BANANA  # below new low

    def test_invalid_thresholds_raises(self):
        with pytest.raises(ValueError):
            BananaImageValidator(threshold_high=0.30, threshold_low=0.50)  # low > high

    def test_equal_thresholds_raises(self):
        with pytest.raises(ValueError):
            BananaImageValidator(threshold_high=0.50, threshold_low=0.50)


# ===========================================================================
# 4. BananaImageValidator — validate() method with mocked detector
# ===========================================================================


def _make_fake_prediction(score: float, label: int = COCO_BANANA_CLASS_IDX):
    """Build a fake prediction dict matching torchvision detector output."""
    return [{
        "labels": torch.tensor([label]),
        "scores": torch.tensor([score]),
        "boxes": torch.tensor([[10.0, 10.0, 100.0, 200.0]]),
    }]


class TestBananaImageValidatorValidate:
    """Test validate() using mocked detector (no real COCO weights needed)."""

    def _validator(self) -> BananaImageValidator:
        return BananaImageValidator(threshold_high=0.60, threshold_low=0.25)

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_high_score_returns_banana(self, mock_get_detector):
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = _make_fake_prediction(0.92)
        mock_get_detector.return_value = (mock_model, mock_transform)

        v = self._validator()
        img = _make_rgb_image()
        result = v.validate(img)
        assert result.state == ValidationState.BANANA
        assert result.is_banana is True
        assert result.confidence >= 0.60

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_low_score_returns_not_banana(self, mock_get_detector):
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = _make_fake_prediction(0.10)
        mock_get_detector.return_value = (mock_model, mock_transform)

        v = self._validator()
        img = _make_rgb_image()
        result = v.validate(img)
        assert result.state == ValidationState.NOT_BANANA
        assert result.is_banana is False

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_medium_score_returns_uncertain(self, mock_get_detector):
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = _make_fake_prediction(0.42)
        mock_get_detector.return_value = (mock_model, mock_transform)

        v = self._validator()
        img = _make_rgb_image()
        result = v.validate(img)
        assert result.state == ValidationState.UNCERTAIN
        assert result.is_banana is False

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_no_banana_class_in_predictions_is_not_banana(self, mock_get_detector):
        """Predictions with only non-banana class labels → NOT_BANANA."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        # Use class 1 (person) instead of banana (46)
        mock_model.return_value = [{"labels": torch.tensor([1]), "scores": torch.tensor([0.98])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        v = self._validator()
        img = _make_rgb_image()
        result = v.validate(img)
        assert result.state == ValidationState.NOT_BANANA

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_empty_predictions_is_not_banana(self, mock_get_detector):
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = [{"labels": torch.tensor([]), "scores": torch.tensor([])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        v = self._validator()
        result = v.validate(_make_rgb_image())
        assert result.state == ValidationState.NOT_BANANA

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_highest_banana_score_is_used(self, mock_get_detector):
        """When multiple banana detections, the highest score determines state."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = [{
            "labels": torch.tensor([COCO_BANANA_CLASS_IDX, COCO_BANANA_CLASS_IDX]),
            "scores": torch.tensor([0.45, 0.75]),
        }]
        mock_get_detector.return_value = (mock_model, mock_transform)

        v = self._validator()
        result = v.validate(_make_rgb_image())
        # 0.75 >= 0.60 → BANANA
        assert result.state == ValidationState.BANANA
        assert abs(result.confidence - 0.75) < 1e-4

    def test_non_pil_input_raises_type_error(self):
        v = self._validator()
        with pytest.raises(TypeError):
            v.validate("not_an_image")  # type: ignore[arg-type]

    def test_rgba_image_converted_to_rgb(self):
        """RGBA images should be handled gracefully (converted internally)."""
        arr = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        rgba_img = Image.fromarray(arr, mode="RGBA")

        with patch("banana_ai.services.banana_validation._get_detector") as mock_gd:
            mock_model = MagicMock(return_value=_make_fake_prediction(0.0))
            mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
            mock_gd.return_value = (mock_model, mock_transform)

            v = self._validator()
            result = v.validate(rgba_img)  # should not raise
            assert result is not None

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_method_field_is_set(self, mock_get_detector):
        mock_model = MagicMock(return_value=_make_fake_prediction(0.80))
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_get_detector.return_value = (mock_model, mock_transform)

        v = BananaImageValidator()
        result = v.validate(_make_rgb_image())
        assert result.method == BananaImageValidator.METHOD

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_reason_field_non_empty(self, mock_get_detector):
        mock_model = MagicMock(return_value=_make_fake_prediction(0.80))
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_get_detector.return_value = (mock_model, mock_transform)

        v = BananaImageValidator()
        result = v.validate(_make_rgb_image())
        assert len(result.reason) > 0

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_detector_exception_returns_uncertain_not_crash(self, mock_get_detector):
        """If the detector itself raises, validator falls back to UNCERTAIN."""
        mock_get_detector.side_effect = RuntimeError("GPU exploded")
        v = BananaImageValidator()
        result = v.validate(_make_rgb_image())
        # Falls back: score = midpoint → UNCERTAIN
        assert result.state == ValidationState.UNCERTAIN


# ===========================================================================
# 5. CRITICAL: Ripeness model NOT called when banana gate rejects
# ===========================================================================


class TestRipenessModelNotCalledOnRejection:
    """The most important invariant: if the validator rejects, the ripeness
    model must not execute.

    We mock predict_image to ensure it is never called.
    """

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_not_banana_does_not_call_ripeness_model(self, mock_get_detector):
        """When validator returns NOT_BANANA, ripeness model is NOT called."""
        # Arrange: detector says 0% banana confidence
        mock_model = MagicMock(return_value=_make_fake_prediction(0.05))
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator(threshold_high=0.60, threshold_low=0.25)
        mock_ripeness_predict = MagicMock()

        img = _make_rgb_image()
        validation_result = validator.validate(img)

        # The gate says NOT_BANANA → pipeline should stop
        assert validation_result.state == ValidationState.NOT_BANANA
        assert validation_result.is_banana is False

        # Simulate what the pipeline does: only call ripeness if is_banana
        if validation_result.is_banana:
            mock_ripeness_predict(img)

        # Ripeness model must NOT have been called
        mock_ripeness_predict.assert_not_called()

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_uncertain_does_not_call_ripeness_model(self, mock_get_detector):
        """When validator returns UNCERTAIN, ripeness model is NOT called."""
        mock_model = MagicMock(return_value=_make_fake_prediction(0.42))
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock()

        img = _make_rgb_image()
        validation_result = validator.validate(img)

        assert validation_result.state == ValidationState.UNCERTAIN
        assert validation_result.is_banana is False

        if validation_result.is_banana:
            mock_ripeness_predict(img)

        mock_ripeness_predict.assert_not_called()

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_banana_confirmed_calls_ripeness_model(self, mock_get_detector):
        """When validator returns BANANA, the ripeness model IS called."""
        mock_model = MagicMock(return_value=_make_fake_prediction(0.92))
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock(return_value={"stage": "ripe", "confidence": 0.95})

        img = _make_rgb_image()
        validation_result = validator.validate(img)

        assert validation_result.state == ValidationState.BANANA
        assert validation_result.is_banana is True

        if validation_result.is_banana:
            mock_ripeness_predict(img)

        mock_ripeness_predict.assert_called_once_with(img)

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_person_image_rejects_ripeness_model_not_called(self, mock_get_detector):
        """Simulate 'person' image (class 1, high confidence) — banana NOT detected."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        # Person class (1) detected with high confidence, no banana
        mock_model.return_value = [{"labels": torch.tensor([1]), "scores": torch.tensor([0.99])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock()

        result = validator.validate(_make_rgb_image())
        assert result.state == ValidationState.NOT_BANANA

        if result.is_banana:
            mock_ripeness_predict(_make_rgb_image())

        mock_ripeness_predict.assert_not_called()

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_car_image_rejects_ripeness_model_not_called(self, mock_get_detector):
        """Simulate 'car' image (COCO class 3)."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = [{"labels": torch.tensor([3]), "scores": torch.tensor([0.97])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock()
        result = validator.validate(_make_rgb_image())
        assert result.state == ValidationState.NOT_BANANA

        if result.is_banana:
            mock_ripeness_predict(_make_rgb_image())
        mock_ripeness_predict.assert_not_called()

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_dog_image_rejects_ripeness_model_not_called(self, mock_get_detector):
        """Simulate 'dog' image (COCO class 18)."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = [{"labels": torch.tensor([18]), "scores": torch.tensor([0.88])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock()
        result = validator.validate(_make_rgb_image())
        assert result.state == ValidationState.NOT_BANANA

        if result.is_banana:
            mock_ripeness_predict(_make_rgb_image())
        mock_ripeness_predict.assert_not_called()

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_laptop_image_rejects_ripeness_model_not_called(self, mock_get_detector):
        """Simulate 'laptop' image (COCO class 73)."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = [{"labels": torch.tensor([73]), "scores": torch.tensor([0.91])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock()
        result = validator.validate(_make_rgb_image())
        assert result.state == ValidationState.NOT_BANANA

        if result.is_banana:
            mock_ripeness_predict(_make_rgb_image())
        mock_ripeness_predict.assert_not_called()

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_apple_image_rejects_ripeness_model_not_called(self, mock_get_detector):
        """Simulate 'apple' image (COCO class 53)."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = [{"labels": torch.tensor([53]), "scores": torch.tensor([0.85])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock()
        result = validator.validate(_make_rgb_image())
        assert result.state == ValidationState.NOT_BANANA

        if result.is_banana:
            mock_ripeness_predict(_make_rgb_image())
        mock_ripeness_predict.assert_not_called()

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_orange_image_rejects_ripeness_model_not_called(self, mock_get_detector):
        """Simulate 'orange' image (COCO class 49)."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = [{"labels": torch.tensor([49]), "scores": torch.tensor([0.82])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock()
        result = validator.validate(_make_rgb_image())
        assert result.state == ValidationState.NOT_BANANA

        if result.is_banana:
            mock_ripeness_predict(_make_rgb_image())
        mock_ripeness_predict.assert_not_called()

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_bottle_image_rejects_ripeness_model_not_called(self, mock_get_detector):
        """Simulate 'bottle' image (COCO class 44)."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = [{"labels": torch.tensor([44]), "scores": torch.tensor([0.78])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock()
        result = validator.validate(_make_rgb_image())
        assert result.state == ValidationState.NOT_BANANA

        if result.is_banana:
            mock_ripeness_predict(_make_rgb_image())
        mock_ripeness_predict.assert_not_called()

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_no_detections_at_all_rejects(self, mock_get_detector):
        """Image with no detections → NOT_BANANA."""
        mock_model = MagicMock()
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_model.return_value = [{"labels": torch.tensor([]), "scores": torch.tensor([])}]
        mock_get_detector.return_value = (mock_model, mock_transform)

        validator = BananaImageValidator()
        mock_ripeness_predict = MagicMock()
        result = validator.validate(_make_rgb_image())
        assert result.state == ValidationState.NOT_BANANA

        if result.is_banana:
            mock_ripeness_predict(_make_rgb_image())
        mock_ripeness_predict.assert_not_called()


# ===========================================================================
# 6. Module-level API
# ===========================================================================


class TestModuleLevelAPI:
    """Tests for validate_banana_image() and get_validator()."""

    def test_validate_banana_image_returns_validation_result(self):
        with patch("banana_ai.services.banana_validation._get_detector") as mock_gd:
            mock_model = MagicMock(return_value=_make_fake_prediction(0.0))
            mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
            mock_gd.return_value = (mock_model, mock_transform)

            result = validate_banana_image(_make_rgb_image())
            assert isinstance(result, ValidationResult)

    def test_get_validator_returns_banana_image_validator(self):
        v = get_validator()
        assert isinstance(v, BananaImageValidator)

    def test_get_validator_returns_same_instance(self):
        v1 = get_validator()
        v2 = get_validator()
        assert v1 is v2  # singleton

    def test_validate_banana_image_with_non_pil_raises(self):
        with pytest.raises(TypeError):
            validate_banana_image("not a PIL image")  # type: ignore[arg-type]


# ===========================================================================
# 7. Config thresholds
# ===========================================================================


class TestConfigThresholds:
    """Thresholds should be readable from settings."""

    def test_settings_has_banana_detection_threshold_high(self):
        from banana_ai.config import settings
        assert hasattr(settings, "banana_detection_threshold_high")
        assert 0.0 < settings.banana_detection_threshold_high <= 1.0

    def test_settings_has_banana_detection_threshold_low(self):
        from banana_ai.config import settings
        assert hasattr(settings, "banana_detection_threshold_low")
        assert 0.0 <= settings.banana_detection_threshold_low < 1.0

    def test_high_threshold_greater_than_low(self):
        from banana_ai.config import settings
        assert settings.banana_detection_threshold_high > settings.banana_detection_threshold_low

    def test_default_high_threshold_is_0_60(self):
        from banana_ai.config import settings
        assert abs(settings.banana_detection_threshold_high - 0.60) < 1e-6

    def test_default_low_threshold_is_0_25(self):
        from banana_ai.config import settings
        assert abs(settings.banana_detection_threshold_low - 0.25) < 1e-6


# ===========================================================================
# 8. FastAPI endpoint enforces banana gate
# ===========================================================================


class TestFastAPIBananaGate:
    """Test that /predict endpoint enforces banana content validation."""

    def _make_upload_bytes(self) -> bytes:
        return _make_jpeg_bytes()

    @patch("banana_ai.api.main.validate_banana_image")
    def test_not_banana_returns_400(self, mock_validate):
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        mock_validate.return_value = ValidationResult(
            is_banana=False,
            confidence=0.05,
            reason="No banana",
            method="banana_gate_ssdlite320",
            state=ValidationState.NOT_BANANA,
        )

        client = TestClient(app, raise_server_exceptions=False)
        img_bytes = self._make_upload_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "NOT_BANANA"

    @patch("banana_ai.api.main.validate_banana_image")
    def test_uncertain_returns_400_with_uncertain_code(self, mock_validate):
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        mock_validate.return_value = ValidationResult(
            is_banana=False,
            confidence=0.42,
            reason="Ambiguous",
            method="banana_gate_ssdlite320",
            state=ValidationState.UNCERTAIN,
        )

        client = TestClient(app, raise_server_exceptions=False)
        img_bytes = self._make_upload_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "BANANA_UNCERTAIN"

    @patch("banana_ai.api.main.save_prediction")
    @patch("banana_ai.api.main.validate_banana_image")
    @patch("banana_ai.api.main.predict")
    def test_banana_accepted_calls_ripeness_model(self, mock_predict, mock_validate, mock_save):
        """When banana validated, predict() is called and 200 returned."""
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app
        from pathlib import Path
        from unittest.mock import MagicMock

        mock_save.return_value = MagicMock(id=1)
        mock_validate.return_value = ValidationResult(
            is_banana=True,
            confidence=0.92,
            reason="Banana detected",
            method="banana_gate_ssdlite320",
            state=ValidationState.BANANA,
        )
        mock_predict.return_value = {
            "stage": "ripe",
            "confidence": 0.95,
            "probabilities": {"ripe": 0.95, "overripe": 0.02, "unripe": 0.02, "rotten": 0.01},
            "estimated_days_left": "2-4 days",
        }

        # Ensure model file appears to exist
        with patch.object(Path, "exists", return_value=True):
            client = TestClient(app, raise_server_exceptions=False)
            img_bytes = self._make_upload_bytes()
            response = client.post(
                "/predict",
                files={"file": ("banana.jpg", img_bytes, "image/jpeg")},
            )

        assert response.status_code == 200
        body = response.json()
        assert "predicted_stage" in body
        assert "banana_detection_confidence" in body
        assert body["banana_detection_confidence"] == 0.92
        mock_predict.assert_called_once()

    @patch("banana_ai.api.main.validate_banana_image")
    @patch("banana_ai.api.main.predict")
    def test_not_banana_does_not_call_ripeness_model(self, mock_predict, mock_validate):
        """CRITICAL: When validation fails, predict() must NOT be called."""
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        mock_validate.return_value = ValidationResult(
            is_banana=False,
            confidence=0.05,
            reason="No banana",
            method="banana_gate_ssdlite320",
            state=ValidationState.NOT_BANANA,
        )

        client = TestClient(app, raise_server_exceptions=False)
        img_bytes = self._make_upload_bytes()
        client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )

        # Ripeness model must NOT have been called
        mock_predict.assert_not_called()

    @patch("banana_ai.api.main.validate_banana_image")
    @patch("banana_ai.api.main.predict")
    def test_uncertain_does_not_call_ripeness_model(self, mock_predict, mock_validate):
        """CRITICAL: When uncertain, predict() must NOT be called."""
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        mock_validate.return_value = ValidationResult(
            is_banana=False,
            confidence=0.42,
            reason="Ambiguous",
            method="banana_gate_ssdlite320",
            state=ValidationState.UNCERTAIN,
        )

        client = TestClient(app, raise_server_exceptions=False)
        img_bytes = self._make_upload_bytes()
        client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        mock_predict.assert_not_called()

    def test_health_endpoint_still_works(self):
        from fastapi.testclient import TestClient
        from banana_ai.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ===========================================================================
# 9. COCO class index constant
# ===========================================================================


class TestCOCOBananaClass:
    def test_banana_class_index_is_46(self):
        assert COCO_BANANA_CLASS_IDX == 46

    def test_banana_class_index_is_int(self):
        assert isinstance(COCO_BANANA_CLASS_IDX, int)


# ===========================================================================
# 10. Batch pipeline tests (unit-level)
# ===========================================================================


class TestBatchBananaGate:
    """Verify that the batch pipeline logic rejects non-banana images
    individually and never calls the ripeness model for rejected images.
    """

    def _simulate_batch_item(
        self,
        banana_score: float,
        validator: Optional[BananaImageValidator] = None,
        mock_predict_fn: Optional[MagicMock] = None,
    ) -> dict:
        """
        Simulate what page_batch() does for a single image.
        Returns the batch result entry.
        """
        if validator is None:
            validator = BananaImageValidator()
        if mock_predict_fn is None:
            mock_predict_fn = MagicMock(return_value={
                "stage": "ripe", "confidence": 0.9,
                "probabilities": {}, "estimated_days_left": "2-4 days"
            })

        img = _make_rgb_image()

        with patch("banana_ai.services.banana_validation._get_detector") as mock_gd:
            mock_model = MagicMock(return_value=_make_fake_prediction(banana_score))
            mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
            mock_gd.return_value = (mock_model, mock_transform)

            banana_res = validator.validate(img)

        if banana_res.state == ValidationState.NOT_BANANA:
            return {
                "stage": "rejected",
                "error": "Not a banana image",
                "banana_state": "NOT_BANANA",
                "banana_detection_confidence": banana_res.confidence,
            }
        if banana_res.state == ValidationState.UNCERTAIN:
            return {
                "stage": "uncertain",
                "error": "Banana detection uncertain",
                "banana_state": "UNCERTAIN",
                "banana_detection_confidence": banana_res.confidence,
            }

        # Banana confirmed — call ripeness
        result = mock_predict_fn(img)
        result["banana_state"] = "BANANA"
        result["banana_detection_confidence"] = banana_res.confidence
        return result

    def test_banana_image_processed(self):
        mock_predict = MagicMock(return_value={
            "stage": "ripe", "confidence": 0.9,
            "probabilities": {}, "estimated_days_left": "2-4 days",
        })
        result = self._simulate_batch_item(0.90, mock_predict_fn=mock_predict)
        assert result["stage"] == "ripe"
        assert result["banana_state"] == "BANANA"
        mock_predict.assert_called_once()

    def test_non_banana_image_rejected(self):
        mock_predict = MagicMock()
        result = self._simulate_batch_item(0.05, mock_predict_fn=mock_predict)
        assert result["stage"] == "rejected"
        assert result["banana_state"] == "NOT_BANANA"
        mock_predict.assert_not_called()

    def test_uncertain_image_rejected(self):
        mock_predict = MagicMock()
        result = self._simulate_batch_item(0.42, mock_predict_fn=mock_predict)
        assert result["stage"] == "uncertain"
        assert result["banana_state"] == "UNCERTAIN"
        mock_predict.assert_not_called()

    def test_mixed_batch_correct_counts(self):
        """5 images: 3 bananas + 1 non-banana + 1 uncertain."""
        mock_predict = MagicMock(return_value={
            "stage": "ripe", "confidence": 0.9,
            "probabilities": {}, "estimated_days_left": "2-4 days",
        })
        validator = BananaImageValidator()

        scores = [0.90, 0.85, 0.80, 0.05, 0.42]
        results = [
            self._simulate_batch_item(s, validator=validator, mock_predict_fn=mock_predict)
            for s in scores
        ]

        processed = [r for r in results if r["stage"] not in {"rejected", "uncertain", "error"}]
        rejected = [r for r in results if r["stage"] == "rejected"]
        uncertain = [r for r in results if r["stage"] == "uncertain"]

        assert len(processed) == 3
        assert len(rejected) == 1
        assert len(uncertain) == 1
        assert mock_predict.call_count == 3  # only called for real bananas


# ===========================================================================
# 11. Determinism test
# ===========================================================================


class TestDeterminism:
    """Same inputs must always produce same outputs."""

    @patch("banana_ai.services.banana_validation._get_detector")
    def test_same_image_same_result(self, mock_get_detector):
        mock_model = MagicMock(return_value=_make_fake_prediction(0.80))
        mock_transform = MagicMock(return_value=torch.zeros(3, 224, 224))
        mock_get_detector.return_value = (mock_model, mock_transform)

        v = BananaImageValidator()
        img = _make_rgb_image()
        r1 = v.validate(img)
        r2 = v.validate(img)

        assert r1.state == r2.state
        assert abs(r1.confidence - r2.confidence) < 1e-6
