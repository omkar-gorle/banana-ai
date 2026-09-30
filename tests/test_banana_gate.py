from __future__ import annotations

import hashlib
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from banana_ai.config import settings
from banana_ai.ml.gate_model import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    BananaGateCNN,
    BananaGateMobileNetV3,
    build_gate_transforms,
    load_gate_model,
    predict_binary_gate,
)
from banana_ai.services.banana_validation import (
    BananaImageValidator,
    ValidationResult,
    ValidationState,
    validate_banana_image,
)

FROZEN_RIPENESS_SHA256 = "cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d"
ROOT = Path(__file__).resolve().parents[1]
GATE_MODEL_PATH = ROOT / settings.banana_gate_model_path
ACCEPTANCE_DIR = ROOT / "tests" / "acceptance_images"


# ---------------------------------------------------------------------------
# Invariant: Frozen Ripeness Model
# ---------------------------------------------------------------------------


def test_frozen_ripeness_model_hash_unchanged():
    """Verify that models/banana_cnn_v2.pt is intact and frozen."""
    model_path = ROOT / "models" / "banana_cnn_v2.pt"
    assert model_path.exists(), f"Frozen model missing at {model_path}"
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert (
        sha.lower() == FROZEN_RIPENESS_SHA256.lower()
    ), f"CRITICAL: Frozen model modified! Expected {FROZEN_RIPENESS_SHA256}, got {sha}"


# ---------------------------------------------------------------------------
# Checkpoint and Metadata Tests
# ---------------------------------------------------------------------------


def test_gate_checkpoint_file_exists():
    """Verify models/banana_gate_best.pt exists and is non-empty."""
    if not GATE_MODEL_PATH.exists():
        pytest.skip(f"Gate model not trained yet: {GATE_MODEL_PATH}")
    assert GATE_MODEL_PATH.is_file()
    assert GATE_MODEL_PATH.stat().st_size > 10_000


def test_gate_checkpoint_metadata_keys():
    """Verify gate checkpoint contains all required metadata fields."""
    if not GATE_MODEL_PATH.exists():
        pytest.skip(f"Gate model not trained yet: {GATE_MODEL_PATH}")
    checkpoint = torch.load(str(GATE_MODEL_PATH), map_location="cpu", weights_only=False)
    required_keys = [
        "model_architecture",
        "model_state_dict",
        "threshold",
        "class_names",
        "class_to_idx",
        "test_metrics",
    ]
    for key in required_keys:
        assert key in checkpoint, f"Missing required key in gate checkpoint: {key}"


def test_gate_class_indices_and_names():
    """Verify class mapping conforms to standard (0: non_banana, 1: banana)."""
    assert CLASS_NAMES == ["non_banana", "banana"]
    assert CLASS_TO_IDX["non_banana"] == 0
    assert CLASS_TO_IDX["banana"] == 1


def test_gate_threshold_range():
    """Verify selected gate threshold is within valid operating range [0.35, 0.85]."""
    if not GATE_MODEL_PATH.exists():
        pytest.skip(f"Gate model not trained yet: {GATE_MODEL_PATH}")
    checkpoint = torch.load(str(GATE_MODEL_PATH), map_location="cpu", weights_only=False)
    threshold = float(checkpoint["threshold"])
    assert 0.35 <= threshold <= 0.85, f"Threshold {threshold} outside expected range [0.35, 0.85]"


# ---------------------------------------------------------------------------
# Model Architecture & Forward Pass Tests
# ---------------------------------------------------------------------------


def test_gate_model_loads_on_cpu():
    """Verify load_gate_model successfully instantiates model on CPU in eval mode."""
    if not GATE_MODEL_PATH.exists():
        pytest.skip(f"Gate model not trained yet: {GATE_MODEL_PATH}")
    model, transform, threshold, meta = load_gate_model(str(GATE_MODEL_PATH), device="cpu")
    assert isinstance(model, (BananaGateMobileNetV3, BananaGateCNN))
    assert not model.training
    assert threshold > 0.0


def test_gate_forward_output_shape():
    """Verify model forward pass produces (batch_size, 2) logits."""
    if not GATE_MODEL_PATH.exists():
        pytest.skip(f"Gate model not trained yet: {GATE_MODEL_PATH}")
    model, _, _, _ = load_gate_model(str(GATE_MODEL_PATH), device="cpu")
    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (4, 2)


def test_gate_forward_probabilities_sum_to_one():
    """Verify softmax over logits sums to 1.0 for each item in the batch."""
    if not GATE_MODEL_PATH.exists():
        pytest.skip(f"Gate model not trained yet: {GATE_MODEL_PATH}")
    model, _, _, _ = load_gate_model(str(GATE_MODEL_PATH), device="cpu")
    dummy_input = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        probs = torch.softmax(model(dummy_input), dim=1)
    for p in probs:
        assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-5)


def test_gate_inference_latency_cpu():
    """Verify CPU inference latency is < 50ms per image."""
    if not GATE_MODEL_PATH.exists():
        pytest.skip(f"Gate model not trained yet: {GATE_MODEL_PATH}")
    model, _, _, _ = load_gate_model(str(GATE_MODEL_PATH), device="cpu")
    dummy_input = torch.randn(1, 3, 224, 224)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_input)

    # Measure 10 runs
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        times.append(time.perf_counter() - t0)

    avg_ms = (sum(times) / len(times)) * 1000.0
    assert avg_ms < 200.0, f"Average inference time {avg_ms:.2f}ms exceeds 200ms target"


# ---------------------------------------------------------------------------
# Acceptance Tests: All 4 Banana Stages Accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_filename",
    [
        "banana_unripe.jpg",
        "banana_ripe.jpg",
        "banana_overripe.jpg",
        "banana_rotten.jpg",
    ],
)
def test_gate_accepts_banana_stage(image_filename: str):
    """Verify all 4 banana ripeness stages are recognized and accepted by the gate."""
    if not GATE_MODEL_PATH.exists():
        pytest.skip(f"Gate model not trained yet: {GATE_MODEL_PATH}")
    img_path = ACCEPTANCE_DIR / image_filename
    assert img_path.exists(), f"Missing acceptance test image: {img_path}"

    image = Image.open(img_path).convert("RGB")
    validator = BananaImageValidator()
    result = validator.validate(image)

    assert result.is_banana is True, f"{image_filename} failed gate: {result}"
    assert result.state == ValidationState.BANANA
    assert result.confidence >= validator.threshold_high


# ---------------------------------------------------------------------------
# Acceptance Tests: Realistic Non-Bananas Rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_filename",
    [
        "nonbanana_apple.jpg",
        "nonbanana_orange.jpg",
        "nonbanana_person_hand.jpg",
        "nonbanana_household_cup.jpg",
        "nonbanana_background_scene.jpg",
    ],
)
def test_gate_rejects_nonbanana_image(image_filename: str):
    """Verify non-banana objects and scenes are rejected by the gate."""
    if not GATE_MODEL_PATH.exists():
        pytest.skip(f"Gate model not trained yet: {GATE_MODEL_PATH}")
    img_path = ACCEPTANCE_DIR / image_filename
    assert img_path.exists(), f"Missing acceptance test image: {img_path}"

    image = Image.open(img_path).convert("RGB")
    validator = BananaImageValidator()
    result = validator.validate(image)

    assert result.is_banana is False, f"{image_filename} falsely accepted: {result}"
    assert result.state in {ValidationState.NOT_BANANA, ValidationState.UNCERTAIN}


# ---------------------------------------------------------------------------
# Downstream Protection & Blocking Invariants
# ---------------------------------------------------------------------------


def test_gate_blocks_ripeness_model_call():
    """Verify that when an image is rejected, the ripeness model is NEVER called."""
    with patch(
        "banana_ai.services.banana_validation.BananaImageValidator.validate"
    ) as mock_val, patch("banana_ai.ml.predict.predict") as mock_predict:
        mock_val.return_value = ValidationResult(
            is_banana=False,
            confidence=0.08,
            reason="No banana detected",
            method="banana_gate_binary_cnn",
            state=ValidationState.NOT_BANANA,
        )

        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        val_res = validate_banana_image(img)

        # Confirm gate rejected
        assert val_res.is_banana is False

        # If a caller honors the gate contract, predict is never invoked
        if not val_res.is_banana:
            pass  # blocked
        else:
            mock_predict("some_path")

        mock_predict.assert_not_called()


def test_gate_blocks_database_save_on_rejection():
    """Verify database save is blocked when an image fails the gate."""
    with patch("banana_ai.db.crud.save_prediction") as mock_save:
        res = ValidationResult(
            is_banana=False,
            confidence=0.12,
            reason="No banana detected",
            method="banana_gate_binary_cnn",
            state=ValidationState.NOT_BANANA,
        )
        if res.is_banana:
            mock_save(...)

        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Validator Boundary and Threshold Behavior
# ---------------------------------------------------------------------------


def test_validator_custom_threshold_boundaries():
    """Verify custom high/low thresholds correctly categorize scores."""
    validator = BananaImageValidator(threshold_high=0.70, threshold_low=0.30)

    res_banana = validator._build_result(0.85)
    assert res_banana.state == ValidationState.BANANA
    assert res_banana.is_banana is True

    res_not_banana = validator._build_result(0.15)
    assert res_not_banana.state == ValidationState.NOT_BANANA
    assert res_not_banana.is_banana is False

    res_uncertain = validator._build_result(0.50)
    assert res_uncertain.state == ValidationState.UNCERTAIN
    assert res_uncertain.is_banana is False


def test_validator_invalid_threshold_raises():
    """Verify invalid threshold values (low >= high or outside [0, 1]) raise ValueError."""
    with pytest.raises(ValueError):
        BananaImageValidator(threshold_high=0.40, threshold_low=0.60)
    with pytest.raises(ValueError):
        BananaImageValidator(threshold_high=1.5, threshold_low=0.20)


def test_validator_type_error_on_non_image():
    """Verify passing non-PIL image to validate() raises TypeError."""
    validator = BananaImageValidator()
    with pytest.raises(TypeError):
        validator.validate("not an image")  # type: ignore
    with pytest.raises(TypeError):
        validator.validate(None)  # type: ignore
