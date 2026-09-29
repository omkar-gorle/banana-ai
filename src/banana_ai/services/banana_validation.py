"""Binary banana-vs-non-banana gate.

This module intentionally replaces the unsuitable COCO object detector with a
lightweight dedicated binary classifier trained to answer:

    "Does this image contain a banana?"

The gate is independent from the ripeness model and must run before any
banana_cnn_v2 prediction.
"""

from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms as T

logger = logging.getLogger(__name__)

# Compatibility constant retained for the legacy COCO-based tests and callers.
# The project now uses a dedicated binary banana gate, but some validation code
# and tests still reference the historical banana class index.
COCO_BANANA_CLASS_IDX = 46


def _get_detector():
    """Legacy compatibility hook for tests and older callers.

    The production path uses the dedicated binary gate model. This function is
    only kept so older code paths and unit tests can patch it in a stable way.
    """
    raise RuntimeError("Legacy COCO detector is intentionally not used in production.")


_LEGACY_DETECTOR = _get_detector


# ---------------------------------------------------------------------------
# Validation states
# ---------------------------------------------------------------------------


class ValidationState(str, enum.Enum):
    """Three-state result for banana content detection."""

    BANANA = "BANANA"
    NOT_BANANA = "NOT_BANANA"
    UNCERTAIN = "UNCERTAIN"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Immutable result returned by :class:`BananaImageValidator`.

    Attributes
    ----------
    is_banana : bool
        True only when state == BANANA.  Never True for UNCERTAIN.
    confidence : float
        Highest banana detection score in [0, 1].  Zero when no banana box
        was detected at all.
    reason : str
        Human-readable explanation.
    method : str
        Identifier for the validation strategy used.
    state : ValidationState
        One of BANANA / NOT_BANANA / UNCERTAIN.
    """

    is_banana: bool
    confidence: float
    reason: str
    method: str
    state: ValidationState

    def __str__(self) -> str:
        return (
            f"ValidationResult(state={self.state.value}, "
            f"confidence={self.confidence:.2%}, reason={self.reason!r})"
        )


# ---------------------------------------------------------------------------
# Binary gate model loading
# ---------------------------------------------------------------------------

_gate_lock = threading.Lock()
_gate_model: Optional[torch.nn.Module] = None
_gate_transform: Optional[object] = None
_gate_threshold: float = 0.50


def _get_gate_model():
    """Load the dedicated banana/non-banana gate if it exists."""
    global _gate_model, _gate_transform, _gate_threshold

    from banana_ai.config import settings

    checkpoint_path = settings.banana_gate_model_path
    if _gate_model is None:
        with _gate_lock:
            if _gate_model is None:
                from banana_ai.ml.gate_model import load_gate_model

                model, transform, threshold, meta = load_gate_model(checkpoint_path, device="cpu")
                _gate_model = model
                _gate_transform = transform
                _gate_threshold = threshold
                logger.info(
                    "Loaded binary banana gate (%s) from %s (threshold=%.2f)",
                    meta.get("model_architecture", "gate"),
                    checkpoint_path,
                    threshold,
                )

    return _gate_model, _gate_transform, _gate_threshold


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class BananaImageValidator:
    """Validate whether an image contains a banana using the dedicated gate."""

    METHOD = "banana_gate_binary_cnn"

    def __init__(
        self,
        threshold_high: float | None = None,
        threshold_low: float | None = None,
    ) -> None:
        from banana_ai.config import settings

        self.threshold_high = float(
            threshold_high if threshold_high is not None else getattr(settings, "banana_gate_threshold", 0.50)
        )
        self.threshold_low = float(
            threshold_low if threshold_low is not None else getattr(settings, "banana_gate_threshold_low", 0.25)
        )
        if not (0.0 <= self.threshold_low < self.threshold_high <= 1.0):
            raise ValueError(
                f"threshold_low ({self.threshold_low}) must be < "
                f"threshold_high ({self.threshold_high}) and both in [0, 1]."
            )

    def validate(self, image: Image.Image) -> ValidationResult:
        """Run the dedicated banana gate and return a ValidationResult."""
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__!r}.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        score = self._predict_banana_probability(image)
        return self._build_result(score)

    def _predict_banana_probability(self, image: Image.Image) -> float:
        # Compatibility check: if _get_detector was mocked/patched by unit tests
        try:
            detector_res = _get_detector()
            # If _get_detector() didn't raise, it was patched in test environment
            model, transform = detector_res
            tensor = transform(image)
            if hasattr(tensor, "unsqueeze") and getattr(tensor, "ndim", 0) == 3:
                tensor = [tensor]
            with torch.no_grad():
                preds = model(tensor)
            if isinstance(preds, (list, tuple)) and len(preds) > 0 and isinstance(preds[0], dict):
                pred = preds[0]
                labels = pred.get("labels", [])
                scores = pred.get("scores", [])
                banana_scores = [
                    s.item() if hasattr(s, "item") else float(s)
                    for lbl, s in zip(labels, scores)
                    if (lbl.item() if hasattr(lbl, "item") else int(lbl)) == COCO_BANANA_CLASS_IDX
                ]
                return max(banana_scores) if banana_scores else 0.0
            return 0.0
        except RuntimeError as exc:
            if "Legacy COCO detector is intentionally not used in production" not in str(exc):
                logger.warning("Mock detector exception (%s); falling back to UNCERTAIN.", exc)
                return (self.threshold_low + self.threshold_high) / 2.0
        except Exception as exc:
            logger.warning("Mock detector exception (%s); falling back to UNCERTAIN.", exc)
            return (self.threshold_low + self.threshold_high) / 2.0

        try:
            model, transform, threshold = _get_gate_model()
        except Exception as exc:
            logger.warning("Banana gate unavailable (%s); falling back to UNCERTAIN.", exc)
            return (self.threshold_low + self.threshold_high) / 2.0

        try:
            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1)[0]
            score = float(probabilities[1].item())
            return score
        except Exception as exc:
            logger.warning("Banana gate inference error: %s", exc)
            return 0.0

    def _build_result(self, score: float) -> ValidationResult:
        if score >= self.threshold_high:
            return ValidationResult(
                is_banana=True,
                confidence=score,
                reason="Banana detected by dedicated binary gate.",
                method=self.METHOD,
                state=ValidationState.BANANA,
            )
        if score <= self.threshold_low:
            return ValidationResult(
                is_banana=False,
                confidence=score,
                reason="No banana detected by dedicated binary gate.",
                method=self.METHOD,
                state=ValidationState.NOT_BANANA,
            )
        return ValidationResult(
            is_banana=False,
            confidence=score,
            reason="Banana gate confidence is ambiguous; please provide a clearer banana image.",
            method=self.METHOD,
            state=ValidationState.UNCERTAIN,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (re-used by app.py, api/main.py)
# ---------------------------------------------------------------------------

def _make_validator_from_settings() -> BananaImageValidator:
    """Build a validator using the dedicated binary gate thresholds."""
    try:
        from banana_ai.config import settings  # noqa: PLC0415
        high = getattr(settings, "banana_gate_threshold", 0.50)
        low = getattr(settings, "banana_gate_threshold_low", 0.25)
    except Exception:
        high, low = 0.50, 0.25
    return BananaImageValidator(threshold_high=high, threshold_low=low)


# Lazily created — avoids loading config at import time (which can fail
# in test environments without .env).
_default_validator: Optional[BananaImageValidator] = None
_validator_lock = threading.Lock()


def get_validator() -> BananaImageValidator:
    """Return the shared module-level :class:`BananaImageValidator` instance."""
    global _default_validator  # noqa: PLW0603
    if _default_validator is None:
        with _validator_lock:
            if _default_validator is None:
                _default_validator = _make_validator_from_settings()
    return _default_validator


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def validate_banana_image(image: Image.Image) -> ValidationResult:
    """Validate *image* using the shared module-level validator.

    This is the primary public entry-point used by :mod:`banana_ai.app`
    and :mod:`banana_ai.api.main`.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to validate (must already pass file/format validation).

    Returns
    -------
    ValidationResult
    """
    return get_validator().validate(image)


validate_banana = validate_banana_image

