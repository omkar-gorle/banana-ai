"""Banana shelf-life estimation service.

This module provides environment-aware shelf-life estimation for bananas.

IMPORTANT DISCLAIMERS:
- This is a PROTOTYPE HEURISTIC, NOT a trained shelf-life model.
- The coefficients below are prototype assumptions, not experimentally
  validated scientific constants.
- Results must NEVER be presented as guaranteed shelf life, guaranteed
  safety, or medically/scientifically validated estimates.
- Always advise inspecting the banana for visible mold, unusual odor,
  leakage, or other signs of spoilage.

Future:
    The ShelfLifeEstimator base class is designed to allow a future
    MLShelfLifeEstimator to replace HeuristicShelfLifeEstimator once
    a real longitudinal dataset (banana_id, temperature_c, humidity_pct,
    storage_condition, days_since_start, days_left, observed_stage) has
    been collected and a regression/time-to-event model trained on it.
    Split such longitudinal data by banana_id to prevent leakage.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants — clearly documented as PROTOTYPE ASSUMPTIONS
# ---------------------------------------------------------------------------

#: Baseline good-quality window ranges (min_days, max_days) per stage.
#: These are prototype assumptions, NOT experimentally validated constants.
STAGE_RANGES: dict[str, tuple[int, int]] = {
    "unripe": (4, 7),
    "ripe": (2, 4),
    "overripe": (0, 2),
    "rotten": (0, 0),
}

#: Valid temperature bounds (°C). Values outside this range are clamped.
TEMP_MIN_C: float = -20.0
TEMP_MAX_C: float = 60.0

#: Valid humidity bounds (%). Values outside this range are clamped.
HUMIDITY_MIN_PCT: float = 0.0
HUMIDITY_MAX_PCT: float = 100.0

#: Reference temperature (°C) — moderate ambient. Prototype assumption.
TEMP_REFERENCE_C: float = 22.0

#: How many days are added/removed per degree BELOW/ABOVE reference.
#: Prototype assumption: cooler = slower ripening = more days.
TEMP_COEFFICIENT: float = 0.08  # days per °C deviation

#: Reference humidity (%). Prototype assumption.
HUMIDITY_REFERENCE_PCT: float = 60.0

#: How many days are adjusted per 10 % above/below reference humidity.
#: Prototype assumption: higher humidity slightly accelerates deterioration.
HUMIDITY_COEFFICIENT: float = 0.05  # days per % deviation

#: Storage condition multipliers — prototype assumptions only.
#: These are NOT calibrated from experimental data.
STORAGE_MULTIPLIERS: dict[str, float] = {
    "room": 1.0,
    "cool": 1.3,
    "refrigerator": 1.6,
    "other": 1.0,
}

#: Human-readable storage condition labels.
STORAGE_LABELS: list[str] = ["room", "cool", "refrigerator", "other"]

#: Absolute maximum days the estimator will ever return (sanity cap).
MAX_DAYS_CAP: int = 14


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class ShelfLifeEstimate:
    """Result of a shelf-life estimation call.

    Supports two calling conventions for backward compatibility:

    New (preferred)::

        ShelfLifeEstimate(
            estimated_min_days=2, estimated_max_days=4,
            explanation="...", warning="..."
        )

    Old (legacy — keeps existing tests working)::

        ShelfLifeEstimate(minimum_days=2, maximum_days=4, method="test")

    Attributes
    ----------
    estimated_min_days / minimum_days : int
        Lower bound of the estimated good-quality window.
    estimated_max_days / maximum_days : int
        Upper bound of the estimated good-quality window.
    explanation : str
        Short human-readable explanation of the estimate.
    warning : str
        Mandatory disclaimer about the prototype nature of this estimate.
    model_type : str
        Always ``"heuristic"`` for the current implementation.
    method : str
        Specific method label (e.g. ``"heuristic_env_aware"``).
    """

    def __init__(
        self,
        # New-style keyword args
        estimated_min_days: Optional[int] = None,
        estimated_max_days: Optional[int] = None,
        explanation: str = "",
        warning: str = "",
        model_type: str = "heuristic",
        method: str = "heuristic_env_aware",
        # Old-style keyword args (backward compat)
        minimum_days: Optional[int] = None,
        maximum_days: Optional[int] = None,
    ):
        # Resolve min/max from whichever style was used
        if estimated_min_days is None and minimum_days is not None:
            estimated_min_days = minimum_days
        if estimated_max_days is None and maximum_days is not None:
            estimated_max_days = maximum_days
        if estimated_min_days is None:
            estimated_min_days = 0
        if estimated_max_days is None:
            estimated_max_days = 0

        self.estimated_min_days: int = estimated_min_days
        self.estimated_max_days: int = estimated_max_days
        # Aliases (backward compat)
        self.minimum_days: int = estimated_min_days
        self.maximum_days: int = estimated_max_days
        self.explanation: str = explanation
        self.warning: str = warning
        self.model_type: str = model_type
        self.method: str = method

    def display(self) -> str:
        """Return a short human-readable range string."""
        if self.estimated_min_days == self.estimated_max_days:
            return f"{self.estimated_min_days} day(s)"
        return f"{self.estimated_min_days}-{self.estimated_max_days} days"


# ---------------------------------------------------------------------------
# Abstract base class — enables future MLShelfLifeEstimator
# ---------------------------------------------------------------------------


class ShelfLifeEstimator(ABC):
    """Abstract base for shelf-life estimators.

    Both the current HeuristicShelfLifeEstimator and any future
    MLShelfLifeEstimator must implement this interface so the UI/API
    layer never needs to know which implementation is active.
    """

    @abstractmethod
    def estimate(
        self,
        predicted_stage: str,
        temperature_c: Optional[float] = None,
        humidity_pct: Optional[float] = None,
        storage_condition: Optional[str] = None,
    ) -> ShelfLifeEstimate:
        ...


# ---------------------------------------------------------------------------
# Heuristic implementation
# ---------------------------------------------------------------------------


class HeuristicShelfLifeEstimator(ShelfLifeEstimator):
    """Prototype environment-aware shelf-life estimator.

    Uses a simple bounded linear formula:
    ``adjusted_days = baseline + temp_effect + humidity_effect``
    then applies a storage multiplier.

    All coefficients are prototype assumptions documented in the module-level
    constants. This estimator is deterministic: the same inputs always
    produce the same output.
    """

    def estimate(
        self,
        predicted_stage: str,
        temperature_c: Optional[float] = None,
        humidity_pct: Optional[float] = None,
        storage_condition: Optional[str] = None,
    ) -> ShelfLifeEstimate:
        stage = (predicted_stage or "").lower().strip()
        base_min, base_max = STAGE_RANGES.get(stage, (0, 0))

        if stage == "rotten":
            return ShelfLifeEstimate(
                estimated_min_days=0,
                estimated_max_days=0,
                explanation=(
                    "Model classified banana as rotten. "
                    "Do not estimate a remaining good-quality window."
                ),
                warning=_mandatory_warning(),
                method="heuristic_env_aware",
            )

        # ---- Validate / clamp inputs ----
        temp_used: Optional[float] = None
        hum_used: Optional[float] = None

        if temperature_c is not None:
            temp_used = float(
                max(TEMP_MIN_C, min(TEMP_MAX_C, temperature_c))
            )
        if humidity_pct is not None:
            hum_used = float(
                max(HUMIDITY_MIN_PCT, min(HUMIDITY_MAX_PCT, humidity_pct))
            )

        # ---- Compute adjustments ----
        temp_adj = 0.0
        if temp_used is not None:
            # Cooler than reference → positive adjustment (more days)
            temp_adj = (TEMP_REFERENCE_C - temp_used) * TEMP_COEFFICIENT

        hum_adj = 0.0
        if hum_used is not None:
            # Higher humidity than reference → negative adjustment (fewer days)
            hum_adj = (HUMIDITY_REFERENCE_PCT - hum_used) * HUMIDITY_COEFFICIENT

        # ---- Apply storage multiplier ----
        storage_key = (storage_condition or "room").lower().strip()
        multiplier = STORAGE_MULTIPLIERS.get(storage_key, 1.0)

        # ---- Compute final range ----
        adjusted_min = (base_min + temp_adj + hum_adj) * multiplier
        adjusted_max = (base_max + temp_adj + hum_adj) * multiplier

        # Clamp to [0, MAX_DAYS_CAP]
        final_min = max(0, min(MAX_DAYS_CAP, round(adjusted_min)))
        final_max = max(0, min(MAX_DAYS_CAP, round(adjusted_max)))

        # Ensure min <= max
        if final_min > final_max:
            final_min, final_max = final_max, final_min

        # ---- Build explanation ----
        parts = [f"Stage: {stage}."]
        if temp_used is not None:
            direction = "cooler" if temp_used < TEMP_REFERENCE_C else "warmer"
            parts.append(
                f"Temperature {temp_used:.1f} °C is {direction} than reference "
                f"({TEMP_REFERENCE_C} °C) — prototype adjustment: {temp_adj:+.2f} days."
            )
        if hum_used is not None:
            direction = "drier" if hum_used < HUMIDITY_REFERENCE_PCT else "more humid"
            parts.append(
                f"Humidity {hum_used:.0f} % is {direction} than reference "
                f"({HUMIDITY_REFERENCE_PCT:.0f} %) — prototype adjustment: {hum_adj:+.2f} days."
            )
        if storage_key != "room":
            parts.append(
                f"Storage condition '{storage_key}' — prototype multiplier: ×{multiplier}."
            )

        explanation = " ".join(parts)

        return ShelfLifeEstimate(
            estimated_min_days=final_min,
            estimated_max_days=final_max,
            explanation=explanation,
            warning=_mandatory_warning(),
            method="heuristic_env_aware",
        )


# ---------------------------------------------------------------------------
# Module-level singleton (default estimator)
# ---------------------------------------------------------------------------

_default_estimator = HeuristicShelfLifeEstimator()


def estimate_shelf_life(
    predicted_stage: str,
    temperature_c: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    storage_condition: Optional[str] = None,
    estimator: Optional[ShelfLifeEstimator] = None,
) -> ShelfLifeEstimate:
    """Estimate the good-quality window for a banana.

    This is the main public entry-point for both the Streamlit UI and the
    FastAPI layer. Both must call this function so the formula is never
    duplicated.

    Parameters
    ----------
    predicted_stage : str
        One of ``overripe``, ``ripe``, ``rotten``, ``unripe``.
    temperature_c : float, optional
        Ambient temperature in °C.
    humidity_pct : float, optional
        Relative humidity in %.
    storage_condition : str, optional
        One of ``room``, ``cool``, ``refrigerator``, ``other``.
    estimator : ShelfLifeEstimator, optional
        Override the default estimator (useful for testing).

    Returns
    -------
    ShelfLifeEstimate
        Contains ``estimated_min_days``, ``estimated_max_days``,
        ``explanation``, ``warning``, ``model_type``, ``method``.
    """
    est = estimator or _default_estimator
    try:
        return est.estimate(predicted_stage, temperature_c, humidity_pct, storage_condition)
    except Exception as exc:
        logger.warning("Shelf-life estimation failed: %s", exc)
        return ShelfLifeEstimate(
            estimated_min_days=0,
            estimated_max_days=0,
            explanation="Estimation unavailable.",
            warning=_mandatory_warning(),
            method="heuristic_env_aware",
        )


# ---------------------------------------------------------------------------
# Backward-compatible wrapper (keeps existing tests & API calls working)
# ---------------------------------------------------------------------------


def prototype_estimate(stage: str) -> str:
    """Return a simple stage-only shelf-life estimate as a display string.

    This wrapper preserves backward compatibility with existing code that
    calls ``prototype_estimate(stage)``. Internally it now delegates to the
    full ``estimate_shelf_life`` service so the formula is not duplicated.
    """
    result = estimate_shelf_life(stage)
    return result.display()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mandatory_warning() -> str:
    return (
        "This estimate is a prototype based on ripeness and environmental "
        "heuristics. It is NOT a food-safety guarantee. Check the fruit for "
        "visible mold, unusual odor, leakage, or other signs of spoilage."
    )
