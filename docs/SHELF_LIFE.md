# Banana AI — Shelf-Life Estimation Methodology

## Status: PROTOTYPE HEURISTIC

The shelf-life estimation feature is a **prototype heuristic**.

It is **NOT**:
- A trained machine learning model
- A scientifically validated formula
- A food-safety guarantee
- An exact measurement of remaining days

It **IS**:
- A simple bounded linear formula using prototype coefficients
- Transparent and deterministic (same inputs → same output)
- Clearly labeled throughout the UI and API
- **Strictly protected by the Banana Gate**: Non-banana images are rejected upstream by `BananaGateMobileNetV3`, ensuring shelf-life estimation is only executed for verified bananas.

---

## How It Works

### 1. Baseline Stage Ranges

| Stage | Min Days | Max Days |
|-------|----------|----------|
| Unripe | 4 | 7 |
| Ripe | 2 | 4 |
| Overripe | 0 | 2 |
| Rotten | 0 | 0 |

These baseline ranges are **prototype assumptions**. They are NOT derived from experimental data.

### 2. Temperature Adjustment

```
reference_temperature = 22.0 °C
temp_coefficient = 0.08 days/°C

temp_adjustment = (reference_temperature - temperature_c) × temp_coefficient
```

- Cooler than 22 °C → positive adjustment (more days)
- Warmer than 22 °C → negative adjustment (fewer days)
- Input is clamped to [-20, 60] °C before calculation

**This coefficient is a prototype assumption, NOT a validated scientific constant.**

### 3. Humidity Adjustment

```
reference_humidity = 60.0 %
humidity_coefficient = 0.05 days/%

humidity_adjustment = (reference_humidity - humidity_pct) × humidity_coefficient
```

- Drier than 60% → positive adjustment (more days)
- More humid than 60% → negative adjustment (fewer days)
- Input is clamped to [0, 100] %

**This coefficient is a prototype assumption, NOT a validated scientific constant.**

### 4. Storage Condition Multiplier

| Storage | Multiplier |
|---------|------------|
| Room / Ambient | ×1.0 |
| Cool Storage | ×1.3 |
| Refrigerator | ×1.6 |
| Other | ×1.0 |

**These multipliers are prototype assumptions. They are NOT calibrated from experimental refrigeration data.**

### 5. Final Formula

```
adjusted_min = (baseline_min + temp_adj + humidity_adj) × storage_multiplier
adjusted_max = (baseline_max + temp_adj + humidity_adj) × storage_multiplier

# Clamped to [0, 14] days and rounded to integers
final_min = clamp(round(adjusted_min), 0, 14)
final_max = clamp(round(adjusted_max), 0, 14)
```

The maximum allowed output is capped at 14 days as a sanity bound.

---

## Example Calculation (Prototype)

Stage: RIPE, Temperature: 27 °C, Humidity: 65%, Storage: Room

```
baseline = (2, 4)
temp_adj = (22 - 27) × 0.08 = -0.40 days
hum_adj  = (60 - 65) × 0.05 = -0.25 days
multiplier = 1.0

adjusted_min = (2 - 0.40 - 0.25) × 1.0 = 1.35 → 1 day
adjusted_max = (4 - 0.40 - 0.25) × 1.0 = 3.35 → 3 days

Result: ~1-3 days (prototype estimate)
```

---

## Mandatory Disclaimer

> This estimate is a prototype based on ripeness and environmental heuristics.
> It is NOT a food-safety guarantee. Check the fruit for visible mold, unusual
> odor, leakage, or other signs of spoilage.

---

## Architecture for Future ML Model

The code is structured to allow a trained `MLShelfLifeEstimator` to replace the heuristic:

```python
# Abstract base (src/banana_ai/services/shelf_life.py)
class ShelfLifeEstimator(ABC):
    def estimate(self, predicted_stage, temperature_c, humidity_pct,
                 storage_condition) -> ShelfLifeEstimate: ...

# Current implementation
class HeuristicShelfLifeEstimator(ShelfLifeEstimator): ...

# Future implementation (not yet built)
class MLShelfLifeEstimator(ShelfLifeEstimator): ...
```

### Required Longitudinal Dataset

To train a real shelf-life model, collect:

| Field | Type | Description |
|-------|------|-------------|
| `banana_id` | str | Unique ID per banana bunch |
| `image_path` | str | Captured image at observation |
| `temperature_c` | float | Ambient temperature |
| `humidity_pct` | float | Relative humidity |
| `storage_condition` | str | Storage type |
| `days_since_start` | int | Days from first observation |
| `days_left` | int | Days until quality threshold crossed |
| `observed_stage` | str | Human-labeled stage at observation |

**Critical:** Split data by `banana_id` (not by row) to prevent data leakage between train/test sets.

Possible model types: regression, survival analysis (time-to-event), ordinal regression.
