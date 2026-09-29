"""Enhanced FastAPI application for Banana AI.

Endpoints:
  GET  /health      - application health check
  POST /predict     - classify a banana image (with optional env-aware shelf-life)

Backward compatibility:
  Existing /health and /predict responses are preserved.
  New fields are additive.
"""

from pathlib import Path
import tempfile
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import UnidentifiedImageError
from sqlalchemy.orm import Session

from banana_ai.config import settings
from banana_ai.db.crud import save_prediction
from banana_ai.db.session import get_db
from banana_ai.ml.predict import predict
from banana_ai.services.shelf_life import estimate_shelf_life
from banana_ai.services.image_validation import (
    validate_image_bytes,
    ImageValidationError,
    ALLOWED_EXTENSIONS,
)
from banana_ai.services.banana_validation import (
    validate_banana_image,
    ValidationState,
)


app = FastAPI(
    title="Banana AI API",
    version="2.0.0",
    description=(
        "Banana Quality & Shelf-Life Intelligence API. "
        "Classifies banana images into: overripe, ripe, rotten, unripe. "
        "Optional environment-aware shelf-life estimation (prototype heuristic). "
        "Shelf-life estimate is NOT a trained model and NOT a food-safety guarantee."
    ),
)


@app.get("/health", summary="Health check")
def health():
    """Check application availability. Returns model path and version."""
    model_exists = Path(settings.model_path).exists()
    return {
        "status": "ok",
        "model_version": settings.model_version,
        "model_path": settings.model_path,
        "model_ready": model_exists,
    }


@app.post("/predict", summary="Classify a banana image")
async def predict_banana(
    file: UploadFile = File(..., description="Banana image (jpg/jpeg/png/webp)"),
    temperature_c: Optional[float] = Query(
        None,
        description="Ambient temperature in \u00b0C (optional, for shelf-life heuristic)",
        ge=-20.0,
        le=60.0,
    ),
    humidity_pct: Optional[float] = Query(
        None,
        description="Relative humidity in % (optional, for shelf-life heuristic)",
        ge=0.0,
        le=100.0,
    ),
    storage_condition: Optional[str] = Query(
        None,
        description="Storage condition: room | cool | refrigerator | other",
    ),
    db: Session = Depends(get_db),
):
    """Accept a banana image, run the CNN classifier, and return the result.

    The request is processed in the following order:

    1. File extension validation
    2. Model availability check
    3. Image bytes validation (format, size, dimensions)
    4. **Banana content validation** -- non-banana images are rejected here
       with HTTP 400 before the ripeness model is ever invoked.
    5. Ripeness classification
    6. Shelf-life estimation
    7. Database save (optional)

    Returns:

    - **predicted_stage**: overripe | ripe | rotten | unripe
    - **confidence**: model softmax probability for predicted class
    - **probabilities**: softmax probability for all classes
    - **model_version**: identifier of the model used
    - **banana_detection_confidence**: score from banana content validator
    - **estimated_days_left**: prototype heuristic display string
    - **estimated_min_days**: lower bound (if environmental inputs provided)
    - **estimated_max_days**: upper bound (if environmental inputs provided)
    - **shelf_life_method**: always "heuristic"
    - **shelf_life_warning**: mandatory disclaimer
    - **prediction_id**: database row ID (if DB is available)

    Note: estimated_days_left / shelf-life fields are PROTOTYPE HEURISTICS.
    They are NOT a trained shelf-life model and NOT a food-safety guarantee.
    """
    # ---- Validate file extension ----
    filename = file.filename or "upload.jpg"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # ---- Validate model exists ----
    if not Path(settings.model_path).exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not found at '{settings.model_path}'. "
                "Train the model first: python -m banana_ai.ml.train"
            ),
        )

    # ---- Read and validate image bytes ----
    content = await file.read()
    try:
        pil_image = validate_image_bytes(content, filename=filename)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ---- Banana content validation ----
    # IMPORTANT: The ripeness model must NEVER be called for non-banana images.
    # validate_banana_image runs the dedicated Banana Gate binary classifier
    # (BananaGateMobileNetV3) to confirm banana presence before any ripeness
    # classification, shelf-life estimation, or database writes.
    banana_result = validate_banana_image(pil_image)

    if banana_result.state == ValidationState.NOT_BANANA:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "NOT_BANANA",
                "message": (
                    "The uploaded image does not appear to contain a banana. "
                    "Please upload a clear image of a banana."
                ),
                "banana_detection_confidence": round(banana_result.confidence, 4),
                "method": banana_result.method,
            },
        )

    if banana_result.state == ValidationState.UNCERTAIN:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "BANANA_UNCERTAIN",
                "message": (
                    "Unable to confidently confirm that the image contains a banana. "
                    "Please provide a clearer image with the banana more prominent and "
                    "well-lit."
                ),
                "banana_detection_confidence": round(banana_result.confidence, 4),
                "method": banana_result.method,
            },
        )

    # ---- Ripeness classification (banana confirmed) ----
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(content)
        temp_path = temp.name

    try:
        result = predict(temp_path, settings.model_path)

        # ---- Shelf-life estimation ----
        shelf_life = estimate_shelf_life(
            predicted_stage=result["stage"],
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            storage_condition=storage_condition,
        )

        # ---- Attempt DB save ----
        prediction_id = None
        db_error = None
        try:
            row = save_prediction(
                db=db,
                image_path=filename,
                stage=result["stage"],
                confidence=result["confidence"],
                estimated_days_left=shelf_life.display(),
                model_version=settings.model_version,
                temperature_c=temperature_c,
                humidity_pct=humidity_pct,
                storage_condition=storage_condition,
                input_method="api",
                estimated_min_days=shelf_life.estimated_min_days,
                estimated_max_days=shelf_life.estimated_max_days,
                shelf_life_method=shelf_life.method,
            )
            prediction_id = row.id
        except Exception as exc:
            db_error = str(exc)

        response = {
            # Core ripeness prediction (existing fields -- backward compatible)
            "predicted_stage": result["stage"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "model_version": settings.model_version,
            # Banana detection metadata
            "banana_detection_confidence": round(banana_result.confidence, 4),
            "banana_detection_method": banana_result.method,
            # Shelf-life fields
            "estimated_days_left": shelf_life.display(),
            "estimated_min_days": shelf_life.estimated_min_days,
            "estimated_max_days": shelf_life.estimated_max_days,
            "shelf_life_method": shelf_life.model_type,
            "shelf_life_warning": (
                "estimated_days_left is a PROTOTYPE heuristic estimate. "
                "It is NOT a trained shelf-life model and NOT a food-safety guarantee."
            ),
        }
        if temperature_c is None and humidity_pct is None:
            response["shelf_life_note"] = (
                "Environmental inputs not provided -- shelf-life estimate used stage-only baseline."
            )
        if prediction_id is not None:
            response["prediction_id"] = prediction_id
        if db_error:
            response["db_warning"] = f"Prediction not saved to DB: {db_error}"

        return JSONResponse(content=response)

    except (ValueError, UnidentifiedImageError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
    finally:
        Path(temp_path).unlink(missing_ok=True)
