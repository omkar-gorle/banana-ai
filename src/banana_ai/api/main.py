"""Enhanced FastAPI application for Banana AI.

Endpoints:
  GET  /health      - application health check
  POST /predict     - classify a banana image
"""

from pathlib import Path
import tempfile

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import UnidentifiedImageError
from sqlalchemy.orm import Session

from banana_ai.config import settings
from banana_ai.db.crud import save_prediction
from banana_ai.db.session import get_db
from banana_ai.ml.predict import predict


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(
    title="Banana AI API",
    version="1.0.0",
    description=(
        "Banana ripeness classification API. "
        "Classifies banana images into: overripe, ripe, rotten, unripe."
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
    db: Session = Depends(get_db),
):
    """Accept a banana image, run the CNN classifier, and save the result.

    Returns:
    - **predicted_stage**: overripe | ripe | rotten | unripe
    - **confidence**: model softmax probability for predicted class
    - **probabilities**: softmax probability for all classes
    - **estimated_days_left**: prototype heuristic estimate (NOT a trained model)
    - **model_version**: identifier of the model used
    - **prediction_id**: database row ID (if DB is available)
    """
    # Validate file extension
    filename = file.filename or "upload.jpg"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Validate model exists
    if not Path(settings.model_path).exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not found at '{settings.model_path}'. "
                "Train the model first: python -m banana_ai.ml.train"
            ),
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        temp.write(content)
        temp_path = temp.name

    try:
        result = predict(temp_path, settings.model_path)

        # Attempt to save to DB; continue even if DB is unavailable
        prediction_id = None
        db_error = None
        try:
            row = save_prediction(
                db=db,
                image_path=filename,
                stage=result["stage"],
                confidence=result["confidence"],
                estimated_days_left=result["estimated_days_left"],
                model_version=settings.model_version,
            )
            prediction_id = row.id
        except Exception as exc:
            db_error = str(exc)

        response = {
            "predicted_stage": result["stage"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "estimated_days_left": result["estimated_days_left"],
            "model_version": settings.model_version,
            "shelf_life_warning": (
                "estimated_days_left is a PROTOTYPE heuristic estimate based on the "
                "predicted ripeness stage. It is NOT a trained shelf-life model."
            ),
        }
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
