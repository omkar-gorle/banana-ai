"""Database CRUD operations for Banana AI.

Saving is always explicit — the UI/API layer must call these functions
intentionally.  No automatic saves happen on Streamlit reruns.
"""

from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from banana_ai.db.models import Observation, Prediction, PredictionFeedback


def save_prediction(
    db: Session,
    image_path: str,
    stage: str,
    confidence: float,
    estimated_days_left: str,
    model_version: str,
    temperature_c: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    storage_condition: Optional[str] = None,
    input_method: Optional[str] = None,
    estimated_min_days: Optional[int] = None,
    estimated_max_days: Optional[int] = None,
    shelf_life_method: Optional[str] = None,
) -> Prediction:
    """Store an observation and its AI prediction.

    Parameters
    ----------
    db : Session
        SQLAlchemy session.
    image_path : str
        Filename or path label for the image.
    stage : str
        Predicted ripeness stage.
    confidence : float
        Model softmax confidence for the predicted class.
    estimated_days_left : str
        Human-readable shelf-life estimate string (prototype heuristic).
    model_version : str
        Model identifier (e.g. ``banana-cnn-v2``).
    temperature_c : float, optional
        Ambient temperature in °C.
    humidity_pct : float, optional
        Relative humidity in %.
    storage_condition : str, optional
        Storage condition label.
    input_method : str, optional
        ``"camera"`` or ``"upload"``.
    estimated_min_days : int, optional
        Lower bound from the shelf-life service.
    estimated_max_days : int, optional
        Upper bound from the shelf-life service.
    shelf_life_method : str, optional
        Method label from the shelf-life service.

    Returns
    -------
    Prediction
        The committed prediction row (with populated ``id``).
    """
    observation = Observation(
        image_path=image_path,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        storage_condition=storage_condition,
        input_method=input_method,
    )
    db.add(observation)
    db.flush()  # Get observation.id without committing yet.

    prediction = Prediction(
        observation_id=observation.id,
        predicted_stage=stage,
        confidence=confidence,
        estimated_days_left=estimated_days_left,
        model_version=model_version,
        estimated_min_days=estimated_min_days,
        estimated_max_days=estimated_max_days,
        shelf_life_method=shelf_life_method,
    )
    db.add(prediction)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(prediction)

    return prediction


def save_feedback(
    db: Session,
    prediction_id: int,
    feedback: str,
    corrected_stage: Optional[str] = None,
) -> PredictionFeedback:
    """Store human feedback for a prediction.

    Note: This does NOT retrain the model. The feedback table is for
    future dataset collection only.
    """
    row = PredictionFeedback(
        prediction_id=prediction_id,
        feedback=feedback,
        corrected_stage=corrected_stage,
    )
    db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(row)
    return row


def recent_predictions(db: Session, limit: int = 20):
    """Return the most recent predictions joined with observations."""
    statement = (
        select(Prediction, Observation)
        .join(Observation, Prediction.observation_id == Observation.id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
    )
    return db.execute(statement).all()


def all_predictions(
    db: Session,
    stage_filter: Optional[str] = None,
    model_version_filter: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 500,
):
    """Return predictions with optional filters."""
    stmt = (
        select(Prediction, Observation)
        .join(Observation, Prediction.observation_id == Observation.id)
        .order_by(Prediction.created_at.desc())
    )
    if stage_filter:
        stmt = stmt.where(Prediction.predicted_stage == stage_filter)
    if model_version_filter:
        stmt = stmt.where(Prediction.model_version == model_version_filter)
    if min_confidence is not None:
        stmt = stmt.where(Prediction.confidence >= min_confidence)
    stmt = stmt.limit(limit)
    return db.execute(stmt).all()


def count_by_stage(db: Session) -> dict:
    """Return a dict mapping stage -> count from all stored predictions."""
    rows = db.execute(
        select(Prediction.predicted_stage, func.count(Prediction.id))
        .group_by(Prediction.predicted_stage)
    ).all()
    return {stage: cnt for stage, cnt in rows}


def total_predictions(db: Session) -> int:
    """Return total number of stored predictions."""
    result = db.execute(select(func.count(Prediction.id))).scalar()
    return result or 0
