from sqlalchemy import select
from sqlalchemy.orm import Session

from banana_ai.db.models import Observation, Prediction


def save_prediction(
    db: Session,
    image_path: str,
    stage: str,
    confidence: float,
    estimated_days_left: str,
    model_version: str,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
):
    """Store an observation and its AI prediction."""

    observation = Observation(
        image_path=image_path,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
    )
    db.add(observation)
    db.flush()  # Get observation.id without committing yet.

    prediction = Prediction(
        observation_id=observation.id,
        predicted_stage=stage,
        confidence=confidence,
        estimated_days_left=estimated_days_left,
        model_version=model_version,
    )
    db.add(prediction)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(prediction)

    return prediction


def recent_predictions(db: Session, limit: int = 20):
    statement = (
        select(Prediction, Observation)
        .join(Observation, Prediction.observation_id == Observation.id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
    )
    return db.execute(statement).all()
