from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from banana_ai.db.session import Base


class Observation(Base):
    """One image observation of a banana."""

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional environmental information. This becomes useful later
    # for a real shelf-life model.
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Storage condition metadata (prototype only; not scientifically calibrated)
    storage_condition: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Input method: "camera" or "upload" (informational only)
    input_method: Mapped[str | None] = mapped_column(String(32), nullable=True)

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class Prediction(Base):
    """The AI result associated with one observation."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observation_id: Mapped[int] = mapped_column(Integer, nullable=False)

    predicted_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Prototype now; real learned shelf-life model later.
    estimated_days_left: Mapped[str] = mapped_column(String(32), nullable=False)

    # Extended shelf-life fields (additive — NULL for pre-existing rows)
    estimated_min_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_max_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shelf_life_method: Mapped[str | None] = mapped_column(String(64), nullable=True)

    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class PredictionFeedback(Base):
    """Human feedback on a prediction.

    Stored in a separate table so it never overwrites or corrupts existing
    prediction records.  DO NOT use this table to retrain the model
    automatically; it is a human-feedback dataset for future use only.
    """

    __tablename__ = "prediction_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # "correct" or "incorrect"
    feedback: Mapped[str] = mapped_column(String(32), nullable=False)

    # Optional correction provided by the user
    corrected_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
