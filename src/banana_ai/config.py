from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the project.

    Values can come from a .env file or environment variables.
    Keeping configuration here prevents database URLs and model
    paths from being hard-coded throughout the application.
    """

    database_url: str = (
        "postgresql+psycopg://banana_user:banana_password@localhost:5432/banana_ai"
    )
    # V3 is the production model. V1 and V2 remain available explicitly for
    # historical training/evaluation commands.
    model_path: str = "models/banana_cnn_v3.pt"
    model_version: str = "banana-cnn-v3"
    image_size: int = 224
    dataset_root: str = "data"

    # ---------------------------------------------------------------------------
    # Banana content validation thresholds
    # ---------------------------------------------------------------------------
    # These are application-level thresholds and are NOT scientifically optimal.
    # Tune them using real banana and non-banana test images.
    #
    #   score >= banana_detection_threshold_high  →  BANANA   (proceed to ripeness)
    #   score <= banana_detection_threshold_low   →  NOT_BANANA (reject)
    #   between the two                           →  UNCERTAIN  (reject, ask for clearer image)
    #
    # Override via environment variables:
    #   BANANA_DETECTION_THRESHOLD_HIGH=0.60
    #   BANANA_DETECTION_THRESHOLD_LOW=0.25
    banana_detection_threshold_high: float = 0.60
    banana_detection_threshold_low: float = 0.25

    # Abstention threshold (EXPERIMENTAL)
    ripeness_abstention_enabled: bool = True
    ripeness_abstention_threshold: float = 0.70

    # Diagnostic tools
    banana_diagnostics: bool = False

    # Dedicated binary banana-vs-non-banana gate checkpoint and threshold.
    banana_gate_model_path: str = "models/banana_gate_best.pt"
    # Keep the application policy at 0.50 to preserve the existing
    # BANANA/UNCERTAIN boundary. The v2 checkpoint also rejects all held-out
    # HaGRIDv2 hand negatives at this policy threshold.
    banana_gate_threshold: float = 0.50
    banana_gate_threshold_low: float = 0.25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
    )

    @property
    def root_dir(self) -> Path:
        return Path(__file__).resolve().parents[2]


settings = Settings()
