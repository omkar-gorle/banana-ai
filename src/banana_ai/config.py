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
    # V2 is the production model. V1 remains available explicitly for
    # historical training/evaluation commands.
    model_path: str = "models/banana_cnn_v2.pt"
    model_version: str = "banana-cnn-v2"
    image_size: int = 224
    dataset_root: str = "data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
    )

    @property
    def root_dir(self) -> Path:
        return Path(__file__).resolve().parents[2]


settings = Settings()
