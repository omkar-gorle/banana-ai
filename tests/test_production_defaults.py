from banana_ai.config import Settings
from banana_ai.app import should_save_prediction


def test_defaults_select_v3_checkpoint():
    settings = Settings(_env_file=None)
    assert settings.model_path.endswith("models/banana_cnn_v3.pt")
    assert settings.model_version == "banana-cnn-v3"


def test_save_guard_blocks_duplicate_streamlit_rerun():
    assert should_save_prediction("reports/upload.jpg", "reports/upload.jpg") is False
    assert should_save_prediction(None, "reports/upload.jpg") is True
    assert should_save_prediction("reports/old.jpg", "reports/upload.jpg") is True
