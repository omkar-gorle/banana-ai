import hashlib
from pathlib import Path

# The production ripeness model at models/banana_cnn_v2.pt must remain FROZEN.
FROZEN_RIPENESS_SHA256_V2 = "cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d"

# The NEW production model at models/banana_cnn_v3.pt must remain FROZEN.
FROZEN_RIPENESS_SHA256_V3 = "68d79e0c46ef5fa02c9b164ebb5afbddbdc9cbcacc3a4d93bff672169e7aedec"

def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def test_production_ripeness_model_v2_is_unchanged():
    model_path = Path("models/banana_cnn_v2.pt")
    assert model_path.exists(), f"Production model missing: {model_path}"
    current_hash = compute_sha256(model_path)
    if current_hash != FROZEN_RIPENESS_SHA256_V2:
        raise AssertionError("CRITICAL: models/banana_cnn_v2.pt hash mismatch!")

def test_production_ripeness_model_v3_is_unchanged():
    model_path = Path("models/banana_cnn_v3.pt")
    assert model_path.exists(), f"Production model missing: {model_path}"
    current_hash = compute_sha256(model_path)
    if current_hash != FROZEN_RIPENESS_SHA256_V3:
        raise AssertionError("CRITICAL: models/banana_cnn_v3.pt hash mismatch!")

def test_production_model_size_is_reasonable():
    model_path = Path("models/banana_cnn_v3.pt")
    assert model_path.exists()
    size_mb = model_path.stat().st_size / (1024 * 1024)
    assert size_mb < 50.0, f"Model is suspiciously large: {size_mb:.1f} MB"
