import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import torch
from PIL import Image

from banana_ai.config import settings
from banana_ai.services.banana_validation import get_validator


def get_git_commit() -> str:
    # Try common deployment environment variables
    for var in [
        "VERCEL_GIT_COMMIT_SHA",
        "RENDER_GIT_COMMIT",
        "HEROKU_SLUG_COMMIT",
        "SOURCE_VERSION",
        "GITHUB_SHA",
    ]:
        if var in os.environ:
            return os.environ[var]

    # Try local git
    try:
        if (settings.root_dir / ".git").exists():
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(settings.root_dir),
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
    except Exception:
        pass

    return "unknown"


def get_file_sha256(filepath: str | Path) -> str:
    path = Path(filepath)
    if not path.is_absolute():
        path = settings.root_dir / path
    if not path.exists():
        return "File not found"
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception as e:
        return f"Error: {e}"


def run_deterministic_gate_test() -> Dict[str, Any]:
    fixture_path = settings.root_dir / "tests" / "acceptance_images" / "banana_ripe.jpg"
    if not fixture_path.exists():
        return {"available": False, "message": "Known banana fixture unavailable in deployment artifact."}

    try:
        validator = get_validator()
        image = Image.open(fixture_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        score = validator._predict_banana_probability(image)
        result = validator._build_result(score)

        return {
            "available": True,
            "banana_probability": score,
            "non_banana_probability": 1.0 - score,
            "predicted_class": result.state.value,
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


def render_diagnostics_ui():
    if not settings.banana_diagnostics:
        return

    with st.expander("Developer Diagnostics", expanded=False):
        st.markdown("### Runtime Fingerprint")

        commit = get_git_commit()
        st.write(f"**Runtime Commit:** `{commit}`")
        st.write(f"**Model Version:** `{settings.model_version}`")

        gate_path = Path(settings.banana_gate_model_path)
        if not gate_path.is_absolute():
            gate_path = settings.root_dir / gate_path

        st.write(f"**Gate Model Path:** `{gate_path}`")
        st.write(f"**Gate Model Exists:** `{gate_path.exists()}`")

        gate_sha = get_file_sha256(gate_path)
        st.write(f"**Gate Model SHA-256:** `{gate_sha}`")

        v2_sha = get_file_sha256("models/banana_cnn_v2.pt")
        v3_sha = get_file_sha256("models/banana_cnn_v3.pt")
        st.write(f"**V2 SHA-256:** `{v2_sha}`")
        st.write(f"**V3 SHA-256:** `{v3_sha}`")

        st.write(f"**Gate Threshold:** `{settings.banana_gate_threshold:.2f}`")
        st.write(f"**Abstention Threshold:** `{settings.ripeness_abstention_threshold:.2f}`")

        st.markdown("""
        **Gate Class Mapping:**
        - `0 = non_banana`
        - `1 = banana`

        *Note: The displayed 'Gate Confidence' represents the probability of class 1 (banana).*
        """)

        st.markdown("""
        **Preprocessing:**
        - Mode: `RGB`
        - Resize: `224x224`
        - Tensor: `ToTensor()`
        - Normalization: `ImageNet` (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`)
        - Crop: `none`
        - Rotation: `none`
        - Random augmentation: `none`
        """)

        st.markdown("### Environment")
        st.write(f"**Python Version:** `{sys.version}`")
        st.write(f"**PyTorch Version:** `{torch.__version__}`")
        st.write(f"**Streamlit Version:** `{st.__version__}`")

        st.markdown("### Deterministic Gate Self-Test")
        test_res = run_deterministic_gate_test()
        if not test_res["available"]:
            st.warning(test_res["message"])
        elif "error" in test_res:
            st.error(f"Test failed: {test_res['error']}")
        else:
            st.success("Fixture available.")
            st.write(f"**Fixture:** `tests/acceptance_images/banana_ripe.jpg`")
            st.write(f"**Banana Probability (class 1):** `{test_res['banana_probability']:.4f}`")
            st.write(f"**Non-Banana Probability (class 0):** `{test_res['non_banana_probability']:.4f}`")
            st.write(f"**Predicted Class:** `{test_res['predicted_class']}`")
