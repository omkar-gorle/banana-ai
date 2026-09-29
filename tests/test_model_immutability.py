"""Regression test for Banana CNN V2 model immutability.

CRITICAL INVARIANT:
The production ripeness model at models/banana_cnn_v2.pt must remain FROZEN.
Its SHA-256 must match exactly:
cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest
import torch

EXPECTED_V2_SHA256 = "cf053a109c32f30ec50d007712dc0dde7be0c47375d3c708f14483f3b1a8869d"
EXPECTED_CLASSES = ["overripe", "ripe", "rotten", "unripe"]


def test_banana_cnn_v2_sha256_exact():
    model_path = Path("models/banana_cnn_v2.pt")
    assert model_path.exists(), f"Model file not found at {model_path}"

    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()

    assert actual_hash.lower() == EXPECTED_V2_SHA256.lower(), (
        f"CRITICAL: models/banana_cnn_v2.pt hash mismatch!\n"
        f"Expected: {EXPECTED_V2_SHA256}\n"
        f"Actual:   {actual_hash}"
    )


def test_banana_cnn_v2_can_load_and_classes():
    model_path = Path("models/banana_cnn_v2.pt")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    assert "model_state_dict" in checkpoint or isinstance(checkpoint, dict)
    if "class_names" in checkpoint:
        assert checkpoint["class_names"] == EXPECTED_CLASSES
