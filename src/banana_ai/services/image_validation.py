"""Image validation utilities for Banana AI.

Validates uploaded or camera-captured images before they reach the ML pipeline.
Security policy:
- Only allow known image MIME types and extensions.
- Reject files that cannot be opened as valid images.
- Reject files that are too large.
- Reject files with unreasonable dimensions.
- Do not execute uploaded files.
- Prevent path traversal via safe filename generation.
"""

from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Maximum allowed file size in bytes (20 MB)
MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024

#: Minimum image dimension (width or height) in pixels
MIN_DIMENSION_PX: int = 32

#: Maximum image dimension in pixels (guard against huge images)
MAX_DIMENSION_PX: int = 16_000

#: Allowed file extensions (lower-case)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

#: Allowed PIL format strings
ALLOWED_PIL_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP"})


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""


def validate_image_bytes(
    data: bytes,
    filename: Optional[str] = None,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Image.Image:
    """Validate raw image bytes and return an opened PIL Image.

    Parameters
    ----------
    data : bytes
        Raw file contents.
    filename : str, optional
        Original filename (used for extension check).
    max_size_bytes : int
        Maximum allowed file size.

    Returns
    -------
    PIL.Image.Image
        Opened and verified image in RGB mode.

    Raises
    ------
    ImageValidationError
        If any validation check fails.
    """
    # 1. Size check
    if len(data) == 0:
        raise ImageValidationError("Uploaded file is empty.")
    if len(data) > max_size_bytes:
        mb = max_size_bytes // (1024 * 1024)
        raise ImageValidationError(
            f"File is too large ({len(data) // 1024} KB). Maximum allowed: {mb} MB."
        )

    # 2. Extension check (if filename provided)
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            raise ImageValidationError(
                f"File extension '{suffix}' is not allowed. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            )

    # 3. PIL format validation (actually opens the file)
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # Checks file integrity without decoding
        img = Image.open(io.BytesIO(data))  # Reopen after verify
    except UnidentifiedImageError as exc:
        raise ImageValidationError(f"File is not a recognised image: {exc}") from exc
    except Exception as exc:
        raise ImageValidationError(f"Could not open image: {exc}") from exc

    if img.format not in ALLOWED_PIL_FORMATS:
        raise ImageValidationError(
            f"Image format '{img.format}' is not allowed. "
            f"Allowed: {sorted(ALLOWED_PIL_FORMATS)}"
        )

    # 4. Dimension check
    w, h = img.size
    if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
        raise ImageValidationError(
            f"Image too small ({w}×{h} px). Minimum: {MIN_DIMENSION_PX}×{MIN_DIMENSION_PX}."
        )
    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
        raise ImageValidationError(
            f"Image too large ({w}×{h} px). Maximum dimension: {MAX_DIMENSION_PX} px."
        )

    # Convert to RGB (handles RGBA, palette, etc.)
    return img.convert("RGB")


def safe_temp_filename(original_name: Optional[str] = None, suffix: str = ".jpg") -> str:
    """Generate a safe, random filename to prevent path traversal attacks.

    The returned name contains only a UUID — the original filename is never
    used directly in filesystem paths.
    """
    if original_name:
        ext = Path(original_name).suffix.lower()
        if ext in ALLOWED_EXTENSIONS:
            suffix = ext
    return f"banana_{uuid.uuid4().hex}{suffix}"
