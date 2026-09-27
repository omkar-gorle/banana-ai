"""Pytest configuration file.

Ensures the src/ directory is on the Python path so tests can import
banana_ai modules without needing to set PYTHONPATH manually.

Note: pyproject.toml already sets pythonpath = ["src"] via pytest.ini_options,
but this conftest is kept for explicit clarity and compatibility.
"""

import sys
from pathlib import Path

# Add src to path (redundant with pyproject.toml but explicit)
src = Path(__file__).parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
