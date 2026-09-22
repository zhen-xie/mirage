"""Make the Qwen3 demo modules importable under the pytest console script."""

import sys
from pathlib import Path


QWEN3_DIR = Path(__file__).resolve().parents[2] / "demo" / "qwen3"
sys.path.insert(0, str(QWEN3_DIR))
