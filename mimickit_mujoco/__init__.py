"""Mac-first MuJoCo execution tools for MimicKit."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bootstrap():
    """Keep upstream's absolute imports isolated behind one entry point."""
    path = str(ROOT / "mimickit")
    if path not in sys.path:
        sys.path.insert(0, path)
