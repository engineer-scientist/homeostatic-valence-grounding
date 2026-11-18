#!/usr/bin/env python3
"""Thin wrapper: python scripts/patch.py ..."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hvg.cli import main_patch
if __name__ == "__main__":
    main_patch()
