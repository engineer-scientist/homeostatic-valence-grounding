#!/usr/bin/env python3
"""Thin wrapper: python scripts/fly.py ..."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hvg.cli import main_fly
if __name__ == "__main__":
    main_fly()
