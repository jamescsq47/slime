"""Make the PD runtime modules importable when tests are collected in-place."""

from __future__ import annotations

import sys
from pathlib import Path


PD_DIR = Path(__file__).resolve().parents[1]
if str(PD_DIR) not in sys.path:
    sys.path.insert(0, str(PD_DIR))
