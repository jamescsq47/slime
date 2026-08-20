"""Lazy harness discovery confined to ``examples/pd/data``."""

from __future__ import annotations

import importlib
from functools import lru_cache

from .api import HarnessSpec


@lru_cache(maxsize=None)
def get_harness(name: str) -> HarnessSpec:
    module = importlib.import_module(f"data.{name}")
    harness = getattr(module, "HARNESS", None)
    if not isinstance(harness, HarnessSpec):
        raise TypeError(f"data.{name} must export HARNESS: HarnessSpec")
    if harness.name != name:
        raise ValueError(f"harness package {name!r} exports mismatched name {harness.name!r}")
    return harness
