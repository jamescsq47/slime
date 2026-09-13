"""Analyze the unchanged OpenEnv trajectory schema without modifying harnesses.

Some inference versions strip the copied metadata.turn_metrics dictionaries.
The complete OpenEnv turn_events remain in the raw record and are authoritative.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PD_DIR = Path(__file__).resolve().parents[3]


def load_analyzer():
    path = PD_DIR / "scripts/tools/analyze_swe_bench_run.py"
    spec = importlib.util.spec_from_file_location("mamba_swe_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.harness_turns

    def complete_turns(row):
        trajectory = (row.get("metadata") or {}).get("openenv_trajectory") or {}
        events = trajectory.get("turn_events") if isinstance(trajectory, dict) else None
        if isinstance(events, list) and events:
            return events
        return original(row)

    module.harness_turns = complete_turns
    return module


if __name__ == "__main__":
    load_analyzer().main()
