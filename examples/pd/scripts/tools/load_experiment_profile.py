#!/usr/bin/env python3
"""Emit a checked experiment profile as KEY<TAB>VALUE records for Bash."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


# Profiles may configure the experiment, but may not inject arbitrary process
# environment into a serving launch.
ALLOWED_KEYS = {
    "WORKLOAD_CONFIG",
    "MODEL_PATH",
    "PD_DATA_ROOT",
    "PREFILL_GPUS",
    "PREFILL_PORTS",
    "BOOTSTRAP_PORTS",
    "DECODE_GPUS",
    "DECODE_PORTS",
    "DECODE_MEM_FRACTION_STATICS",
    "SEARCH_GPU",
    "SEARCH_PORT",
    "REQUESTS",
    "WARMUP_REQUESTS",
    "MAX_INFLIGHT",
    "SEED",
    "TEMPERATURE",
    "TOP_P",
    "TOP_K",
    "MAX_CONTEXT_LENGTH",
    "MAX_RESPONSE_LENGTH",
    "WARMUP_SECONDS",
    "MAX_WARMUP_SECONDS",
    "MEASURE_SECONDS",
    "CLOSED_LOOP",
    "D_TARGET_KV_FRACTION",
    "FAST_TOOL_THRESHOLD_SECONDS",
    "DIRECT_WAIT_SECONDS",
    "D2P_HOST_ARENA_GIB_PER_P",
    "P2D_HOST_ARENA_GIB_PER_P",
    "P2D_HOST_STAGING",
    "P_TO_D_CONSUMERS",
    "MAX_PREFILL_INFLIGHT",
    "PD_INFERENCE_RETURN_LOGPROB",
}
PATH_KEYS = {"WORKLOAD_CONFIG"}


def scalar_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        text = str(value)
        if "\n" in text or "\t" in text:
            raise ValueError("profile values may not contain tabs or newlines")
        return text
    raise TypeError(f"profile environment values must be scalar, got {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()

    path = args.profile.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("experiment profile must contain one object")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("only experiment profile schema_version=1 is supported")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("experiment profile environment must be a non-empty object")
    unknown = sorted(set(environment) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"unsupported experiment profile keys: {', '.join(unknown)}")

    for key, raw_value in environment.items():
        value = scalar_text(raw_value)
        if key in PATH_KEYS:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            value = str(candidate.resolve())
        print(f"{key}\t{value}")


if __name__ == "__main__":
    main()
