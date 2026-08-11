#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import sglang


def digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("baseline", "modified"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    package = Path(sglang.__file__).resolve().parent
    disagg = package / "srt" / "disaggregation"
    custom = {
        name: disagg / name
        for name in (
            "agentic_host_staging.py",
            "agentic_kv_lifecycle.py",
            "agentic_direct_transfer.py",
            "agentic_early_claim.py",
        )
    }
    custom_present = {name: path.exists() for name, path in custom.items()}
    result = {
        "expect": args.expect,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "sglang_version": version("sglang") or getattr(sglang, "__version__", None),
        "sglang_package": str(package),
        "mooncake_transfer_engine_version": version("mooncake-transfer-engine"),
        "custom_modules": custom_present,
        "decode_manager_sha256": digest(disagg / "decode_kvcache_offload_manager.py"),
        "agentic_environment_variables": sorted(
            key for key in os.environ if key.startswith("SGLANG_AGENTIC_KV_")
        ),
    }
    valid = (
        not any(custom_present.values())
        if args.expect == "baseline"
        else all(custom_present.values())
    )
    result["valid"] = valid
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if not valid:
        raise SystemExit(f"environment does not match expected role: {args.expect}")


if __name__ == "__main__":
    main()
