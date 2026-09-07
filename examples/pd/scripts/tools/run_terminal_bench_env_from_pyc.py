#!/usr/bin/env python3
"""Start the tracked Terminal-Bench OpenEnv service from orphaned bytecode.

The serving node's OpenEnv checkout is occasionally mounted read-only and can
temporarily lose the six ``tbench2_env`` source files while retaining their
CPython bytecode.  This loader is a narrow runtime fallback: normal source
imports remain preferred by ``start_terminal_bench_env.sh``.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _package(name: str, directory: Path) -> None:
    module = ModuleType(name)
    module.__file__ = str(directory)
    module.__package__ = name
    module.__path__ = [str(directory)]
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=True
    )
    sys.modules[name] = module


def _load(name: str, bytecode: Path):
    if not bytecode.is_file():
        raise FileNotFoundError(f"missing Terminal-Bench bytecode: {bytecode}")
    loader = importlib.machinery.SourcelessFileLoader(name, str(bytecode))
    spec = importlib.util.spec_from_loader(name, loader)
    if spec is None:
        raise ImportError(f"could not construct a module spec for {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    root = args.root.resolve()
    server = root / "server"
    tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    _package("tbench2_env", root)
    _load("tbench2_env.models", root / "__pycache__" / f"models.{tag}.pyc")
    _package("tbench2_env.server", server)
    _load(
        "tbench2_env.server.tbench2_env_environment",
        server / "__pycache__" / f"tbench2_env_environment.{tag}.pyc",
    )
    app = _load(
        "tbench2_env.server.app",
        server / "__pycache__" / f"app.{tag}.pyc",
    )
    app.main(port=args.port)


if __name__ == "__main__":
    main()
