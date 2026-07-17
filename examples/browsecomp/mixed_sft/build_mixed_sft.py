#!/usr/bin/env python3
"""Build a deterministic, randomly shuffled 50/50 BrowseComp/math SFT mix."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browsecomp", required=True)
    parser.add_argument("--math", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--per-source", type=int, default=558)
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat and independently shuffle the balanced mix this many times.")
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    browsecomp, math = _read(Path(args.browsecomp)), _read(Path(args.math))
    if len(browsecomp) < args.per_source or len(math) < args.per_source:
        raise SystemExit(f"need {args.per_source}/source, got BrowseComp={len(browsecomp)}, math={len(math)}")
    rng.shuffle(browsecomp)
    rng.shuffle(math)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    base_mix = browsecomp[:args.per_source] + math[:args.per_source]
    for item in base_mix:
        meta = dict(item.get("metadata") or {})
        meta.setdefault("sft_source", "browsecomp")
        item["metadata"] = meta
    # Flattening repeated, independently shuffled passes into one epoch avoids
    # SLIME's implicit save-at-every-epoch behavior while preserving exactly
    # the same sample exposures as repeated epochs.
    mixed = []
    for _ in range(args.repeat):
        epoch_mix = list(base_mix)
        rng.shuffle(epoch_mix)
        mixed.extend(epoch_mix)
    output, manifest = Path(args.output), Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in mixed:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest.write_text(json.dumps({
        "seed": args.seed, "browsecomp_records_per_pass": args.per_source,
        "math_records_per_pass": args.per_source, "repeat": args.repeat,
        "browsecomp_records": args.per_source * args.repeat,
        "math_records": args.per_source * args.repeat,
        "total_records": len(mixed), "global_batch_size": 256, "epochs": 1,
        "optimizer_steps": len(mixed) // 256, "target_mixed_rl_steps": 100,
    }, indent=2) + "\n", encoding="utf-8")
    print(manifest.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
