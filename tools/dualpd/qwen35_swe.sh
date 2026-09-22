#!/usr/bin/env bash
# SWE-bench fixed sampling: temperature=0.6, top_p=0.95, top_k=20.
# Reuse the 27B openai_tools workload verbatim: 8192/turn, 64 turns, 81920 total,
# context 131072, prefill chunk/batch 8192; no new retries or grading changes.
# qwen35_swe.json validates workload/data hashes before launching in pd_multi_node.
set -euo pipefail
dualpd_tools="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${DUALPD_PYTHON:-python3}" "${dualpd_tools}/qwen35_swe.py" "$@"
