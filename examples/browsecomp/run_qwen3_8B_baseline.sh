#!/bin/bash
# BrowseComp length-penalty experiment cell: Qwen3-8B / baseline.
# See docs/experiments/browsecomp-length-penalty.md.
#
# On the GB300 cluster this wraps the parameterized fully-async launcher
# (1 train node + 3 rollout nodes, 4 GPUs each). Submit via Slurm:
#   sbatch -J bc-8B-baseline -w <4-node-range> \
#     --export=ALL,MODEL_SIZE=8B,MODE=baseline \
#     examples/browsecomp/slurm_gb300/train_async_exp.sbatch
#
# Or run directly inside an interactive 4-node allocation (train node,
# after ray workers joined and search server is up):
export MODEL_SIZE=8B
export MODE=baseline
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "${SCRIPT_DIR}/slurm_gb300/run_browsecomp_async_gb300.sh"
