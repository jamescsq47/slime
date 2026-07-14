#!/bin/bash
# Sweep offline eval over all saved checkpoints of the 4 experiment cells.
# Each eval is one 4-GPU slurm job (eval_ckpt.sbatch); this driver submits
# them serially (sbatch --wait) so at most ${PARALLEL} evals run at a time.
# Skips (cell, iter) pairs already present in results.csv — safe to re-run.
#
# Usage: bash eval_sweep.sh [PARALLEL]      # default 2 lanes

set -u
PARALLEL=${1:-2}
BASE=/data/home/syang
LAUNCH=${SLIME_REPO:-${BASE}/CM/slime}/examples/browsecomp/slurm_gb300
RESULTS=${LAUNCH}/eval_results
mkdir -p ${RESULTS}
touch ${RESULTS}/results.csv

# Build the work list: model,mode,iter for every saved ckpt.
worklist() {
  for spec in "8B:baseline:${BASE}/ckpts/browsecomp_qwen3-8B-async" \
              "8B:length_penalty:${BASE}/ckpts/browsecomp_qwen3-8B-length_penalty" \
              "8B:length_penalty_global_ref:${BASE}/ckpts/browsecomp_qwen3-8B-length_penalty_global_ref" \
              "8B:length_penalty_trunc:${BASE}/ckpts/browsecomp_qwen3-8B-length_penalty_trunc" \
              "32B:baseline:${BASE}/ckpts/browsecomp_qwen3-32B-baseline" \
              "32B:length_penalty:${BASE}/ckpts/browsecomp_qwen3-32B-length_penalty"; do
    IFS=: read -r SIZE MODE DIR <<< "${spec}"
    for d in ${DIR}/iter_*; do
      [ -d "$d" ] || continue
      ITER=$((10#$(basename $d | cut -d_ -f2)))
      grep -q "^qwen3-${SIZE},${MODE},${ITER}," ${RESULTS}/results.csv && continue
      echo "${SIZE} ${MODE} ${ITER}"
    done
  done
}

WORK=$(worklist)
N=$(echo "${WORK}" | grep -c . || true)
echo "eval sweep: ${N} checkpoints to evaluate, ${PARALLEL} lanes"
[ "${N}" = "0" ] && exit 0

# Round-robin the work across PARALLEL sequential lanes.
i=0
declare -a LANES
while read -r SIZE MODE ITER; do
  LANES[$((i % PARALLEL))]+="${SIZE}:${MODE}:${ITER} "
  i=$((i+1))
done <<< "${WORK}"

for lane in $(seq 0 $((PARALLEL-1))); do
  (
    for item in ${LANES[$lane]:-}; do
      IFS=: read -r SIZE MODE ITER <<< "${item}"
      echo "[lane ${lane}] eval qwen3-${SIZE}/${MODE} iter ${ITER}"
      sbatch --wait -J bc-eval-${SIZE}-${MODE}-${ITER} \
        --export=ALL,MODEL_SIZE=${SIZE},MODE=${MODE},ITER=${ITER} \
        ${LAUNCH}/eval_ckpt.sbatch
      echo "[lane ${lane}] done qwen3-${SIZE}/${MODE} iter ${ITER} (exit $?)"
    done
  ) &
done
wait
echo "eval sweep complete:"
column -t -s, ${RESULTS}/results.csv
