#!/bin/bash
# Periodic health check for the 4 BrowseComp length-penalty experiment cells.
# Usage: bash health_check.sh   (prints a compact report; tracks log offsets
# in launch/.health_state so error greps only cover new content)

BASE=/data/home/syang
LOGD=${SLIME_LOGD:-${BASE}/CM/slime/logs}
STATE=${SLIME_REPO:-${BASE}/CM/slime}/examples/browsecomp/slurm_gb300/.health_state
mkdir -p ${STATE}

# job_id:cell:train_node:log_file
# (8B cells paused 2026-07-05: baseline froze at iter 179, length_penalty at
# iter 199; re-add entries here if they are resumed.)
CELLS=(
  "613:32B-baseline:rdx-gb300-r01-c010:${LOGD}/exp_bc-32B-base_613.log"
  "614:32B-length-penalty:rdx-gb300-r01-c014:${LOGD}/exp_bc-32B-lp_614.log"
  "700:8B-lp-globalref:rdx-gb300-r01-c006:${LOGD}/exp_bc-8B-lp-gref_700.log"
  "701:8B-lp-truncpen:rdx-gb300-r01-c018:${LOGD}/exp_bc-8B-lp-trunc_701.log"
)

echo "===== BrowseComp experiment health check: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
echo "--- slurm ---"
squeue -u $USER -o "%.6i %.14j %.2t %.10M %N"

for entry in "${CELLS[@]}"; do
  IFS=: read -r JOB CELL TNODE LOG <<< "$entry"
  echo "--- ${CELL} (job ${JOB}) ---"
  ST=$(squeue -h -j ${JOB} -o "%T" 2>/dev/null)
  echo "slurm_state: ${ST:-GONE}"

  # search server health
  H=$(timeout 10 curl -sf http://${TNODE}:8010/health 2>/dev/null)
  echo "search_server(${TNODE}): ${H:-UNREACHABLE}"

  if [ -f "${LOG}" ]; then
    SZ=$(stat -c %s ${LOG})
    OFFF=${STATE}/off_${JOB}
    OFF=$(cat ${OFFF} 2>/dev/null || echo 0)
    [ "${OFF}" -gt "${SZ}" ] && OFF=0
    NEW_ERRS=$(tail -c +$((OFF+1)) ${LOG} | grep -av "Finish rollout:" | grep -aE "CUDA out of memory|^Traceback|SYSTEM_ERROR|srun: error|Xid [0-9]+|ECC error|uncorrectable|NCCL (error|timeout)|Connection refused" | grep -av "avoid NVLS OOM" | head -5 | cut -c1-300)
    echo "${SZ}" > ${OFFF}
    if [ -n "${NEW_ERRS}" ]; then
      echo "NEW ERRORS:"; echo "${NEW_ERRS}"
    else
      echo "new_errors: none"
    fi
    # latest progress markers (step metrics / eval / penalty stats)
    LAST_METRIC=$(grep -aE "eval [0-9]+:|rollout_data_postprocess|browsecomp_length_penalty|perf/actor_train_time" ${LOG} | tail -1 | cut -c1-220)
    echo "last_metric: ${LAST_METRIC:-none yet}"
    echo "log_growth: $((SZ-OFF)) bytes since last check"
  else
    echo "log: MISSING"
  fi
done

echo "--- ckpt disk ---"
du -sh ${BASE}/ckpts/* 2>/dev/null
df -h /data | tail -1
echo "===== end ====="
