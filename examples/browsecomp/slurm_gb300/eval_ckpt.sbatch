#!/bin/bash
# Offline eval of one BrowseComp checkpoint on the 150-question test set.
# Submit with:
#   sbatch -J bc-eval-<cell>-<iter> \
#     --export=ALL,MODEL_SIZE=8B,MODE=baseline,ITER=19 eval_ckpt.sbatch
# MODE selects the ckpt dir: baseline-8B uses browsecomp_qwen3-8B-async
# (the original 609 run); others use browsecomp_qwen3-<size>-<mode>.
# Appends one CSV row to launch/eval_results/results.csv on success.
#SBATCH -A customer
#SBATCH -p batch_1
#SBATCH --qos=batch_1_qos
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -t 03:00:00
#SBATCH -o /data/home/syang/CM/slime/logs/eval_%x_%j.log

set -x
MODEL_SIZE=${MODEL_SIZE:?}; MODE=${MODE:?}; ITER=${ITER:?}

BASE=/data/home/syang
REPO=${SLIME_REPO:-${BASE}/CM/slime}
LAUNCH=${SLIME_REPO:-${BASE}/CM/slime}/examples/browsecomp/slurm_gb300
IMG=${SLIME_IMAGE:-${BASE}/images/slime-dev.sqsh}
LOGD=${SLIME_LOGD:-${BASE}/CM/slime/logs}
RESULTS=${LAUNCH}/eval_results
mkdir -p ${RESULTS}

# Map cell -> checkpoint dir (8B baseline was trained as job 609 into -async).
if [ "${MODEL_SIZE}" = "8B" ] && [ "${MODE}" = "baseline" ]; then
  CKPT_DIR=${BASE}/ckpts/browsecomp_qwen3-8B-async
else
  CKPT_DIR=${BASE}/ckpts/browsecomp_qwen3-${MODEL_SIZE}-${MODE}
fi
ITER_DIR=$(printf "iter_%07d" ${ITER})
[ -d "${CKPT_DIR}/${ITER_DIR}" ] || { echo "missing ${CKPT_DIR}/${ITER_DIR}"; exit 1; }

# Eval-load dir pinned to this iteration.
ELOAD=${BASE}/ckpts/eval_tmp/${MODEL_SIZE}-${MODE}-${ITER}
mkdir -p ${ELOAD}
ln -sfn ${CKPT_DIR}/${ITER_DIR} ${ELOAD}/${ITER_DIR}
echo ${ITER} > ${ELOAD}/latest_checkpointed_iteration.txt

srun -N1 -n1 --gres=gpu:4 \
     --container-image=${IMG} \
     --container-mounts=/data:/data \
     --container-workdir=${REPO} \
     bash -c "
set -x
export HOME=${BASE}
export HF_HOME=${BASE}/.cache/huggingface
source ${BASE}/.grader_env
export PYTHONBUFFERED=16

python3 ${REPO}/examples/browsecomp/search_server.py \
  --model ${BASE}/models/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --host 0.0.0.0 --port 8010 > ${LOGD}/search_server_eval_\${SLURM_JOB_ID}.log 2>&1 &
until curl -sf http://localhost:8010/health > /dev/null; do sleep 10; done
export LOCAL_SEARCH_URL=http://localhost:8010

source ${REPO}/scripts/models/qwen3-${MODEL_SIZE}.sh
SGLANG_CTX_LEN=40960
MAX_SEQ_LEN=\$((SGLANG_CTX_LEN - 4096))

ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats

RUNTIME_ENV_JSON=\"{
  \\\"env_vars\\\": {
    \\\"PYTHONPATH\\\": \\\"/root/Megatron-LM/:${REPO}/examples/browsecomp\\\",
    \\\"CUDA_DEVICE_MAX_CONNECTIONS\\\": \\\"1\\\",
    \\\"LOCAL_SEARCH_URL\\\": \\\"\${LOCAL_SEARCH_URL}\\\",
    \\\"GRADER_API_KEY\\\": \\\"\${GRADER_API_KEY}\\\",
    \\\"GRADER_BASE_URL\\\": \\\"\${GRADER_BASE_URL:-}\\\",
    \\\"GRADER_MODEL\\\": \\\"\${GRADER_MODEL:-}\\\",
    \\\"BROWSECOMP_MAX_TURNS\\\": \\\"100\\\",
    \\\"BROWSECOMP_TURN_MAX_NEW_TOKENS\\\": \\\"2048\\\",
    \\\"BROWSECOMP_MUST_SEARCH\\\": \\\"1\\\",
    \\\"HOME\\\": \\\"${BASE}\\\"
  }
}\"

cd ${REPO}
ray job submit --address=http://127.0.0.1:8265 \
   --runtime-env-json=\"\${RUNTIME_ENV_JSON}\" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   \${MODEL_ARGS[@]} \
   --hf-checkpoint ${BASE}/models/Qwen3-${MODEL_SIZE} \
   --ref-load ${BASE}/models/Qwen3-${MODEL_SIZE}_torch_dist \
   --load ${ELOAD} --no-load-optim --no-load-rng \
   --num-rollout 0 \
   --lr-decay-iters 10 \
   --prompt-data ${REPO}/examples/browsecomp/data/bc_train.jsonl \
   --input-key prompt --label-key label --metadata-key metadata \
   --rollout-batch-size 32 --n-samples-per-prompt 8 \
   --rollout-max-response-len 32768 --rollout-temperature 1 \
   --global-batch-size 256 \
   --eval-interval 1 \
   --eval-prompt-data browsecomp ${REPO}/examples/browsecomp/data/bc_test.jsonl \
   --n-samples-per-eval-prompt 1 \
   --eval-temperature 0.6 \
   --tensor-model-parallel-size 4 --sequence-parallel \
   --pipeline-model-parallel-size 1 --context-parallel-size 1 \
   --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
   --use-dynamic-batch-size --max-tokens-per-gpu 40960 \
   --advantage-estimator grpo \
   --optimizer adam --lr 1e-6 --lr-decay-style constant \
   --rollout-num-gpus-per-engine 4 --sglang-mem-fraction-static 0.7 \
   --sglang-context-length \${SGLANG_CTX_LEN} \
   --attention-dropout 0.0 --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
   --attention-backend flash \
   --custom-generate-function-path browsecomp_agent.generate \
   --custom-rm-path browsecomp_rm.reward_func \
"

# Parse the eval metric line and append to CSV. set +x first: the xtrace of
# these greps lands in the SAME log file and would match its own pattern.
set +x
LOG=$(ls ${LOGD}/eval_*_${SLURM_JOB_ID}.log 2>/dev/null | head -1)
LINE=$(grep -a "eval 0: {" ${LOG} | grep -a "'eval/browsecomp':" | tail -1)
ACC=$(echo "${LINE}" | grep -oE "'eval/browsecomp': [0-9.]+" | grep -oE "[0-9.]+$")
RLEN=$(echo "${LINE}" | grep -oE "'eval/browsecomp/response_len/mean': [0-9.]+" | grep -oE "[0-9.]+$")
TRUNC=$(echo "${LINE}" | grep -oE "'eval/browsecomp-truncated_ratio': [0-9.]+" | grep -oE "[0-9.]+$")
if [ -n "${ACC}" ]; then
  echo "qwen3-${MODEL_SIZE},${MODE},${ITER},${ACC},${RLEN},${TRUNC},$(date -u +%FT%TZ)" >> ${RESULTS}/results.csv
  echo "EVAL_OK model=qwen3-${MODEL_SIZE} mode=${MODE} iter=${ITER} acc=${ACC} resp_len=${RLEN}"
else
  echo "EVAL_FAILED_TO_PARSE model=qwen3-${MODEL_SIZE} mode=${MODE} iter=${ITER}"
  exit 1
fi
