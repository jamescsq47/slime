#!/bin/bash
# Runs INSIDE the slime container: one offline eval of one checkpoint.
# Usage: eval_inner.sh <SIZE> <MODE> <ITER> <ELOAD_DIR>
# Cleans up its own background processes on exit (no slurm/cgroup here).
set -x
SIZE=$1; MODE=$2; ITER=$3; ELOAD=$4
BASE=/data/home/syang
REPO=${SLIME_REPO:-${BASE}/CM/slime}

export HOME=${BASE}
export HF_HOME=${BASE}/.cache/huggingface
export PYTHONBUFFERED=16
source ${BASE}/.grader_env

cleanup() {
  ray stop --force 2>/dev/null
  pkill -9 -f search_server.py 2>/dev/null
  pkill -9 -f sglang 2>/dev/null
  sleep 2
}
trap cleanup EXIT

python3 ${REPO}/examples/browsecomp/search_server.py \
  --model ${BASE}/models/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --host 0.0.0.0 --port 8010 > /tmp/search_server.log 2>&1 &
until curl -sf http://localhost:8010/health > /dev/null; do sleep 10; done
export LOCAL_SEARCH_URL=http://localhost:8010

source ${REPO}/scripts/models/qwen3-${SIZE}.sh
SGLANG_CTX_LEN=40960
MAX_SEQ_LEN=$((SGLANG_CTX_LEN - 4096))

# cd BEFORE ray start: without pyxis --container-workdir the ray job
# entrypoint inherits the head's cwd, and train.py lives in the repo root.
cd ${REPO}
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${REPO}/examples/browsecomp\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
    \"GRADER_API_KEY\": \"${GRADER_API_KEY}\",
    \"GRADER_BASE_URL\": \"${GRADER_BASE_URL:-}\",
    \"GRADER_MODEL\": \"${GRADER_MODEL:-}\",
    \"BROWSECOMP_MAX_TURNS\": \"100\",
    \"BROWSECOMP_TURN_MAX_NEW_TOKENS\": \"2048\",
    \"BROWSECOMP_MAX_SEQ_LEN\": \"\${MAX_SEQ_LEN}\",
    \"BROWSECOMP_MUST_SEARCH\": \"1\",
    \"HOME\": \"${BASE}\"
  }
}"

ray job submit --address=http://127.0.0.1:8265 \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${REPO}/train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 4 --colocate \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint ${BASE}/models/Qwen3-${SIZE} \
   --ref-load ${BASE}/models/Qwen3-${SIZE}_torch_dist \
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
   --n-samples-per-eval-prompt 1 --eval-temperature 0.6 \
   --tensor-model-parallel-size 4 --sequence-parallel \
   --pipeline-model-parallel-size 1 --context-parallel-size 1 \
   --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
   --use-dynamic-batch-size --max-tokens-per-gpu 40960 \
   --advantage-estimator grpo \
   --optimizer adam --lr 1e-6 --lr-decay-style constant \
   --rollout-num-gpus-per-engine 4 --sglang-mem-fraction-static 0.7 \
   --sglang-context-length ${SGLANG_CTX_LEN} \
   --attention-dropout 0.0 --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
   --attention-backend flash \
   --custom-generate-function-path browsecomp_agent.generate \
   --custom-rm-path browsecomp_rm.reward_func \
