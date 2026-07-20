#!/bin/bash
ulimit -n 65536
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray

pkill -u $(whoami) -f "ray_csq" 2>/dev/null
sleep 1
rm -rf /tmp/ray_csq
mkdir -p /tmp/ray_csq
TEMP_DIR="/tmp/ray_csq"
set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BROWSECOMP_DIR="$(cd -- "${SCRIPT_DIR}/../browsecomp" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"

# BrowseComp eval environment.
export LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL:?"export LOCAL_SEARCH_URL to the BrowseComp-Plus search server"}
if [ "${BROWSECOMP_EM_ONLY_REWARD:-0}" != "1" ]; then
   if [ -z "${GRADER_API_KEY:-${OPENAI_API_KEY:-}}" ]; then
      echo "export GRADER_API_KEY (or OPENAI_API_KEY) for the BrowseComp LLM judge"
      exit 1
   fi
fi
export GRADER_FALLBACK_MODEL=${GRADER_FALLBACK_MODEL:-${GRADER_MODEL:-}}
export BROWSECOMP_MAX_TURNS=${BROWSECOMP_MAX_TURNS:-100}
export BROWSECOMP_TURN_MAX_NEW_TOKENS=${BROWSECOMP_TURN_MAX_NEW_TOKENS:-2048}
export BROWSECOMP_MUST_SEARCH=${BROWSECOMP_MUST_SEARCH:-1}
export BROWSECOMP_ENABLE_THINKING=${BROWSECOMP_ENABLE_THINKING:-0}
export BROWSECOMP_SEARCH_MAX_TOPK=${BROWSECOMP_SEARCH_MAX_TOPK:-5}
export BROWSECOMP_SEARCH_SNIPPET_WORDS=${BROWSECOMP_SEARCH_SNIPPET_WORDS:-256}
export BROWSECOMP_OPEN_PAGE_WORDS=${BROWSECOMP_OPEN_PAGE_WORDS:-2048}
SGLANG_CTX_LEN=${SGLANG_CTX_LEN:-40960}
EVAL_RETOOL_MAX_RESPONSE_LEN=${EVAL_RETOOL_MAX_RESPONSE_LEN:-32768}
EVAL_BROWSECOMP_MAX_RESPONSE_LEN=${EVAL_BROWSECOMP_MAX_RESPONSE_LEN:-36864}
BROWSECOMP_MAX_SEQ_LEN=${BROWSECOMP_MAX_SEQ_LEN:-${EVAL_BROWSECOMP_MAX_RESPONSE_LEN}}

MODE=${MODE:-"one_step_off"}
FULLY_ASYNC_VERSION_WINDOW=${FULLY_ASYNC_VERSION_WINDOW:-1}
FULLY_ASYNC_MAX_COMPLETED_SAMPLES=${FULLY_ASYNC_MAX_COMPLETED_SAMPLES:-128}
FULLY_ASYNC_EVICTION_POLICY=${FULLY_ASYNC_EVICTION_POLICY:-"drop_oldest_version"}
FULLY_ASYNC_MAX_PARTIAL_SPAN=${FULLY_ASYNC_MAX_PARTIAL_SPAN:-3}
echo "=== Running hybrid async benchmark: mode=${MODE} ==="

CKPT_ARGS=(
   --hf-checkpoint /workspace/Qwen3-8B
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load /workspace/Qwen3-8B_torch_dist
   # --load /root/Qwen3-4B_slime/
#    --save /workspace/Qwen3-4B_sync_hybrid0.5/
#    --save-interval 100s
)
# Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200/iter399_torch_dist

WANDB_ARGS=(
   --use-wandb
   --wandb-project hybrid-qwen3-8b-eval
   --wandb-group Qwen3-8B-browsecomp-sft
   --wandb-key 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
)

PROMPT_SET=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rollout-seed 47
   
   --rm-type dapo
   --reward-key score

   --num-rollout 0
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192 # 8192&512 
   --rollout-temperature 1

   --global-batch-size 256
   --num-steps-per-rollout 1
   --balance-data
   --rollout-health-check-interval 600
   --rollout-health-check-timeout 600
   --save-debug-rollout-data /workspace/slime/examples/mixed/debug/eval/model-name-browsecomp.pt
)


EVAL_ARGS=(
   --eval-interval 10
   # All name/path pairs go under a single --eval-prompt-data (uses nargs='+')
   # --eval-prompt-data gsm8k /workspace/data/gsm8k/test.parquet \
   #                    aime25 /workspace/data/aime25/test.jsonl \
   #                    aime2024 /workspace/data/aime-2024/aime-2024.jsonl \
   #                    browsecomp /workspace/data/browsecomp/bc_test.jsonl
                      # dapo_math_17k /workspace/data/dapo-math-17k/dapo-math-17k.jsonl 
   # --eval-prompt-data gsm8k /workspace/data/gsm8k/test.parquet \
   #                    browsecomp /workspace/data/browsecomp/bc_test.jsonl
   --eval-prompt-data browsecomp /workspace/data/browsecomp/bc_test.jsonl
   # Per-dataset overrides (Dataset 1: gsm8k / competition math)
   # --eval-dataset-override gsm8k.n_samples_per_eval_prompt=1
   # --eval-dataset-override gsm8k.max_response_len=8192
   # --eval-dataset-override gsm8k.label_key=reward_model
   # --eval-dataset-override gsm8k.task_type=math
   # --eval-dataset-override gsm8k.eval_reward_key=acc
   # --eval-dataset-override gsm8k.label_sub_key=ground_truth
   # --eval-dataset-override gsm8k.wandb_prefix=eval1
   # Per-dataset overrides (Dataset 2: nq_test / search QA)
   # --eval-dataset-override nq_test.n_samples_per_eval_prompt=1
   # --eval-dataset-override nq_test.input_key=prompt
   # --eval-dataset-override nq_test.label_key=reward_model
   # --eval-dataset-override nq_test.wandb_prefix=eval2
   # --eval-dataset-override nq_test.task_type=qa
   # Per-dataset overrides (Dataset 3: browsecomp / factual QA with browsing)
   # --eval-dataset-override browsecomp.n_samples_per_eval_prompt=1
   # --eval-dataset-override browsecomp.input_key=prompt
   # --eval-dataset-override browsecomp.label_key=reward_model
   # --eval-dataset-override browsecomp.wandb_prefix=eval3
   # --eval-dataset-override browsecomp.task_type=qa
   # Per-dataset overrides (Dataset 7: BrowseComp / factual QA with browsing)
   --eval-dataset-override browsecomp.n_samples_per_eval_prompt=64
   --eval-dataset-override browsecomp.max_response_len=${EVAL_BROWSECOMP_MAX_RESPONSE_LEN}
   --eval-dataset-override browsecomp.input_key=prompt
   --eval-dataset-override browsecomp.label_key=label
   --eval-dataset-override browsecomp.metadata_key=metadata
   --eval-dataset-override browsecomp.task_type=qa
   --eval-dataset-override browsecomp.eval_reward_key=score
   --eval-dataset-override browsecomp.wandb_prefix=eval7
   # Per-dataset overrides (Dataset 4: aime / math)
   # --eval-dataset-override aime25.n_samples_per_eval_prompt=32
   # --eval-dataset-override aime25.max_response_len=32768
   # --eval-dataset-override aime25.input_key=problem
   # --eval-dataset-override aime25.label_key=answer
   # --eval-dataset-override aime25.task_type=math
   # --eval-dataset-override aime25.eval_reward_key=acc
   # --eval-dataset-override aime25.wandb_prefix=eval4
   # # Per-dataset overrides (Dataset 5: aime2024 / math)
   # --eval-dataset-override aime2024.n_samples_per_eval_prompt=32
   # --eval-dataset-override aime2024.max_response_len=32768
   # --eval-dataset-override aime2024.label_key=label
   # --eval-dataset-override aime2024.task_type=math
   # --eval-dataset-override aime2024.eval_reward_key=acc
   # --eval-dataset-override aime2024.wandb_prefix=eval5
   # Per-dataset overrides (Dataset 6: dapo_math_17k / math)
   # --eval-dataset-override dapo_math_17k.n_samples_per_eval_prompt=1
   # --eval-dataset-override dapo_math_17k.max_response_len=8192
   # --eval-dataset-override dapo_math_17k.input_key=prompt
   # --eval-dataset-override dapo_math_17k.label_key=label
   # --eval-dataset-override dapo_math_17k.task_type=math
   # --eval-dataset-override dapo_math_17k.eval_reward_key=acc
   # --eval-dataset-override dapo_math_17k.wandb_prefix=eval6
)


PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1 # 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 36864
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 16
   --sglang-router-disable-health-check
   --sglang-context-length ${SGLANG_CTX_LEN}
)

CUSTOM_ARGS=(
   --data-source-path custom_data_source.CustomDataSource
   --custom-generate-function-path generate_with_hybrid.generate_unified
   --custom-rm-path generate_with_hybrid.reward_func_unified
   --math-data-path /workspace/data/dapo-math-17k/dapo-math-17k.jsonl
   --qa-data-path /workspace/data/browsecomp/bc_test.jsonl
   --math-ratio 0
   # --batch-alternation \
   # --math-batches-per-cycle 200 \
   # --qa-batches-per-cycle 200 \
   # --dynamic-alternation \
   # --lag-version 3 \
   #  --grad-log-dir /workspace/slime/examples/hybrid/debug/math
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)


# single node
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
ray start --head --node-ip-address ${MASTER_ADDR} --temp-dir ${TEMP_DIR} --dashboard-port ${DASHBOARD_PORT} --num-gpus 8 --num-cpus 64 --disable-usage-stats

# Wait for the Ray dashboard and job agent to be fully ready
echo "Waiting for Ray agent to initialize..."
for i in {1..30}; do
  if curl -s http://${MASTER_ADDR}:${DASHBOARD_PORT}/api/jobs/ 1>/dev/null; then
    echo "Ray agent is up!"
    break
  fi
  sleep 1
done

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${BROWSECOMP_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
    \"GRADER_API_KEY\": \"${GRADER_API_KEY:-${OPENAI_API_KEY:-}}\",
    \"GRADER_BASE_URL\": \"${GRADER_BASE_URL:-}\",
    \"GRADER_MODEL\": \"${GRADER_MODEL:-}\",
    \"GRADER_FALLBACK_MODEL\": \"${GRADER_FALLBACK_MODEL:-}\",
    \"GRADER_API_VERSION\": \"${GRADER_API_VERSION:-}\",
    \"BROWSECOMP_MAX_TURNS\": \"${BROWSECOMP_MAX_TURNS}\",
    \"BROWSECOMP_TURN_MAX_NEW_TOKENS\": \"${BROWSECOMP_TURN_MAX_NEW_TOKENS}\",
    \"BROWSECOMP_MUST_SEARCH\": \"${BROWSECOMP_MUST_SEARCH}\",
    \"BROWSECOMP_ENABLE_THINKING\": \"${BROWSECOMP_ENABLE_THINKING}\",
    \"BROWSECOMP_SEARCH_MAX_TOPK\": \"${BROWSECOMP_SEARCH_MAX_TOPK}\",
    \"BROWSECOMP_SEARCH_SNIPPET_WORDS\": \"${BROWSECOMP_SEARCH_SNIPPET_WORDS}\",
    \"BROWSECOMP_OPEN_PAGE_WORDS\": \"${BROWSECOMP_OPEN_PAGE_WORDS}\",
    \"BROWSECOMP_MAX_SEQ_LEN\": \"${BROWSECOMP_MAX_SEQ_LEN}\",
    \"BROWSECOMP_EM_ONLY_REWARD\": \"${BROWSECOMP_EM_ONLY_REWARD:-0}\",
    \"MIXED_RETOOL_MAX_RESPONSE_LEN\": \"${EVAL_RETOOL_MAX_RESPONSE_LEN}\",
    \"MIXED_BROWSECOMP_MAX_RESPONSE_LEN\": \"${EVAL_BROWSECOMP_MAX_RESPONSE_LEN}\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"NCCL_DEBUG\": \"WARN\",
    \"NCCL_TIMEOUT\": \"1800000\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

# 提交作业到Ray集群
# 注意：Ray dashboard端口是8266，但job submission使用的是8265（默认）
ray job submit --address="http://${MASTER_ADDR}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}