#!/bin/bash
ulimit -n 65536
# for rerun the task - 多节点环境下不要执行这些停止命令！
# 如果需要重启，请在各个节点上手动执行清理
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# 网络配置 - 与head节点保持一致
export NCCL_SOCKET_IFNAME=ibp169s0f1
export GLOO_SOCKET_IFNAME=ibp169s0f1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"

MODE=${MODE:-"one_step_off"}
FULLY_ASYNC_VERSION_WINDOW=${FULLY_ASYNC_VERSION_WINDOW:-1}
FULLY_ASYNC_MAX_COMPLETED_SAMPLES=${FULLY_ASYNC_MAX_COMPLETED_SAMPLES:-128}
FULLY_ASYNC_EVICTION_POLICY=${FULLY_ASYNC_EVICTION_POLICY:-"drop_oldest_version"}
FULLY_ASYNC_MAX_PARTIAL_SPAN=${FULLY_ASYNC_MAX_PARTIAL_SPAN:-3}
echo "=== Running hybrid async benchmark: mode=${MODE} ==="

CKPT_ARGS=(
   # --hf-checkpoint /workspace/DeepSeek-R1-Distill-Qwen-7B
   --hf-checkpoint /workspace/Qwen3-8B
   #--hf-checkpoint /root/Qwen3-8B-FP8
   # --ref-load /workspace/DeepSeek-R1-Distill-Qwen-7B_sft_torch_dist
   --ref-load /workspace/Qwen3-8B_sft_torch_dist
   # --load /root/Qwen3-8B_slime/
   # --save /workspace/Qwen3-8B_sync_hybrid0.5/
   # --save-interval 100
   --rotary-base 1000000
)
# Qwen3-4B_sync_qa Qwen3-4B_sync_math Qwen3-4B_sync_hybrid0.5 Qwen3-4B_sync_hybrid200-200

WANDB_ARGS=(
   --use-wandb
   --wandb-project hybrid-qwen3-8b-sync
   --wandb-group qwen3-8B-math
   --wandb-key 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
)

PROMPT_SET=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data /workspace/data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rollout-seed 42
   
   --rm-type dapo
   --reward-key score

   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192 # 8192&512 
   --rollout-temperature 1

   --global-batch-size 256
   --num-steps-per-rollout 1
   --balance-data
   --rollout-health-check-interval 30
   --rollout-health-check-timeout 30
#    --save-debug-rollout-data /workspace/slime/examples/hybrid/debug/ratio_0.5_8192_threshold/rollout_{rollout_id}.pt
)

EVAL_ARGS=(
   --eval-interval 10
   # All name/path pairs go under a single --eval-prompt-data (uses nargs='+')
   --eval-prompt-data gsm8k  /workspace/data/gsm8k/test.parquet@[0:64] \
                     nq_test /workspace/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:64]
   #--eval-prompt-data math500  /workspace/data/math500/math500_test.jsonl@[0:64]
   # Per-dataset overrides (Dataset 1: aime / math)
   --eval-dataset-override gsm8k.n_samples_per_eval_prompt=1
   --eval-dataset-override gsm8k.max_response_len=8192
   --eval-dataset-override gsm8k.label_key=reward_model
   --eval-dataset-override gsm8k.task_type=math
   --eval-dataset-override gsm8k.eval_reward_key=acc
   --eval-dataset-override gsm8k.label_sub_key=ground_truth
   # --eval-dataset-override gsm8k.top_p=1
   --eval-dataset-override gsm8k.wandb_prefix=eval1
   # --eval-dataset-override aime.n_samples_per_eval_prompt=8
   # --eval-dataset-override aime.max_response_len=16384
   # --eval-dataset-override aime.task_type=math
   # --eval-dataset-override aime.wandb_prefix=eval1
   # --eval-dataset-override math500.n_samples_per_eval_prompt=1
   # --eval-dataset-override math500.max_response_len=16384
   # --eval-dataset-override math500.input_key=problem
   # --eval-dataset-override math500.label_key=answer
   # --eval-dataset-override math500.task_type=math
   # --eval-dataset-override math500.wandb_prefix=eval1
   # Per-dataset overrides (Dataset 2: nq_test / search QA)
   --eval-dataset-override nq_test.n_samples_per_eval_prompt=1
   --eval-dataset-override nq_test.input_key=prompt
   --eval-dataset-override nq_test.label_key=reward_model
   --eval-dataset-override nq_test.wandb_prefix=eval2
   --eval-dataset-override nq_test.task_type=qa
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
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
   --max-tokens-per-gpu 9216
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
   --sglang-mem-fraction-static 0.5
)

CUSTOM_ARGS=(
   --data-source-path custom_data_source.CustomDataSource
   --custom-generate-function-path generate_with_hybrid.generate_unified
   # --custom-generate-function-path generate_retool.generate
   --custom-rm-path generate_with_hybrid.reward_func_unified
   --math-data-path /workspace/data/dapo-math-17k/dapo-math-17k.jsonl
   --qa-data-path /workspace/Search-R1/data/nq_hotpotqa_train/train.parquet
   --math-ratio 0.5
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

# 多节点配置 - 假设Ray已经在head节点启动
# 根据head节点启动脚本的配置
export MASTER_ADDR="${MASTER_ADDR:-10.0.1.171}"
export SLIME_HOST_IP="${MASTER_ADDR}"
export RAY_PORT="${RAY_PORT:-6382}"  # 注意：head节点使用的是6382端口

# 检查Ray集群状态 (ray status 需要指定 redis 端口而不是 http 控制台端口)
echo "Checking Ray cluster status at ${MASTER_ADDR}:${RAY_PORT}..."
export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
ray status || echo "Warning: Ray cluster may not be running properly"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"SLIME_HOST_IP\": \"${MASTER_ADDR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_SOCKET_IFNAME\": \"ibp169s0f1\",
    \"GLOO_SOCKET_IFNAME\": \"ibp169s0f1\",
    \"NCCL_IB_DISABLE\": \"0\",
    \"NCCL_IB_HCA\": \"mlx5_1\"
  }
}"

export RAY_DASHBOARD_PORT=8266
# 提交作业到Ray集群
# 注意：Ray dashboard端口是8266，但job submission使用的是8265（默认）
ray job submit --address="http://${MASTER_ADDR}:8266" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 8 \
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