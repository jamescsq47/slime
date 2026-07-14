#!/bin/bash

# for rerun the task
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

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "/root/slime/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint /workspace/Qwen3-8B
   --ref-load /workspace/Qwen3-8B_sft_torch_dist
   # --load /root/Qwen3-4B_slime/
   # --save /root/font-info/qwen3-4b-sft/qwen3-4b-sft-multi-turn/
   # --save /workspace/Qwen3-8B_sync_hybrid0.5/
   --rotary-base 1000000
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project hybrid-qwen3-8b-sync
   --wandb-group qwen3-8B-math-fix
   --wandb-key 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
)

ROLLOUT_ARGS=(
   --prompt-data /workspace/data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
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
   --recompute-num-layers 1

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

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

# launch the master node of ray in container
# export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

export MASTER_ADDR="${MASTER_ADDR:-10.0.1.171}"
export RAY_PORT="${RAY_PORT:-6382}"  # 注意：head节点使用的是6382端口

# 检查Ray集群状态 (ray status 需要指定 redis 端口而不是 http 控制台端口)
echo "Checking Ray cluster status at ${MASTER_ADDR}:${RAY_PORT}..."
export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
ray status || echo "Warning: Ray cluster may not be running properly"


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
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