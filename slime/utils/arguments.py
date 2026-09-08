import argparse
import json
import logging
import os
import re
from typing import Any

import yaml
from sglang_router.launch_router import RouterArgs

from slime.backends.sglang_utils.arguments import sglang_parse_args
from slime.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from slime.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from slime.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_slime_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )
            parser.add_argument(
                "--critic-num-nodes", type=int, default=None, help="Number of nodes for training actor"
            )
            parser.add_argument(
                "--critic-num-gpus-per-node", type=int, default=None, help="Number of gpus per node for training actor"
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser

        def add_train_arguments(parser):
            # --train-backend is parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--qkv-format",
                type=str,
                choices=["thd", "bshd"],
                default="thd",
                help="The qkv layout for Megatron backend.",
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            parser.add_argument(
                "--disable-weights-backuper",
                action="store_false",
                dest="enable_weights_backuper",
                help="Whether to disable weights backuper to save host memory.",
            )
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="The method to convert megatron weights to hugging face weights for SGLang.",
            )
            parser.add_argument(
                "--custom-model-provider-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom model provider function. "
                    "If set, we will use this function instead of the default model provider. "
                    "The function should have the signature "
                    "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                    "Example: 'my_module.my_model_provider'."
                ),
            )
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Whether to disable recompute loss function to save memory during training.",
            )
            parser.add_argument(
                "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory"
            )
            parser.add_argument(
                "--only-train-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to TRAIN. All other parameters will be FROZEN. 
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Train ONLY MoE experts:
                            --only-train-params-name-list experts

                        2. Train ONLY Indexer parameters:
                            --only-train-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Train ONLY Layer 20 to 23:
                            --only-train-params-name-list layers\.2[0-3]\.
                        """,
            )

            parser.add_argument(
                "--freeze-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to FREEZE. Other parameters will remain trainable.
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Freeze Embeddings and Output Layer (common for fine-tuning):
                            --freeze-params-name-list embedding output_layer

                        2. Freeze Indexer parameters:
                            --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Freeze specific projection layers (e.g., all Gate/Up projections):
                            --freeze-params-name-list linear_fc1
                        """,
            )
            parser.add_argument(
                "--allgather-cp",
                action="store_true",
                default=False,
            )

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `status`."
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout."
                    "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
                ),
            )
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the prompt for the inference engine during rollout. "
                    "If set, we will filter out the long prompts during initialization of the global dataset. "
                    "This is not recommended if the dataset is large."
                ),
            )
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the response for the inference engine during rollout. "
                    "It is basically `max_tokens` in sglang."
                ),
            )
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Whether to skip special tokens in the response during rollout. "
                    "This is useful when you want to use the response as a prompt for the next rollout."
                ),
            )
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "The stop words for the inference engine during rollout. "
                    "It can be a list of strings or a single string. "
                    "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
                ),
            )
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "The stop token ids for the inference engine during rollout. "
                    "It can be a list of integers or a single integer."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )
            parser.add_argument(
                "--train-batch-math-groups",
                type=int,
                default=None,
                help=(
                    "Optional exact number of completed math groups admitted to each training batch. "
                    "Must be used together with --train-batch-qa-groups, and their sum must equal "
                    "rollout_batch_size. Completed groups above either quota are recycled to the rollout buffer."
                ),
            )
            parser.add_argument(
                "--train-batch-qa-groups",
                type=int,
                default=None,
                help=(
                    "Optional exact number of completed QA groups admitted to each training batch. "
                    "Must be used together with --train-batch-math-groups."
                ),
            )

            # partial rollout
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to use partial rollout. "
                    "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                    "This is useful for long responses."
                ),
            )
            parser.add_argument(
                "--partial-rollout-tool-handoff",
                action="store_true",
                default=False,
                help=(
                    "Allow unfinished custom rollout tasks that are waiting in external tools to finish outside "
                    "the rollout abort barrier. Their partial groups are recycled after the tool returns. "
                    "Custom generate functions must observe GenerateState.abort_epoch before issuing another "
                    "generation request. Requires --partial-rollout."
                ),
            )
            parser.add_argument(
                "--rollout-replenish-completed-groups",
                action="store_true",
                default=False,
                help=(
                    "Keep the over-sampling group window full by submitting one replacement whenever a group "
                    "finishes, until rollout_batch_size groups have been collected."
                ),
            )
            parser.add_argument(
                "--rollout-sample-completion-backfill",
                action="store_true",
                default=False,
                help=(
                    "Replenish one new prompt group after n_samples_per_prompt individual sample slots complete, "
                    "instead of waiting for one whole group to finish. Disabled by default so the existing "
                    "group-level rollout scheduler is unchanged."
                ),
            )
            parser.add_argument(
                "--decoupled-gpu-tool-scheduling",
                action="store_true",
                default=False,
                help=(
                    "Limit actual SGLang /generate requests with a request-level GPU slot pool. "
                    "Tool work uses no GPU slot; completed tools resume in FIFO order before fresh samples."
                ),
            )
            parser.add_argument(
                "--gpu-generation-slots",
                type=int,
                default=None,
                help=(
                    "Request-level SGLang concurrency for decoupled GPU/tool scheduling. "
                    "Defaults to rollout_batch_size * n_samples_per_prompt."
                ),
            )
            parser.add_argument(
                "--terminal-live-session-limit",
                type=int,
                default=None,
                help=(
                    "Maximum number of persistent Terminal-Bench sessions held by rollout samples. "
                    "Excess samples wait FIFO instead of opening another environment."
                ),
            )
            parser.add_argument(
                "--terminal-concurrent-resets",
                type=int,
                default=None,
                help=(
                    "Maximum number of Terminal-Bench reset/container-creation operations in flight. "
                    "This is independent of SGLang generation slots."
                ),
            )
            parser.add_argument(
                "--max-inflight-groups",
                type=int,
                default=None,
                help=(
                    "Hard cap on active plus weight-update-deferred rollout groups. "
                    "Used as a safety bound for decoupled GPU/tool scheduling."
                ),
            )
            parser.add_argument(
                "--inflight-group-soft-limit",
                type=int,
                default=None,
                help=(
                    "Preferred inflight-group limit for decoupled scheduling. When all prefetched "
                    "samples are activated but GPU slots still need fresh supply, the scheduler may "
                    "open one additional group container at a time up to --max-inflight-groups; "
                    "sample activation remains exact-demand."
                ),
            )
            parser.add_argument(
                "--adaptive-group-oversampling",
                "--elastic-group-window",
                dest="adaptive_group_oversampling",
                action="store_true",
                default=False,
                help=(
                    "Keep rollout_batch_size as the collection size while dynamically adjusting an "
                    "independent inflight prompt-group target between configured minimum and maximum "
                    "bounds from smoothed SGLang pressure."
                ),
            )
            parser.add_argument(
                "--adaptive-group-oversampling-min-groups",
                type=int,
                default=None,
                help=(
                    "Minimum elastic inflight group target. Defaults to rollout_batch_size; set below "
                    "it to allow natural draining under sustained pressure."
                ),
            )
            parser.add_argument(
                "--adaptive-group-oversampling-running-threshold",
                type=float,
                default=180.0,
                help="Increase group-level oversampling while total SGLang running requests stay below this value.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-queue-threshold",
                type=float,
                default=4.0,
                help="Only increase group-level oversampling while total SGLang queued requests stay below this value.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-expansion-kv-threshold",
                type=float,
                default=0.65,
                help="Only expand while smoothed average KV-cache usage stays below this value.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-window-seconds",
                type=float,
                default=15.0,
                help="Sliding-window duration used to smooth SGLang pressure metrics.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-cooldown-seconds",
                type=float,
                default=15.0,
                help="Minimum time between elastic group-window adjustments.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-post-resume-expansion-grace-seconds",
                type=float,
                default=30.0,
                help=(
                    "After rollout resumes from a weight update, suppress expansion for this many "
                    "seconds while still allowing pressure-driven contraction."
                ),
            )
            parser.add_argument(
                "--adaptive-group-oversampling-recovery-seconds",
                type=float,
                default=10.0,
                help=(
                    "Seconds of sustained queued-request and KV-cache pressure before reducing "
                    "the inflight group target."
                ),
            )
            parser.add_argument(
                "--adaptive-group-oversampling-step-groups",
                type=int,
                default=2,
                help="Number of whole prompt groups added or removed per adaptive adjustment.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-pressure-queue-threshold",
                type=float,
                default=12.0,
                help=(
                    "Global queued-request threshold used by soft pressure and required alongside "
                    "the average-KV hard-pressure condition."
                ),
            )
            parser.add_argument(
                "--adaptive-group-oversampling-pressure-kv-threshold",
                type=float,
                default=0.65,
                help="Soft-pressure average KV-cache threshold for reducing adaptive oversampling.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-hard-max-engine-kv-threshold",
                type=float,
                default=0.95,
                help="Hard-pressure KV-cache threshold for the most occupied SGLang engine.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-hard-engine-queue-threshold",
                type=float,
                default=4.0,
                help="Required local queue on the max-KV engine for max-engine hard pressure.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-hard-pressure-kv-threshold",
                type=float,
                default=0.78,
                help=(
                    "Hard-pressure average KV-cache threshold for faster oversampling reduction; "
                    "the global queued-request pressure threshold must also be exceeded."
                ),
            )
            parser.add_argument(
                "--adaptive-group-oversampling-hard-pressure-seconds",
                type=float,
                default=5.0,
                help="Seconds of sustained hard pressure before reducing the inflight group target.",
            )
            parser.add_argument(
                "--adaptive-group-oversampling-hard-step-groups",
                type=int,
                default=4,
                help="Number of whole prompt groups removed per hard-pressure adjustment.",
            )
            parser.add_argument(
                "--mask-offpolicy-in-partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to mask previous generation in partial rollout. "
                    "If set, only on-policy generated tokens will be used in training"
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-math",
                type=int,
                default=None,
                help=(
                    "Mask off-policy tokens in math samples when lag_sample_math >= this value. "
                    "lag_sample_math is the total math samples across the lagging versions. "
                    "Requires partial_rollout to be enabled."
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-qa",
                type=int,
                default=None,
                help=(
                    "Mask off-policy tokens in QA samples when lag_sample_qa >= this value. "
                    "lag_sample_qa is the total QA samples across the lagging versions. "
                    "Requires partial_rollout to be enabled."
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-terminal",
                type=int,
                default=None,
                help=(
                    "Mask off-policy tokens in Terminal-Bench samples when lag_sample_terminal "
                    "reaches this value. Requires partial_rollout."
                ),
            )
            parser.add_argument(
                "--enable-tool-delay",
                action="store_true",
                default=False,
                help="Enable artificial async delay after each math/search tool call.",
            )
            parser.add_argument(
                "--tool-delay-mean",
                type=float,
                default=25.0,
                help="Mean in seconds for the artificial log-normal tool delay.",
            )
            parser.add_argument(
                "--tool-delay-variance",
                type=float,
                default=500.0,
                help="Variance in seconds^2 for the artificial log-normal tool delay.",
            )
            parser.add_argument(
                "--tool-delay-check-interval",
                type=float,
                default=0.5,
                help="Polling interval in seconds for making artificial tool delay abort-friendly.",
            )
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging rollout data. The signature of the functions is: "
                    "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging eval rollout data. "
                    "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
            parser.add_argument(
                "--staleness-threshold",
                type=float,
                default=None,
                help=(
                    "Maximum stale backlog ratio for fully async rollout. "
                    "When set, fully async rollout workers will pause scheduling new samples once the "
                    "number of stale samples reaches approximately "
                    "`rollout_batch_size * update_weights_interval * (1 + staleness_threshold)`."
                ),
            )
            parser.add_argument(
                "--fully-async-debug-version-tracking",
                action="store_true",
                default=False,
                help=(
                    "Print per-batch version summaries for fully async rollout consumption. "
                    "Useful for debugging why stale backlog exists while stale processed counters stay at zero."
                ),
            )
            parser.add_argument(
                "--fully-async-buffer-policy",
                type=str,
                choices=["legacy_backpressure", "window_evict"],
                default="legacy_backpressure",
                help=(
                    "Completed-sample buffering policy for fully async rollout. "
                    "`legacy_backpressure` keeps the current stale-budget pause behavior, while "
                    "`window_evict` keeps rollout scheduling active and evicts completed samples that fall "
                    "outside the configured version window."
                ),
            )
            parser.add_argument(
                "--fully-async-version-window",
                type=int,
                default=1,
                help=(
                    "Maximum policy-version distance allowed in the completed-sample window when "
                    "`--fully-async-buffer-policy=window_evict`."
                ),
            )
            parser.add_argument(
                "--fully-async-max-completed-samples",
                type=int,
                default=None,
                help=(
                    "Hard cap on completed fully async samples kept in memory. Defaults to the legacy queue sizing "
                    "heuristic when unset."
                ),
            )
            parser.add_argument(
                "--fully-async-max-partial-span",
                type=int,
                default=None,
                help="Max partial span for window evict in fully async",
            )
            parser.add_argument(
                "--fully-async-eviction-policy",
                type=str,
                choices=["drop_oldest_version", "drop_oldest_fifo"],
                default="drop_oldest_version",
                help=(
                    "Overflow eviction policy for fully async completed samples when "
                    "`--fully-async-buffer-policy=window_evict`."
                ),
            )
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help="Whether to keep the rollout model on training process",
            )

            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "The called after we have all the rollout data including log_probs. "
                    "It may be helpful for updating loss mask."
                ),
            )
            parser.add_argument(
                "--rollout-external",
                action="store_true",
                default=False,
                help="Use external SGLang instances instead of launching them inside the framework.",
            )
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Whether to enable the fault tolerance function during rollout.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health_generate checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health_generate response before killing it.",
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                    "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="slime.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively. "
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
                ),
            )
            
            parser.add_argument(
                "--math-data-path",
                type=str,
                default=None,
                help=(
                    "Path to math training data (JSONL format). "
                    "Used with CustomDataSource for unified data loading. "
                    "Each line should contain 'prompt' and 'label' fields."
                ),
            )
            parser.add_argument(
                "--qa-data-path",
                type=str,
                default=None,
                help=(
                    "Path to QA/search training data (Parquet format). "
                    "Used with CustomDataSource for unified data loading. "
                    "Should contain 'prompt', 'reward_model', 'golden_answers', and other metadata fields."
                ),
            )
            parser.add_argument(
                "--terminal-data-path",
                type=str,
                default=None,
                help=(
                    "Path to Terminal-Bench prompt data (JSONL). Each row contains a prompt "
                    "and metadata.task_id; the local environment supplies the task instruction."
                ),
            )
            parser.add_argument(
                "--math-ratio",
                type=float,
                default=0.7,
                help=(
                    "Ratio of math samples to total samples when mixing math and QA data. "
                    "Valid range: [0.0, 1.0]. Default: 0.7 means 70% math and 30% QA."
                ),
            )
            parser.add_argument(
                "--terminal-ratio",
                type=float,
                default=0.0,
                help=(
                    "Absolute Terminal-Bench fraction in normal mixed dispatch. "
                    "QA receives 1 - math_ratio - terminal_ratio."
                ),
            )
            parser.add_argument(
                "--token-load-adaptive-reordering",
                action="store_true",
                default=False,
                help=(
                    "After each collected rollout batch, compare its full train-sequence token count "
                    "with the previous 16-batch moving average and reorder the next rollout_batch_size "
                    "normal mixed prompts: Terminal first above 1.1x, QA first below 0.9x, random "
                    "otherwise. The first 8 batches stay random."
                ),
            )
            parser.add_argument('--batch-alternation', action='store_true', 
                                help='Enable batch-level alternation mode')
            parser.add_argument('--math-batches-per-cycle', type=int, default=1,
                                help='Number of math batches per cycle in alternation mode')
            parser.add_argument('--qa-batches-per-cycle', type=int, default=1,
                                help='Number of QA batches per cycle in alternation mode')
            parser.add_argument('--terminal-batches-per-cycle', type=int, default=0,
                                help='Number of Terminal-Bench batches per cycle in alternation mode')
            parser.add_argument(
                '--batch-alternation-order',
                type=int,
                default=None,
                choices=range(1, 7),
                help=(
                    'Three-domain dispatch order: '
                    '1=math,qa,terminal; 2=math,terminal,qa; '
                    '3=qa,math,terminal; 4=qa,terminal,math; '
                    '5=terminal,math,qa; 6=terminal,qa,math.'
                ),
            )
            parser.add_argument(
                '--batch-alternation-start-task',
                type=str,
                default=None,
                choices=['math', 'qa', 'terminal'],
                help=(
                    'Deprecated compatibility option for two-domain-style rotation. '
                    'Use --batch-alternation-order for an explicit three-domain order.'
                ),
            )
            parser.add_argument('--phase-aware-alternation', action='store_true',
                                help='Enable phase-aware alternation: prefer QA during training and math after policy update.')
            parser.add_argument('--phase-aware-train-task', type=str, default='qa', choices=['math', 'qa'],
                                help='Task type preferred while training is in progress in phase-aware alternation.')
            parser.add_argument('--phase-aware-post-update-task', type=str, default='math', choices=['math', 'qa'],
                                help='Task type preferred after policy update in phase-aware alternation.')
            parser.add_argument(
                '--count-aware-alternation',
                action='store_true',
                help=(
                    'Select the phase-aware direction from the previous training composition: '
                    'math-heavy selects train=qa/post-update=math; QA-heavy selects '
                    'train=math/post-update=qa, while preserving per-cycle task quotas.'
                ),
            )

            parser.add_argument('--dynamic-alternation', action='store_true', 
                                help='Enable dynamic alternation mode based on policy version and in-flight tasks')
            parser.add_argument('--lag-version', type=int, default=5,
                                help='Lag version tolerance for dynamic alternation mode')
            parser.add_argument('--dynamic-alternation-alpha', type=float, default=0.5,
                                help='Weight of the lag-based ratio in dynamic alternation. 0 keeps math-ratio, 1 uses pure lag-based ratio.')
            parser.add_argument('--dynamic-alternation-warmup-steps', type=int, default=5,
                                help='Number of initial policy versions that use math-ratio before enabling lag-based dynamic alternation.')
            parser.add_argument('--dynamic-alternation-min-math-ratio', type=float, default=0.3,
                                help='Lower bound for math dispatch probability in dynamic alternation.')
            parser.add_argument('--dynamic-alternation-max-math-ratio', type=float, default=0.7,
                                help='Upper bound for math dispatch probability in dynamic alternation.')

            parser.add_argument("--apply-chat-template", action="store_true", default=False)
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default="{}")
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            parser.add_argument("--label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--multimodal-keys",
                type=json.loads,
                default=None,
                help=(
                    'JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''
                ),
            )
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")
            parser.add_argument(
                "--tool-key",
                type=str,
                default="tools",
                help=(
                    "When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
                ),
            )

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Balance the number of tokens between data parallel ranks with `karmarkar_karp` for verl. "
                    "Note that this may allocate the different response of the same prompt into different training steps."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, slime will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                    "`max_response_len // cp_size` instead of `max_response_len`."
                ),
            )
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. "
                    "This is used to calculate the log probs of the responses during rollout, "
                    "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
                ),
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-tool-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            parser.add_argument("--eval-temperature", type=float, default=None)
            parser.add_argument("--eval-top-p", type=float, default=None)
            parser.add_argument("--eval-top-k", type=int, default=None)
            parser.add_argument("--eval-max-response-len", type=int, default=None)
            parser.add_argument("--eval-max-prompt-len", type=int, default=None)
            parser.add_argument("--eval-min-new-tokens", type=int, default=None)
            parser.add_argument("--eval-max-context-len", type=int, default=None)
            parser.add_argument(
                "--eval-dataset-override",
                type=str,
                default=None,
                action="append",
                help=(
                    "Per-dataset eval config overrides in the format 'name.key=value'. "
                    "Repeatable for multiple overrides across datasets. "
                    "Example: --eval-dataset-override aime.n_samples_per_eval_prompt=16 "
                    "--eval-dataset-override nq_test.wandb_prefix=eval2"
                ),
            )

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "
            )
            reset_arg(parser, "--load", type=str, default=None)
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            reset_arg(parser, "--async-save", action="store_true")
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            parser.add_argument(
                "--save-hf",
                type=str,
                default=None,
                help=(
                    "Path to save the model in HuggingFace format when using Megatron backend. "
                    "The model will be saved to `save_hf.format(rollout_id)`. "
                ),
            )
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            reset_arg(parser, "--lr", type=float, default=1e-6)

            parser.add_argument("--num-critic-only-steps", type=int, default=0, help="Number of critic only steps")
            parser.add_argument("--critic-load", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-save", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-lr", type=float, default=None, help="The lr for critic model")
            parser.add_argument("--critic-train-only", action="store_true", default=False, help="Only train critic")
            parser.add_argument(
                "--critic-lr-warmup-iters",
                type=int,
                default=0,
                help="number of iterations to linearly warmup for critic model.",
            )

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
            )
            parser.add_argument("--value-clip", type=float, default=0.2, help="the clip for value loss")
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
            )
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="k1",
                help="Choose KL loss type: kl, k2, k3, low_var_kl",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                ],
                default="grpo",
                help=(
                    "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
                ),
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            parser.add_argument(
                "--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
            )
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Whether to enable unbiased KL estimation.",
            )
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
            parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable rewards normalization",
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help="Whether to calculate the mismatch metrics.",
            )
            parser.add_argument(
                "--reset-optimizer-states",
                action="store_true",
                default=False,
                help=(
                    "Whether to reset optimizer states after each rollout. "
                    "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
                ),
            )
            parser.add_argument(
                "--use-rollout-logprobs",
                action="store_true",
                default=False,
                help=(
                    "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                    "If not set, we will use the logprobs from the actor model."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl for off-policy importance sampling.",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to the custom TIS/RS function (e.g., examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
            )
            parser.add_argument(
                "--custom-pg-loss-reducer-function-path",
                type=str,
                default=None,
                help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean. (e.g., examples/Dr.GRPO/custom_reducer.py:get_pg_loss_reducer).",
            )

            parser.add_argument(
                "--use-routing-replay",
                action="store_true",
                default=False,
                help="The routing replay technique from https://arxiv.org/abs/2507.18071",
            )
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help="The rollout routing replay technique from https://arxiv.org/abs/2510.11370",
            )
            parser.add_argument(
                "--use-opsm",
                action="store_true",
                default=False,
                help="Whether to enable Off-Policy Sequence Masking (OPSM).",
            )
            parser.add_argument(
                "--opsm-delta",
                type=float,
                default=1e-4,
                help="The threshold for Off-Policy Sequence Masking (OPSM).",
            )
            return parser

        def add_on_policy_distillation_arguments(parser):
            """Add on-policy distillation (OPD) related arguments.

            OPD is orthogonal to advantage estimators and can be applied on top of
            any estimator (GRPO, PPO, etc.) by adding a KL penalty to advantages.
            """
            parser.add_argument(
                "--use-opd",
                action="store_true",
                default=False,
                help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled.",
            )
            parser.add_argument(
                "--opd-type",
                type=str,
                choices=["sglang", "megatron"],
                default=None,
                help=(
                    "Type of on-policy distillation. "
                    "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
                    "'megatron': Teacher model is loaded via --opd-teacher-load and forwarded during training."
                ),
            )
            parser.add_argument(
                "--opd-kl-coef",
                type=float,
                default=1.0,
                help="On-policy distillation KL penalty coefficient. Default is 1.0.",
            )
            parser.add_argument(
                "--opd-teacher-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
                    "The teacher model should have the same architecture as policy/ref model."
                ),
            )
            parser.add_argument(
                "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."
            )
            return parser

        def add_router_arguments(parser):
            parser.add_argument(
                "--use-slime-router",
                action="store_true",
                default=False,
                help="Whether to use SlimeRouter for text-based routing instead of SGLang token-based routing",
            )
            RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether to always use train step as the step metric in wandb. "
                    "If set, we will always use the train steps for wandb logging, "
                    "otherwise, will use rollout step for most info other than train/*. "
                ),
            )
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Whether to log information for multi-turn rollout.",
            )
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument(
                "--log-reward-category",
                type=str,
                default=None,
                help=(
                    "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                    "Specify the key in the reward dict using this argument.",
                ),
            )
            parser.add_argument(
                "--log-correct-samples",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            return parser

        def add_dashboard_arguments(parser):
            group = parser.add_argument_group("slime dashboard")
            group.add_argument(
                "--use-slime-dashboard",
                action="store_true",
                default=False,
                help="Collect failure-isolated GPU, SGLang request, and rollout lifecycle telemetry.",
            )
            group.add_argument(
                "--slime-dashboard-dir",
                type=str,
                default=None,
                help="Shared directory for append-only dashboard telemetry. Defaults to <save>/dashboard.",
            )
            group.add_argument("--slime-dashboard-flush-interval", type=float, default=5.0)
            group.add_argument("--slime-dashboard-gpu-sample-interval", type=float, default=1.0)
            group.add_argument("--slime-dashboard-sglang-scrape-interval", type=float, default=2.0)
            group.add_argument("--slime-dashboard-max-buffered-records", type=int, default=500_000)
            return parser

        # tensorboard
        def add_tensorboard_arguments(parser):
            # tb_project_name, tb_experiment_name
            parser.add_argument("--use-tensorboard", action="store_true", default=False)
            parser.add_argument(
                "--tb-project-name",
                type=str,
                default=None,
                help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory.",
            )
            parser.add_argument("--tb-experiment-name", type=str, default=None)

            return parser

        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            # --load-debug-rollout-data, --debug-rollout-only, --debug-train-only
            # are parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Subsample a portion of the debug rollout data for faster debugging.",
            )
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            parser.add_argument(
                "--grad-log-dir",
                type=str,
                default=None,
                help=(
                    "Directory to save per-parameter gradient statistics. "
                    "If set, after each training step a lightweight stats file is saved "
                    "with per-parameter mean_abs, max_abs, std_abs, norm and sparsity. "
                    "Useful for comparing gradient patterns across different training modes."
                ),
            )
            # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
            parser.add_argument(
                "--memory-snapshot-dir",
                type=str,
                default=".",
            )
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
            )
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["train_overall", "train_actor", "train_log_probs"],
                default=["train_overall"],
                nargs="+",
            )
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default="torch",
            )
            parser.add_argument("--check-weight-update-equal", action="store_true")
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--http-proxy", type=str, default=None)
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that converts samples to training data. "
                    "If set, this function will replace the default _convert_samples_to_train_data. "
                    "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
            return parser

        def add_rollout_buffer_arguments(parser):
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer",
            )

            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry",
            )
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio",
            )
            parser.add_argument(
                "--rollout-task-type",
                type=str,
                default="math",
            )
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "qwen3", "qwen3_5", "distill_qwen"],
                help="Loss mask type",
            )
            parser.add_argument(
                "--data-pad-size-multiplier",
                type=int,
                default=128,
                help="Multiplier for data padding size in data processing.",
            )
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "This function determines whether a sample will participate in loss calculation. "
                    "The function should take args and samples (list[Sample]) as input, and return None. "
                    "Please directly modify the remove_sample attribute of Sample. "
                    "Note: This attribute does not determine whether the sample participates in advantage normalization."
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
            parser.add_argument(
                "--disable-rollout-trim-samples",
                action="store_true",
                default=False,
                help="disable trim samples in rollout buffer when converting samples to train data",
            )
            parser.add_argument(
                "--use-dynamic-global-batch-size",
                action="store_true",
                default=False,
                help="enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data",
            )
            return parser

        def add_custom_megatron_plugins_arguments(parser):
            """
            Add custom Megatron plugins arguments.
            This is a placeholder for any additional arguments that might be needed.
            """
            # Custom arguments can be added here
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            return parser

        def add_mtp_training_arguments(parser):
            """Add MTP training specific arguments."""
            reset_arg(parser, "--mtp-num-layers", type=int, default=None)
            reset_arg(parser, "--mtp-loss-scaling-factor", type=float, default=0.2)
            parser.add_argument(
                "--enable-mtp-training",
                action="store_true",
                default=False,
                help="Enable MTP layer parameter updates during training",
            )

            return parser

        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            parser.add_argument(
                "--ci-disable-kl-checker",
                action="store_true",
            )
            parser.add_argument(
                "--ci-save-grad-norm",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-load-grad-norm",
                type=str,
                default=None,
            )
            return parser

        # Add custom arguments in front to prevent overwritten some slime arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_on_policy_distillation_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_dashboard_arguments(parser)
        parser = add_tensorboard_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_mtp_training_arguments(parser)
        parser = add_ci_arguments(parser)
        parser = add_custom_megatron_plugins_arguments(parser)
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)

        return parser

    return add_slime_arguments


def _pre_parse_mode():
    """Pre-parse CLI to extract arguments that control parsing flow.

    These arguments are removed from add_slime_arguments to avoid
    registering them twice.  The returned namespace is merged into
    the final ``args`` after Phase 2 parsing.
    """
    temp_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    temp_parser.add_argument("--train-backend", type=str, choices=["megatron"], default="megatron")
    temp_parser.add_argument("--debug-rollout-only", action="store_true", default=False)
    temp_parser.add_argument("--debug-train-only", action="store_true", default=False)
    temp_parser.add_argument("--load-debug-rollout-data", type=str, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    return temp_args


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()

    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)

    pre = _pre_parse_mode()
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None

    # Phase 1: Parse sglang args independently (separate parser, parse_known_args).
    # Skipped when sglang servers are not needed.
    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()

    # Phase 2: Parse megatron + slime args.
    # Uses ignore_unknown_args=True so that --sglang-* and pre-parsed CLI flags
    # are silently ignored by the megatron parser.
    from slime.backends.megatron_utils.arguments import megatron_parse_args
    from slime.backends.megatron_utils.arguments import validate_args as megatron_validate_args

    args = megatron_parse_args(
        extra_args_provider=add_slime_arguments,
        skip_hf_validate=pre.debug_rollout_only,
    )

    # Merge pre-parsed args into the main namespace
    for key, value in vars(pre).items():
        setattr(args, key, value)

    # Merge sglang args into the main namespace
    if sglang_ns is not None:
        for key, value in vars(sglang_ns).items():
            setattr(args, key, value)

    slime_validate_args(args)

    if pre.train_backend == "megatron" and not args.debug_rollout_only:
        megatron_validate_args(args)

    if not args.debug_train_only:
        sglang_validate_args(args)

    return args


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)

    # Apply per-dataset overrides from --eval-dataset-override
    overrides = getattr(args, "eval_dataset_override", None) or []
    for override in overrides:
        # format: "dataset_name.key=value"
        pattern = r"^([^.]+)\.(.+)=(.+)$"
        m = re.match(pattern, override)
        if m is None:
            logger.warning(f"Malformed --eval-dataset-override: {override!r} (expected name.key=value)")
            continue
        ds_name, ds_key, ds_value = m.group(1), m.group(2), m.group(3)
        for ds in eval_datasets:
            if ds.name == ds_name:
                # Attempt numeric parsing, fall back to string
                try:
                    ds_value = int(ds_value)
                except ValueError:
                    try:
                        ds_value = float(ds_value)
                    except ValueError:
                        pass
                setattr(ds, ds_key, ds_value)
                break
        else:
            logger.warning(f"--eval-dataset-override: no dataset named {ds_name!r} found, skipping")

    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def slime_validate_args(args):
    args.eval_datasets = _resolve_eval_datasets(args)

    if args.use_slime_dashboard:
        if args.slime_dashboard_dir is None:
            if not args.save:
                raise ValueError("--use-slime-dashboard requires --slime-dashboard-dir or --save")
            args.slime_dashboard_dir = os.path.join(args.save, "dashboard")
        for name in (
            "slime_dashboard_flush_interval",
            "slime_dashboard_gpu_sample_interval",
            "slime_dashboard_sglang_scrape_interval",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.slime_dashboard_max_buffered_records <= 0:
            raise ValueError("--slime-dashboard-max-buffered-records must be positive")

    if args.use_slime_router:
        logger.warning(
            "--use-slime-router is deprecated and ignored. slime now always uses sglang_router "
            "built from https://github.com/zhuzilin/sgl-router."
        )
        args.use_slime_router = False

    if args.kl_coef != 0 or args.use_kl_loss:
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            logger.info(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # Validate on-policy distillation (OPD) arguments
    if args.use_opd:
        if args.opd_type is None:
            raise ValueError("--opd-type must be specified when --use-opd is enabled. Choose 'sglang' or 'megatron'.")

        if args.opd_type == "megatron":
            if args.opd_teacher_load is None:
                raise ValueError(
                    "--opd-teacher-load is required when --opd-type=megatron. "
                    "Please provide the path to the teacher model checkpoint."
                )
            if not os.path.exists(args.opd_teacher_load):
                raise FileNotFoundError(
                    f"opd_teacher_load {args.opd_teacher_load} does not exist, please check the path."
                )
            if not os.path.exists(os.path.join(args.opd_teacher_load, "latest_checkpointed_iteration.txt")):
                logger.info(
                    f"opd_teacher_load {args.opd_teacher_load} does not have latest_checkpointed_iteration.txt, "
                    "please make sure it is a valid megatron checkpoint directory."
                )

        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )
    else:
        # If OPD is not enabled, opd_teacher_load should not be set
        if args.opd_teacher_load is not None:
            raise ValueError("--opd-teacher-load is set but --use-opd is not enabled. Please add --use-opd flag.")

    if args.megatron_to_hf_mode == "bridge":
        if (
            args.load is not None
            and os.path.exists(args.load)
            and os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            # If is a Megatron checkpoint, won't use bridge to load hf weight.
            pass
        else:
            if args.load is None:
                args.load = args.ref_load or args.hf_checkpoint
            # If is a HF checkpoint, set start_rollout_id to 0 here.
            args.start_rollout_id = 0
    else:
        if (
            args.load is None
            or not os.path.exists(args.load)
            or not os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                args.ckpt_step = args.ref_ckpt_step
            args.start_rollout_id = 0

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    if args.get_mismatch_metrics:
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"

        if args.use_rollout_logprobs:
            logger.info(
                "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
            )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    args.use_critic = args.advantage_estimator == "ppo"
    if args.critic_train_only:
        if not args.use_critic:
            raise ValueError("--critic-train-only requires --use-critic (or --advantage-estimator ppo).")
        if args.actor_num_nodes != 0 or args.actor_num_gpus_per_node != 0:
            raise ValueError(
                "--critic-train-only requires --actor-num-nodes 0 --actor-num-gpus-per-node 0, "
                f"but got actor_num_nodes={args.actor_num_nodes}, actor_num_gpus_per_node={args.actor_num_gpus_per_node}."
            )
    if args.critic_num_gpus_per_node is None:
        args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    if args.critic_num_nodes is None:
        args.critic_num_nodes = args.actor_num_nodes
    if args.critic_load is None:
        args.critic_load = args.load
    if args.critic_lr is None:
        args.critic_lr = args.lr

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # always true on offload for colocate at the moment.
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
            if args.use_critic:
                args.rollout_num_gpus += args.critic_num_gpus_per_node * args.critic_num_nodes

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size
    if args.adaptive_group_oversampling_min_groups is None:
        args.adaptive_group_oversampling_min_groups = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )
    if args.adaptive_group_oversampling:
        if args.over_sampling_batch_size <= args.rollout_batch_size:
            raise ValueError(
                "--adaptive-group-oversampling requires --over-sampling-batch-size "
                "to be greater than --rollout-batch-size"
            )
        if not args.use_slime_dashboard:
            raise ValueError("--adaptive-group-oversampling requires --use-slime-dashboard")
        if not (
            0
            < args.adaptive_group_oversampling_min_groups
            <= args.rollout_batch_size
            <= args.over_sampling_batch_size
        ):
            raise ValueError(
                "elastic group window requires 0 < min-groups <= rollout-batch-size "
                "<= over-sampling-batch-size"
            )
        if args.adaptive_group_oversampling_running_threshold <= 0:
            raise ValueError("--adaptive-group-oversampling-running-threshold must be positive")
        if args.adaptive_group_oversampling_queue_threshold < 0:
            raise ValueError("--adaptive-group-oversampling-queue-threshold must be non-negative")
        if not 0 <= args.adaptive_group_oversampling_expansion_kv_threshold <= 1:
            raise ValueError(
                "--adaptive-group-oversampling-expansion-kv-threshold must be in [0, 1]"
            )
        if args.adaptive_group_oversampling_window_seconds <= 0:
            raise ValueError("--adaptive-group-oversampling-window-seconds must be positive")
        if args.adaptive_group_oversampling_cooldown_seconds <= 0:
            raise ValueError("--adaptive-group-oversampling-cooldown-seconds must be positive")
        if args.adaptive_group_oversampling_post_resume_expansion_grace_seconds < 0:
            raise ValueError(
                "--adaptive-group-oversampling-post-resume-expansion-grace-seconds "
                "must be non-negative"
            )
        if args.adaptive_group_oversampling_recovery_seconds <= 0:
            raise ValueError("--adaptive-group-oversampling-recovery-seconds must be positive")
        if args.adaptive_group_oversampling_step_groups <= 0:
            raise ValueError("--adaptive-group-oversampling-step-groups must be positive")
        if args.adaptive_group_oversampling_pressure_queue_threshold < 0:
            raise ValueError(
                "--adaptive-group-oversampling-pressure-queue-threshold must be non-negative"
            )
        if not 0 <= args.adaptive_group_oversampling_pressure_kv_threshold <= 1:
            raise ValueError("--adaptive-group-oversampling-pressure-kv-threshold must be in [0, 1]")
        if not 0 <= args.adaptive_group_oversampling_hard_max_engine_kv_threshold <= 1:
            raise ValueError(
                "--adaptive-group-oversampling-hard-max-engine-kv-threshold must be in [0, 1]"
            )
        if args.adaptive_group_oversampling_hard_engine_queue_threshold < 0:
            raise ValueError(
                "--adaptive-group-oversampling-hard-engine-queue-threshold must be non-negative"
            )
        if not 0 <= args.adaptive_group_oversampling_hard_pressure_kv_threshold <= 1:
            raise ValueError(
                "--adaptive-group-oversampling-hard-pressure-kv-threshold must be in [0, 1]"
            )
        if args.adaptive_group_oversampling_hard_pressure_seconds <= 0:
            raise ValueError("--adaptive-group-oversampling-hard-pressure-seconds must be positive")
        if args.adaptive_group_oversampling_hard_step_groups <= 0:
            raise ValueError("--adaptive-group-oversampling-hard-step-groups must be positive")
    if args.gpu_generation_slots is not None and args.gpu_generation_slots <= 0:
        raise ValueError("--gpu-generation-slots must be positive")
    if args.terminal_live_session_limit is not None and args.terminal_live_session_limit <= 0:
        raise ValueError("--terminal-live-session-limit must be positive")
    if args.terminal_concurrent_resets is not None and args.terminal_concurrent_resets <= 0:
        raise ValueError("--terminal-concurrent-resets must be positive")
    if (
        args.terminal_live_session_limit is not None
        and args.terminal_concurrent_resets is not None
        and args.terminal_concurrent_resets > args.terminal_live_session_limit
    ):
        raise ValueError("--terminal-concurrent-resets cannot exceed --terminal-live-session-limit")
    if args.max_inflight_groups is not None and args.max_inflight_groups <= 0:
        raise ValueError("--max-inflight-groups must be positive")
    if args.inflight_group_soft_limit is not None:
        if args.inflight_group_soft_limit <= 0:
            raise ValueError("--inflight-group-soft-limit must be positive")
        if args.max_inflight_groups is None:
            raise ValueError("--inflight-group-soft-limit requires --max-inflight-groups")
        if args.inflight_group_soft_limit > args.max_inflight_groups:
            raise ValueError("--inflight-group-soft-limit cannot exceed --max-inflight-groups")
    if args.math_ratio < 0 or args.terminal_ratio < 0 or args.math_ratio + args.terminal_ratio > 1:
        raise ValueError(
            "--math-ratio and --terminal-ratio must be non-negative and sum to at most 1 "
            f"(got {args.math_ratio} + {args.terminal_ratio})"
        )
    alternation_quotas = (
        args.math_batches_per_cycle,
        args.qa_batches_per_cycle,
        args.terminal_batches_per_cycle,
    )
    if any(quota < 0 for quota in alternation_quotas):
        raise ValueError("batch-alternation task batch counts must be non-negative")
    if args.batch_alternation and sum(alternation_quotas) == 0:
        raise ValueError("--batch-alternation requires at least one positive task batch count")
    if args.batch_alternation_order is not None and args.batch_alternation_start_task is not None:
        raise ValueError(
            "--batch-alternation-order and --batch-alternation-start-task cannot be set together"
        )
    train_batch_quotas = (args.train_batch_math_groups, args.train_batch_qa_groups)
    if any(quota is not None for quota in train_batch_quotas):
        if any(quota is None for quota in train_batch_quotas):
            raise ValueError("--train-batch-math-groups and --train-batch-qa-groups must be set together")
        if any(quota < 0 for quota in train_batch_quotas):
            raise ValueError("train-batch task group quotas must be non-negative")
        if sum(train_batch_quotas) != args.rollout_batch_size:
            raise ValueError(
                "--train-batch-math-groups + --train-batch-qa-groups must equal "
                f"--rollout-batch-size ({args.rollout_batch_size})"
            )
    if args.partial_rollout_tool_handoff and not args.partial_rollout:
        raise ValueError("--partial-rollout-tool-handoff requires --partial-rollout")

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.enable_mtp_training:
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    if args.use_rollout_routing_replay:
        args.use_routing_replay = True

    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)

    if args.eval_max_context_len is None:
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        assert (
            args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
        ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."

    if args.qkv_format == "bshd":
        assert args.train_backend == "megatron", "bshd format is only supported for megatron backend."
        assert (
            args.use_dynamic_batch_size is False
        ), "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."

    if args.only_train_params_name_list and args.freeze_params_name_list:
        raise ValueError("You can only specify ONE of: --only-train-params-name-list, or --freeze-params-name-list.")
