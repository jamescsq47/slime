import asyncio
import copy
import inspect
import logging
import time
import uuid
from argparse import Namespace
from collections.abc import Callable
from typing import Any
import wandb

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_function, trace_span
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)
_wandb_metric_defined = False

def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # This epoch survives reset(). Custom multi-turn rollouts use it to
        # detect a weight-update boundary after a long CPU/network tool
        # returns, even though the boolean abort flag has already been reset.
        self.abort_epoch = 0
        self.reset()

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_kwargs = build_processor_kwargs(sample.multimodal_inputs)
        processor_output = state.processor(text=sample.prompt, **processor_kwargs)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    has_multimodal = sample.multimodal_inputs and sample.multimodal_inputs.get("images")
    if has_multimodal:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    elif has_multimodal:
        # For multimodal first-turn: send text so SGLang handles image token
        # expansion internally (the processor-expanded input_ids have N patch
        # tokens per image which would mismatch the image_data count).
        payload["text"] = sample.prompt
        if not sample.tokens:
            sample.tokens = prompt_ids
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing (SGLang Model Gateway)
    headers = None
    if sample.session_id:
        if getattr(args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": sample.session_id}

    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}) as span:
        output = await post(url, payload, headers=headers)
        span.update(build_sglang_meta_trace_attrs(output["meta_info"]))

    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens, new_response_log_probs = [], []

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]

    # When partial rollout and masking off policy is enabled, update the loss mask
    if sample.loss_mask is not None:
        assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
        sample.loss_mask += [1] * len(new_response_tokens)

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


@trace_function("generate_and_rm", target="sample")
async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
        custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

        if custom_func_path is not None:
            custom_generate_func = load_function(custom_func_path)
            # if signature has evaluation, pass evaluation
            if "evaluation" in inspect.signature(custom_generate_func).parameters:
                sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
            else:
                sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        with trace_span(samples_need_reward, "reward_model"):
            rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            with trace_span(sample, "reward_model"):
                sample.reward = await async_rm(args, sample)

    return sample


@trace_function(
    "generate_and_rm_group",
    target="group",
    attrs_getter=lambda args, group, sampling_params, evaluation=False: {"group_size": len(group)},
)
async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)
    group_abort_epoch = state.abort_epoch

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and state.abort_epoch == group_abort_epoch and args.group_rm:
        with trace_span(group, "group_reward_model"):
            rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(
    args: Namespace,
    rollout_id: int,
    *,
    drain_timeout: float | None = None,
) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True
    state.abort_epoch += 1

    if parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=False):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # SGLang generation requests normally return promptly after /abort_request.
    # A fully-async caller may bound this drain so a CPU/network tool can keep
    # running outside the weight-update barrier.
    count = 0
    drain_deadline = None if drain_timeout is None else time.monotonic() + max(0.0, drain_timeout)
    while state.pendings:
        wait_timeout = None
        if drain_deadline is not None:
            wait_timeout = max(0.0, drain_deadline - time.monotonic())
            if wait_timeout == 0.0:
                break
        done, state.pendings = await asyncio.wait(
            state.pendings,
            timeout=wait_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            break

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(
            "Collected %d partial samples; deferred %d groups still running outside the abort barrier",
            count,
            len(state.pendings),
        )

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
    try:
        print("record tool call counts for analysis")
        tool_time_counts = []
        sample_time_counts = []
        tool_time_ratios = []
        metrics_to_log = {}
        task_types = []
        math_samples = []  
        qa_samples = [] 

        for group in data:
            for sample in group:
                if hasattr(sample, 'tool_time'):
                    tool_time_counts.append(sample.tool_time)
                if hasattr(sample, 'sample_time'):
                    sample_time_counts.append(sample.sample_time)
                if hasattr(sample, 'tool_time') and hasattr(sample, 'sample_time'):
                    tool_time_ratios.append(
                        sample.tool_time / sample.sample_time if sample.sample_time > 0 else 0.0
                    )
                if hasattr(sample, 'metadata'):
                    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
                    task_type = metadata.get("task_type", "error") 
                    task_types.append(task_type)
                    if task_type == "math":
                        math_samples.append(sample)
                    elif task_type == "qa":
                        qa_samples.append(sample)

        if task_types:
            total_math = len(math_samples)
            total_qa = len(qa_samples)

            def _get_reward_score(r):
                if r is None:
                    return 0.0
                if isinstance(r, dict):
                    return r.get("score", 0.0)
                if isinstance(r, (float, int)):
                    return r
                if hasattr(r, "item"):
                    return r.item()
                return 0.0

            def _get_reward_acc(r):
                if r is None:
                    return None
                if isinstance(r, dict):
                    if "acc" in r:
                        return float(r["acc"])
                    if "score" in r:
                        return 1.0 if r["score"] >= 0.8 else 0.0
                    return None
                if isinstance(r, (float, int)):
                    return 1.0 if r >= 0.8 else 0.0
                if hasattr(r, "item"):
                    return 1.0 if r.item() >= 0.8 else 0.0
                return None

            def _average_acc(samples):
                accs = [_get_reward_acc(s.reward) for s in samples]
                accs = [acc for acc in accs if acc is not None]
                return sum(accs) / len(accs) if accs else 0

            math_reward_avg = sum(_get_reward_score(s.reward) for s in math_samples) / total_math if total_math > 0 else 0
            qa_reward_avg = sum(_get_reward_score(s.reward) for s in qa_samples) / total_qa if total_qa > 0 else 0
            math_acc_avg = _average_acc(math_samples)
            qa_acc_avg = _average_acc(qa_samples)

            # Keep the synchronous metrics aligned with the mixed fully-async
            # rollout.  Tool-call counts are attached dynamically by the custom
            # generators, so use a zero default for samples which made no call.
            math_tool_call_counts = [getattr(s, "tool_call_count", 0) or 0 for s in math_samples]
            qa_tool_call_counts = [getattr(s, "tool_call_count", 0) or 0 for s in qa_samples]
            math_tool_calls_ratio = (
                sum(1 for count in math_tool_call_counts if count > 0) / total_math if total_math > 0 else 0.0
            )
            qa_tool_calls_ratio = (
                sum(1 for count in qa_tool_call_counts if count > 0) / total_qa if total_qa > 0 else 0.0
            )
            
            # Calculate math task sample_time stats
            math_sample_times = [s.sample_time for s in math_samples if hasattr(s, 'sample_time')]
            math_sample_time_max = max(math_sample_times) if math_sample_times else 0
            math_sample_time_avg = sum(math_sample_times) / len(math_sample_times) if math_sample_times else 0
            math_attempt_times = [s.attempt_time for s in math_samples if hasattr(s, "attempt_time")]
            math_attempt_counts = [s.attempt_count for s in math_samples if hasattr(s, "attempt_count")]
            
            # Calculate qa task sample_time stats
            qa_sample_times = [s.sample_time for s in qa_samples if hasattr(s, 'sample_time')]
            qa_sample_time_max = max(qa_sample_times) if qa_sample_times else 0
            qa_sample_time_avg = sum(qa_sample_times) / len(qa_sample_times) if qa_sample_times else 0
            qa_attempt_times = [s.attempt_time for s in qa_samples if hasattr(s, "attempt_time")]
            qa_attempt_counts = [s.attempt_count for s in qa_samples if hasattr(s, "attempt_count")]

            def _scheduling_attempt_metrics(samples, domain):
                partial_counts = [int(getattr(s, "partial_resume_count", 0) or 0) for s in samples]
                restart_counts = [int(getattr(s, "restart_count", 0) or 0) for s in samples]
                lifetime_times = [float(getattr(s, "lifetime_attempt_time", 0.0) or 0.0) for s in samples]
                sample_times = [float(getattr(s, "sample_time", 0.0) or 0.0) for s in samples]
                discarded_times = [
                    max(0.0, lifetime - sample_time)
                    for lifetime, sample_time in zip(lifetime_times, sample_times, strict=False)
                ]
                histories = [
                    entry
                    for sample in samples
                    for entry in (
                        sample.metadata.get("attempt_history", [])
                        if isinstance(getattr(sample, "metadata", None), dict)
                        else []
                    )
                    if isinstance(entry, dict)
                ]
                partial_attempt_times = [
                    float(entry.get("duration", 0.0) or 0.0)
                    for entry in histories
                    if entry.get("status") == "aborted"
                ]
                sample_count = len(samples)
                return {
                    f"tool/{domain}_partial_resume_count_avg": sum(partial_counts) / sample_count if sample_count else 0,
                    f"tool/{domain}_partial_resume_count_max": max(partial_counts) if partial_counts else 0,
                    f"tool/{domain}_restart_count_avg": sum(restart_counts) / sample_count if sample_count else 0,
                    f"tool/{domain}_partial_sample_ratio": sum(count > 0 for count in partial_counts) / sample_count if sample_count else 0,
                    f"tool/{domain}_discarded_attempt_time_avg": sum(discarded_times) / sample_count if sample_count else 0,
                    f"tool/{domain}_partial_attempt_time_avg": sum(partial_attempt_times) / len(partial_attempt_times) if partial_attempt_times else 0,
                    f"tool/{domain}_partial_attempt_time_max": max(partial_attempt_times) if partial_attempt_times else 0,
                }

            math_scheduling_attempt_metrics = _scheduling_attempt_metrics(math_samples, "math")
            qa_scheduling_attempt_metrics = _scheduling_attempt_metrics(qa_samples, "qa")
            
            # Calculate math task response_length stats
            math_response_lengths = [s.response_length for s in math_samples if hasattr(s, 'response_length')]
            math_response_length_max = max(math_response_lengths) if math_response_lengths else 0
            math_response_length_avg = sum(math_response_lengths) / len(math_response_lengths) if math_response_lengths else 0
            
            # Calculate qa task response_length stats
            qa_response_lengths = [s.response_length for s in qa_samples if hasattr(s, 'response_length')]
            qa_response_length_max = max(qa_response_lengths) if qa_response_lengths else 0
            qa_response_length_avg = sum(qa_response_lengths) / len(qa_response_lengths) if qa_response_lengths else 0
            
            metrics_to_log.update({
                "tool/math_count": total_math,
                "tool/qa_count": total_qa,
                "tool/math_reward": math_reward_avg,
                "tool/qa_reward": qa_reward_avg,
                "tool/math_acc": math_acc_avg,
                "tool/qa_acc": qa_acc_avg,
                # Match fully_async_rollout.py: math code calls are averaged
                # over math samples only, while the legacy-named search metric
                # is QA external-tool calls (search + open_page) over QA only.
                "tool/avg_code_call_count": sum(math_tool_call_counts) / total_math if total_math > 0 else 0.0,
                "tool/avg_search_call_count": sum(qa_tool_call_counts) / total_qa if total_qa > 0 else 0.0,
                "tool/math_tool_calls_ratio": math_tool_calls_ratio,
                "tool/qa_tool_calls_ratio": qa_tool_calls_ratio,
                "tool/math_sample_time_max": math_sample_time_max,
                "tool/math_sample_time_avg": math_sample_time_avg,
                "tool/qa_sample_time_max": qa_sample_time_max,
                "tool/qa_sample_time_avg": qa_sample_time_avg,
                "tool/math_attempt_time_max": max(math_attempt_times) if math_attempt_times else 0,
                "tool/math_attempt_time_avg": sum(math_attempt_times) / len(math_attempt_times) if math_attempt_times else 0,
                "tool/qa_attempt_time_max": max(qa_attempt_times) if qa_attempt_times else 0,
                "tool/qa_attempt_time_avg": sum(qa_attempt_times) / len(qa_attempt_times) if qa_attempt_times else 0,
                "tool/math_attempt_count_avg": sum(math_attempt_counts) / len(math_attempt_counts) if math_attempt_counts else 0,
                "tool/qa_attempt_count_avg": sum(qa_attempt_counts) / len(qa_attempt_counts) if qa_attempt_counts else 0,
                "tool/math_response_length_max": math_response_length_max,
                "tool/math_response_length_avg": math_response_length_avg,
                "tool/qa_response_length_max": qa_response_length_max,
                "tool/qa_response_length_avg": qa_response_length_avg,
            })
            metrics_to_log.update(math_scheduling_attempt_metrics)
            metrics_to_log.update(qa_scheduling_attempt_metrics)

        if tool_time_counts:
            avg_tool_times = sum(tool_time_counts) / len(tool_time_counts)
            avg_sample_times = sum(sample_time_counts) / len(sample_time_counts) if sample_time_counts else 0.0

            metrics_to_log.update({
                "tool/avg_tool_calls_time": avg_tool_times,
                "tool/avg_sample_time": avg_sample_times,
                "tool/avg_tool_time_ratio_per_sample": sum(tool_time_ratios) / len(tool_time_ratios) if tool_time_ratios else 0.0,
            })
            
        
        tool_call_counts = []
        for group in data:
            for sample in group:
                if hasattr(sample, 'tool_call_count'):
                    tool_call_counts.append(sample.tool_call_count)
        
        if tool_call_counts:
            avg_tool_calls = sum(tool_call_counts) / len(tool_call_counts)
            samples_with_tool_calls = sum(1 for count in tool_call_counts if count > 0)

            metrics_to_log.update({
                "tool/avg_tool_calls_per_sample": avg_tool_calls,
                "tool/total_tool_calls": sum(tool_call_counts),
                "tool/samples_with_tool_calls": samples_with_tool_calls,
            })

        tool_token_counts = []
        response_lengths = []
        for group in data:
            for sample in group:
                if hasattr(sample, 'tool_token_count'):
                    tool_token_counts.append(sample.tool_token_count)
                    response_lengths.append(sample.response_length)

        if tool_token_counts:
            total_tool_tokens = sum(tool_token_counts)
            avg_tool_tokens = total_tool_tokens / len(tool_token_counts)
            per_sample_ratios = [
                t / r if r > 0 else 0.0
                for t, r in zip(tool_token_counts, response_lengths)
            ]
            tool_token_ratio = sum(per_sample_ratios) / len(per_sample_ratios)
            metrics_to_log.update({
                "tool/avg_tool_tokens_per_sample": avg_tool_tokens,
                "tool/tool_token_ratio_in_response": tool_token_ratio,
            })
        
        mismatch_counts = []
        for group in data:
            for sample in group:
                if hasattr(sample, 'mismatch'):
                    mismatch_counts.append(sample.mismatch)
         

        if mismatch_counts:
            total_mismatches = sum(mismatch_counts)
            avg_mismatches = total_mismatches / len(mismatch_counts)
            samples_with_mismatches = sum(1 for m in mismatch_counts if m > 0)

            metrics_to_log.update({
                "debug/total_mismatches": total_mismatches,
                "debug/avg_mismatches_per_sample": avg_mismatches,
                "debug/samples_with_mismatches": samples_with_mismatches,
            })

        # Use a dedicated rollout step axis to avoid conflicts with other threads logging wandb step.
        if metrics_to_log:
            global _wandb_metric_defined
            if not _wandb_metric_defined:
                wandb.define_metric("rollout/step")
                wandb.define_metric("tool/*", step_metric="rollout/step")
                wandb.define_metric("debug/*", step_metric="rollout/step")
                _wandb_metric_defined = True

            metrics_to_log["rollout/step"] = rollout_id
            wandb.log(metrics_to_log)
    except Exception:
        logger.warning("Failed to compute/record tool metrics", exc_info=True)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    # When label_key points to a dict column (e.g. {"ground_truth": "18", ...}),
    # extract the inner field so downstream reward functions get a plain string.
    if dataset_cfg.label_sub_key is not None:
        for s in dataset.samples:
            if isinstance(s.label, dict):
                s.label = str(s.label.get(dataset_cfg.label_sub_key, ""))

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = dataset_cfg.eval_reward_key or args.eval_reward_key or args.reward_key

    def _extract_reward(sample: Sample) -> float:
        if not reward_key:
            return sample.reward
        if isinstance(sample.reward, dict):
            return sample.reward[reward_key]
        return sample.reward  # reward func returned a scalar (e.g. qa reward)

    return {
        dataset_cfg.name: {
            "rewards": [_extract_reward(sample) for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
            "wandb_prefix": dataset_cfg.wandb_prefix or "eval",
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
