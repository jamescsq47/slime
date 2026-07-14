# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
# This is a unified version supporting both local search and Google search, with optional log probability collection

import asyncio
import math
import random
import re
import numpy as np
import time

from qa_em_format import compute_score_em

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Configuration for Search-R1
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 4,
    "topk": 3,
    "search_concurrency": 32,
    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google" or "duckduckgo"
    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual API key
        "snippet_only": True,  # Set to True to only return snippets
        "proxy": None,  # Set to your proxy if needed
    },
    "duckduckgo":{
        "proxy": None,
    },
    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities for TIS metrics
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])
TOOL_DELAY_REMAINING_KEY = "pending_tool_delay_remaining"


def _sample_tool_delay(args) -> float:
    mean = max(0.0, float(getattr(args, "tool_delay_mean", 25.0)))
    variance = max(0.0, float(getattr(args, "tool_delay_variance", 500.0)))
    if mean == 0.0:
        return 0.0
    if variance == 0.0:
        return mean
    sigma2 = math.log1p(variance / (mean * mean))
    mu = math.log(mean) - sigma2 / 2
    return random.lognormvariate(mu, math.sqrt(sigma2))


async def _sleep_after_tool_delay(args, state: GenerateState, sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if not getattr(args, "enable_tool_delay", False) and metadata.get(TOOL_DELAY_REMAINING_KEY) is None:
        return True

    remaining = metadata.get(TOOL_DELAY_REMAINING_KEY)
    if remaining is None:
        remaining = _sample_tool_delay(args)
    remaining = max(0.0, float(remaining))
    metadata[TOOL_DELAY_REMAINING_KEY] = remaining
    check_interval = max(0.01, float(getattr(args, "tool_delay_check_interval", 0.5)))

    while remaining > 0:
        if state.aborted:
            metadata[TOOL_DELAY_REMAINING_KEY] = remaining
            return False

        sleep_for = min(check_interval, remaining)
        start = time.time()
        await asyncio.sleep(sleep_for)
        elapsed = time.time() - start
        remaining = max(0.0, remaining - elapsed)
        sample.tool_delay_time = getattr(sample, "tool_delay_time", 0.0) + elapsed

    metadata.pop(TOOL_DELAY_REMAINING_KEY, None)
    return True


def _should_mask_offpolicy(args, sample):
    """Check if off-policy masking should be applied for THIS sample based on per-group lag."""
    if getattr(args, 'mask_offpolicy_in_partial_rollout', False):
        return True

    mask_math = getattr(args, 'mask_offpolicy_math', None)
    mask_qa = getattr(args, 'mask_offpolicy_qa', None)

    if mask_math is None and mask_qa is None:
        return False

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_type = metadata.get("task_type", "math")
    dispatch_version = metadata.get("dispatch_version", None)

    if dispatch_version is None:
        return False

    threshold = None
    if task_type == "math" and mask_math is not None:
        threshold = mask_math
    elif task_type == "qa" and mask_qa is not None:
        threshold = mask_qa
    else:
        return False

    try:
        import fully_async_rollout as rollout_mod
        worker = rollout_mod.get_existing_worker()
        if worker is None or not hasattr(worker, 'data_buffer'):
            return False
        data_source = worker.data_buffer
        if not hasattr(data_source, 'version_task_counts'):
            return False

        current_version = getattr(args, 'current_policy_version', 0)
        my_lag = current_version - dispatch_version

        if my_lag <= 0:
            return False

        lag_sample = 0
        for v in range(dispatch_version, current_version):
            lag_sample += data_source.version_task_counts.get(v, {}).get(task_type, 0)

        return lag_sample >= threshold

    except Exception as e:
        print(f"[WARNING] Failed to compute lag for mask_offpolicy check: {e}")

    return False


def _passages2string(retrieval_result):
    """
    Convert retrieval results to a formatted string.
    This function works with both google_search and local_search results.
    """
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


async def search(query: str) -> str:
    """
    Perform search using either local search engine or Google search.
    The search backend is determined by SEARCH_R1_CONFIGS["search_backend"].
    """
    backend = SEARCH_R1_CONFIGS["search_backend"]

    if backend == "local":
        from local_search_server import local_search

        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from google_search_server import google_search

        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    elif backend == "duckduckgo":
        from duckduckgo_search_server import duckduckgo_search

        duckduckgo_config = SEARCH_R1_CONFIGS["duckduckgo"]
        result = await duckduckgo_search(
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=duckduckgo_config.get("proxy"),
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}. " f"Must be either 'local' or 'google'.")

    return _passages2string(result)


# IMPORTANT: When we need to collect log probabilities (logp), we CANNOT do any postprocessing
# on the strings returned from the inference engine (sglang). This is because:
# 1. We don't know how to truncate the corresponding tokens/logp arrays to match the modified string
# 2. Re-tokenizing the postprocessed string may produce different tokens than what the engine generated,
#    leading to misalignment between tokens and their log probabilities
# Therefore, postprocess_responses is only used when return_logprob=False.
def postprocess_responses(resp: str) -> str:
    """
    Post-process response to ensure tag completeness.
    Only used when SEARCH_R1_CONFIGS["return_logprob"] is False.
    """
    return (
        resp.split("</search>")[0] + "</search>"
        if "</search>" in resp
        else resp.split("</answer>")[0] + "</answer>" if "</answer>" in resp else resp
    )


def postprocess_predictions(prediction: str):
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()  # Return only the content inside the tags
        action = match.group(1)
    else:
        content = ""
        action = None

    return action, content


async def execute_predictions(prediction: str) -> str:
    action, content = postprocess_predictions(prediction)

    if action == "search":
        search_query = content
        async with SEMAPHORE:
            search_results = await search(search_query)
        next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = "\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
        done = False

    return next_obs, done


def reconstruct_loss_masks(response: str, tokenizer) -> list:
    """Reconstruct loss masks from response content.
    Used when resuming a partial rollout.
    """
    try:
        response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
        loss_masks = [1] * len(response_tokens)

        information_pattern = r'<information>(.*?)</information>'
        matches = list(re.finditer(information_pattern, response, re.DOTALL))

        if not matches:
            return loss_masks

        for match in matches:
            start_char = match.start()
            end_char = match.end()

            prefix = response[:start_char]
            prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            start_token_idx = len(prefix_tokens)

            prefix_with_info = response[:end_char]
            prefix_with_info_tokens = tokenizer(prefix_with_info, add_special_tokens=False)["input_ids"]
            end_token_idx = len(prefix_with_info_tokens)

            for i in range(start_token_idx, end_token_idx):
                if i < len(loss_masks):
                    loss_masks[i] = 0

        return loss_masks

    except Exception as e:
        print(f"[WARNING] Error reconstructing loss masks: {e}")
        response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
        loss_masks = [1] * len(response_tokens)
        return loss_masks


def count_tool_turns(response: str) -> int:
    """Count the number of completed search turns in the response.
    Used to determine where to resume generation.
    """
    information_count = response.count("</information>")
    return information_count


async def generate(args, sample: Sample, sampling_params) -> Sample:

    state = GenerateState(args)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Ensure metadata exists
    if not hasattr(sample, 'metadata') or sample.metadata is None:
        sample.metadata = {}

    # Check if this is a partial rollout resume
    if args.partial_rollout and sample.status == Sample.Status.ABORTED and sample.response:
        # Partial rollout: resume from existing response
        metadata = sample.metadata

        prompt_text = sample.prompt
        prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response = sample.response
        response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]

        # Restore state from saved metadata if available
        if _should_mask_offpolicy(args, sample):
            # Off-policy masking: all existing tokens are off-policy,
            # only newly generated tokens (added after this point) will be marked as on-policy (1).
            loss_mask = [0] * len(response_token_ids)
            sample.metadata["offpolicy_masked"] = True
            if metadata.get("tool_call_count"):
                tool_call_count = metadata.get("tool_call_count", 0)
                start_turn = metadata.get("current_turn", tool_call_count)
            else:
                tool_call_count = count_tool_turns(response)
                start_turn = tool_call_count
        elif metadata.get("partial_rollout") and metadata.get("loss_mask") is not None:
            loss_mask = metadata["loss_mask"]
            if len(loss_mask) != len(response_token_ids):
                loss_mask = reconstruct_loss_masks(response, state.tokenizer)
            tool_call_count = metadata.get("tool_call_count", 0)
            start_turn = metadata.get("current_turn", tool_call_count)
        else:
            loss_mask = reconstruct_loss_masks(response, state.tokenizer)
            tool_call_count = count_tool_turns(response)
            start_turn = tool_call_count

        # Restore rollout_log_probs from sample
        if SEARCH_R1_CONFIGS["return_logprob"]:
            if sample.rollout_log_probs is not None:
                rollout_log_probs = list(sample.rollout_log_probs)
                if len(rollout_log_probs) != len(response_token_ids):
                    rollout_log_probs = [0.0] * len(response_token_ids)
            else:
                rollout_log_probs = [0.0] * len(response_token_ids)
        else:
            rollout_log_probs = None

        # Carry over timing from previous attempt(s)
        _accrued_sample_time = getattr(sample, 'sample_time', 0.0) or 0.0
        tool_time = getattr(sample, 'tool_time', 0.0) or 0.0
    else:
        # Non-partial rollout: start fresh
        prompt_text = sample.prompt
        prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response = ""
        response_token_ids = []
        loss_mask = []
        rollout_log_probs = [] if SEARCH_R1_CONFIGS["return_logprob"] else None
        tool_call_count = 0
        tool_time = 0.0
        _accrued_sample_time = 0.0
        start_turn = 0

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    _start_time = time.time()
    output = None

    def _save_partial_for_resume(current_turn: int) -> Sample:
        sample.status = Sample.Status.ABORTED
        sample.tokens = prompt_tokens_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.response = response
        sample.loss_mask = loss_mask
        sample.tool_call_count = tool_call_count
        if SEARCH_R1_CONFIGS["return_logprob"]:
            sample.rollout_log_probs = rollout_log_probs
        sample.metadata.update({
            "partial_rollout": True,
            "current_turn": current_turn,
            "loss_mask": loss_mask,
            "tool_call_count": tool_call_count,
        })
        sample.sample_time = _accrued_sample_time + (time.time() - _start_time)
        sample.tool_time = tool_time
        return sample

    if sample.metadata.get(TOOL_DELAY_REMAINING_KEY) is not None:
        if not await _sleep_after_tool_delay(args, state, sample):
            return _save_partial_for_resume(start_turn)

    for _turn_idx in range(start_turn, SEARCH_R1_CONFIGS["max_turns"]):
        # Check if total length exceeds max context length
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Clamp per-turn max_new_tokens to the remaining context budget
        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = min(
            sampling_params.get("max_new_tokens", remaining_budget),
            remaining_budget,
        )
        # Stop generation at tool call or answer boundaries so the model
        # doesn't hallucinate search results or continue past the action tag.
        per_turn_sampling_params["stop"] = ["</search>", "</answer>"]

        payload = {
            "text": prompt_text + response,
            "sampling_params": per_turn_sampling_params,
        }
        # Add log probability collection if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)

        # abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            if not args.partial_rollout:
                sample.sample_time = _accrued_sample_time + (time.time() - _start_time)
                sample.tool_time = tool_time
                sample.status = Sample.Status.ABORTED
                return sample
            else:
                # Partial rollout enabled: process partial response and save state
                cur_response = output["text"]

                if SEARCH_R1_CONFIGS["return_logprob"]:
                    if "output_token_logprobs" in output["meta_info"]:
                        cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                        cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                    else:
                        sample.sample_time = _accrued_sample_time + (time.time() - _start_time)
                        sample.tool_time = tool_time
                        sample.status = Sample.Status.ABORTED
                        return sample
                else:
                    cur_response = postprocess_responses(cur_response)
                    cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

                if cur_response:
                    response += cur_response
                    response_token_ids += cur_response_token_ids
                    if _should_mask_offpolicy(args, sample):
                        # All existing + new tokens are off-policy; only tokens
                        # generated on the next complete resume will be on-policy.
                        loss_mask += [0] * len(cur_response_token_ids)
                        sample.metadata["offpolicy_masked"] = True
                    else:
                        loss_mask += [1] * len(cur_response_token_ids)

                    if SEARCH_R1_CONFIGS["return_logprob"]:
                        rollout_log_probs += cur_response_log_probs

                    # Execute search to maintain conversation flow
                    _tool_start = time.time()
                    next_obs, done = await execute_predictions(cur_response)
                    if next_obs:
                        if "<information>" in next_obs:
                            tool_call_count += 1
                        tool_time += time.time() - _tool_start

                        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
                        response += next_obs
                        response_token_ids += obs_tokens_ids
                        loss_mask += [0] * len(obs_tokens_ids)
                        if SEARCH_R1_CONFIGS["return_logprob"]:
                            rollout_log_probs += [0.0] * len(obs_tokens_ids)

                        await _sleep_after_tool_delay(args, state, sample)

                # Trim overflow from tool output
                overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
                if overflow > 0:
                    response_token_ids = response_token_ids[:-overflow]
                    loss_mask = loss_mask[:-overflow]
                    if SEARCH_R1_CONFIGS["return_logprob"]:
                        rollout_log_probs = rollout_log_probs[:-overflow]
                    response = state.tokenizer.decode(response_token_ids)

                # Save state for resumption
                sample.status = Sample.Status.ABORTED
                sample.tokens = prompt_tokens_ids + response_token_ids
                sample.response_length = len(response_token_ids)
                sample.response = response
                sample.loss_mask = loss_mask
                sample.tool_call_count = tool_call_count
                if SEARCH_R1_CONFIGS["return_logprob"]:
                    sample.rollout_log_probs = rollout_log_probs

                sample.metadata.update({
                    "partial_rollout": True,
                    "current_turn": _turn_idx,
                    "loss_mask": loss_mask,
                    "tool_call_count": tool_call_count,
                })

                sample.sample_time = _accrued_sample_time + (time.time() - _start_time)
                sample.tool_time = tool_time
                return sample

        cur_response = output["text"]

        # Extract tokens and log probs based on configuration
        if SEARCH_R1_CONFIGS["return_logprob"]:
            # Extract log probs from output - required for TIS metrics
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure 'return_logprob': True is set in the payload."
                )

            # Use token IDs and log probs directly from output_token_logprobs
            # This ensures perfect alignment between tokens and log probs
            # output_token_logprobs format: [[log_prob, token_id, ...], ...]
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            # When not collecting log probs, we can safely postprocess the response
            cur_response = postprocess_responses(cur_response)
            # Tokenize the (possibly postprocessed) response
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)

        # Add log probs if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        start_time = time.time()
        next_obs, done = await execute_predictions(cur_response)
        elapsed_time = time.time() - start_time
        tool_time += elapsed_time

        if done:
            break

        if "<information>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens if enabled (they won't be used due to loss_mask=0)
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)

            # Verify alignment when collecting log probs
            assert len(response_token_ids) == len(
                rollout_log_probs
            ), f"Token/logp length mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"

        # Tool output is appended verbatim and can push total_length past
        # max_context_length. Trim tail tokens so the final sample fits.
        overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
        if overflow > 0:
            response_token_ids = response_token_ids[:-overflow]
            loss_mask = loss_mask[:-overflow]
            if SEARCH_R1_CONFIGS["return_logprob"]:
                rollout_log_probs = rollout_log_probs[:-overflow]
            response = state.tokenizer.decode(response_token_ids)
            sample.status = Sample.Status.TRUNCATED
            break

        if not await _sleep_after_tool_delay(args, state, sample):
            return _save_partial_for_resume(_turn_idx)

    # Store statistics for wandb logging
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    sample.tool_call_count = tool_call_count
    tool_token_count = loss_mask.count(0)
    sample.tool_token_count = tool_token_count
    sample.tool_time = tool_time
    sample.sample_time = _accrued_sample_time + (time.time() - _start_time)

    # Store log probs if enabled
    if SEARCH_R1_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    # Set status based on finish reason
    if output is not None:
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.TRUNCATED

    return sample


async def reward_func(args, sample, **kwargs):
    """The reward function for retrieval-based question answering.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # 动态适配 Train 和 Eval 的标签格式
    if isinstance(sample.label, str):
        # 针对 GAIA 评估集：把字符串 "答案" 包装成 {"target": ["答案"]}
        formatted_gt = {"target": [sample.label]}
    elif isinstance(sample.label, dict):
        # 针对 eval 集：label = {"ground_truth": {"target": [...]}, "style": "rule"}
        formatted_gt = sample.label.get("ground_truth", {})
        if not isinstance(formatted_gt, dict) or "target" not in formatted_gt:
            # 针对训练集：label 已经是 {"target": [...]}
            if "target" in sample.label:
                formatted_gt = sample.label
            else:
                formatted_gt = {"target": [str(sample.label)]}
    else:
        # 兜底处理
        formatted_gt = {"target": [str(sample.label)]}
        
    score = compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=formatted_gt,
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    return score
