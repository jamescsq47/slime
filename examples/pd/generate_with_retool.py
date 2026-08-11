# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import asyncio
import math
import os
import random
import re
import time
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from attempt_tracking import AttemptTracker
from agentic_kv_request import (
    add_agentic_kv_metadata,
    build_agentic_extra_key,
    confirm_agentic_generation_final,
    confirm_agentic_generation_tool,
    generation_has_visible_content,
    lifecycle_enabled,
)
from slime.dashboard.api import span as dashboard_span
from slime.rollout.sglang_rollout import (
    GenerateState,
    PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY,
    track_sglang_generation,
)
from slime.utils.http_utils import post
from slime.utils.types import Sample
from pd_metrics import sglang_meta_attrs

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality
from tool_sandbox import TOOL_CONFIGS, tool_registry

TOOL_DELAY_REMAINING_KEY = "pending_tool_delay_remaining"


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


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


async def _sleep_after_tool_delay(
    args,
    state: GenerateState,
    sample: Sample,
    *,
    abort_epoch: int | None = None,
) -> bool:
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
        if state.aborted or (
            abort_epoch is not None and int(getattr(state, "abort_epoch", 0)) != abort_epoch
        ):
            metadata[TOOL_DELAY_REMAINING_KEY] = remaining
            return False

        sleep_for = min(check_interval, remaining)
        start = time.monotonic()
        await asyncio.sleep(sleep_for)
        elapsed = time.monotonic() - start
        remaining = max(0.0, remaining - elapsed)
        sample.tool_delay_time = getattr(sample, "tool_delay_time", 0.0) + elapsed

    metadata.pop(TOOL_DELAY_REMAINING_KEY, None)
    return True


def _should_mask_offpolicy(args, sample):
    """Check if off-policy masking should be applied for THIS sample based on per-group lag.

    Uses sample.metadata["dispatch_version"] (the version when this group was dispatched)
    to compute this group's own lag, then sums the task-specific training samples
    across that lag window.

    Returns True if:
    - mask_offpolicy_in_partial_rollout is globally enabled, OR
    - mask_offpolicy_math is set and this math sample's lag_sample >= threshold, OR
    - mask_offpolicy_qa is set and this QA sample's lag_sample >= threshold
    """
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
        # No dispatch_version recorded, fall back to no masking
        return False

    # Only check the threshold for this sample's own task type
    threshold = None
    if task_type == "math" and mask_math is not None:
        threshold = mask_math
    elif task_type == "qa" and mask_qa is not None:
        threshold = mask_qa
    else:
        return False

    # Compute this group's own lag_sample
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

        # Sum this sample's task type across its own lag window
        lag_sample = 0
        for v in range(dispatch_version, current_version):
            lag_sample += data_source.version_task_counts.get(v, {}).get(task_type, 0)

        return lag_sample >= threshold

    except Exception as e:
        print(f"[WARNING] Failed to compute lag for mask_offpolicy check: {e}")

    return False


# Jinja2 template for tool-enabled conversations
TOOL_TEMPLATE = """<|im_start|>system
{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}
{% else %}
You are a helpful assistant.
{% endif %}{% if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{% for tool in tools %}{{ tool | tojson }}
{% endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{% endif %}
<|im_end|>
{% for message in messages %}{% if message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}
<|im_start|>assistant
"""


_QWEN_MESSAGE_PATTERN = re.compile(
    r"<\|im_start\|>(system|user|assistant)\s*\n?(.*?)<\|im_end\|>",
    re.DOTALL,
)

RETOOL_PROTOCOL = (
    "You are a mathematical problem-solving assistant. Reason carefully and solve the user's problem.\n\n"
    "Using code_interpreter is OPTIONAL. If you can solve the problem reliably by reasoning in text, "
    "do not call the tool.\n\n"
    "If you call code_interpreter, follow all of these rules:\n"
    "- Output exactly one <tool_call> JSON block and no text after it in that turn.\n"
    "- Put raw executable Python in the code argument. Do not use Markdown code fences.\n"
    "- Use print(...) so the result appears in the tool output.\n"
    "- Use the returned result in your reasoning; do not repeat an identical tool call.\n"
    "- Call the tool again only when a genuinely different computation is needed.\n\n"
    "For the final response, do not include a tool call. Give concise reasoning and make the last line exactly:\n"
    "#### \\boxed{answer}\n"
    "A final answer and a tool call are mutually exclusive in the same turn."
)


def extract_user_prompt(prompt: str) -> str:
    """Unwrap a single-turn Qwen chat-template prompt if necessary."""
    if not isinstance(prompt, str) or "<|im_start|>" not in prompt:
        return prompt

    messages = _QWEN_MESSAGE_PATTERN.findall(prompt)
    user_messages = [content.strip() for role, content in messages if role == "user"]
    if len(user_messages) == 1:
        return user_messages[0]

    return prompt


def format_conversation_with_tools(
    prompt: str, tools: list[dict[str, Any]] = None, system_prompt: str = None, messages: list[dict[str, Any]] = None
) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = f"{system_prompt.rstrip()}\n\n{RETOOL_PROTOCOL}"
    else:
        system_content = RETOOL_PROTOCOL

    messages_to_render.append({"role": "system", "content": system_content})

    # Add user message if provided
    if prompt:
        messages_to_render.append({"role": "user", "content": extract_user_prompt(prompt)})

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])

    return formatted_text


def postprocess_predictions(prediction: str):
    """Extract action and content from prediction string"""
    # Check for \boxed{...} format (accept with or without "Answer:" prefix)
    # Use a more robust regex that handles nested braces
    answer_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content

    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            import json

            # Clean up the JSON string by removing newlines and extra
            # whitespace
            json_str = tool_call_match.group(1)
            # Replace newlines in string values with \n
            json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "code_interpreter":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    # Then check for <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    # Finally check for ```python code blocks (lowest priority)
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Find the last occurrence of <tool_call>...</tool_call>
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle \boxed{...} format (accept with or without "Answer:" prefix)
    if "\\boxed{" in resp:
        answer_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    return resp


def reconstruct_loss_masks(response: str, tokenizer) -> list:
    """Reconstruct loss masks from response content.
    Used when resuming a partial rollout.
    """
    try:
        response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
        loss_masks = [1] * len(response_tokens)

        interpreter_pattern = r'<interpreter>(.*?)</interpreter>'
        matches = list(re.finditer(interpreter_pattern, response, re.DOTALL))

        if not matches:
            return loss_masks

        for match in matches:
            start_char = match.start()
            end_char = match.end()

            prefix = response[:start_char]
            prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            start_token_idx = len(prefix_tokens)

            prefix_with_interpreter = response[:end_char]
            prefix_with_interp_tokens = tokenizer(prefix_with_interpreter, add_special_tokens=False)["input_ids"]
            end_token_idx = len(prefix_with_interp_tokens)

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
    """Count the number of completed tool turns in the response.
    Used to determine where to resume generation.
    """
    interpreter_count = response.count("</interpreter>")
    return interpreter_count


def append_observation_with_budget(
    state: GenerateState,
    prompt_tokens_ids: list[int],
    response: str,
    response_token_ids: list[int],
    loss_masks: list[int],
    rollout_log_probs: list[float] | None,
    next_obs: str,
    max_context_length: int,
):
    """Append a tool observation without exceeding the training context budget."""
    remaining_budget = max_context_length - len(prompt_tokens_ids) - len(response_token_ids)
    if remaining_budget <= 0:
        return response, response_token_ids, loss_masks, rollout_log_probs, True, 0

    obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
    truncated = False
    if len(obs_tokens_ids) > remaining_budget:
        obs_tokens_ids = obs_tokens_ids[:remaining_budget]
        next_obs = state.tokenizer.decode(obs_tokens_ids)
        truncated = True

    response += next_obs
    response_token_ids += obs_tokens_ids
    loss_masks += [0] * len(obs_tokens_ids)
    if rollout_log_probs is not None:
        rollout_log_probs += [0.0] * len(obs_tokens_ids)

    return response, response_token_ids, loss_masks, rollout_log_probs, truncated, len(obs_tokens_ids)


async def execute_predictions(prediction: str) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content.strip()
        if code:
            result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found" "\n</interpreter>\n\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nThe previous response was not a valid action. Either continue reasoning and end with "
            "the line `#### \\boxed{answer}`, or call code_interpreter using exactly one valid "
            "<tool_call> JSON block with no text after it.\n"
        )
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls with partial rollout support"""

    state = GenerateState(args)
    attempt_abort_epoch = int(getattr(state, "abort_epoch", 0))

    def _crossed_weight_update_boundary() -> bool:
        return int(getattr(state, "abort_epoch", 0)) != attempt_abort_epoch

    pd_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    local_router_port = getattr(args, "retool_local_router_port", None)
    local_url = (
        f"http://{args.sglang_router_ip}:{local_router_port}/generate"
        if local_router_port is not None
        else None
    )

    # Initialize total_off_policy_tokens if it doesn't exist
    if not hasattr(sample, 'total_off_policy_tokens'):
        sample.total_off_policy_tokens = 0

    # Set up tool specs
    tool_specs = tool_registry.get_tool_specs()

    # Ensure metadata exists
    if not hasattr(sample, 'metadata') or sample.metadata is None:
        sample.metadata = {}

    # Check if this is a partial rollout resume
    is_partial_resume = bool(
        args.partial_rollout and sample.status == Sample.Status.ABORTED and sample.response
    )
    if is_partial_resume:
        # Partial rollout: resume from existing response
        metadata = sample.metadata

        if metadata.get("formatted_prompt"):
            prompt = metadata["formatted_prompt"]
        else:
            prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

        # Reuse the exact token IDs saved by the previous attempt. Text
        # decode->encode is not guaranteed to be idempotent, and replacing
        # valid old logprobs with zeroes corrupts TIS.
        saved_prompt_token_count = metadata.get("formatted_prompt_token_count")
        if (
            sample.tokens
            and saved_prompt_token_count is not None
            and 0 <= int(saved_prompt_token_count) <= len(sample.tokens)
        ):
            saved_prompt_token_count = int(saved_prompt_token_count)
            prompt_tokens_ids = list(sample.tokens[:saved_prompt_token_count])
            response_token_ids = list(sample.tokens[saved_prompt_token_count:])
            response = state.tokenizer.decode(response_token_ids)
        else:
            prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response = sample.response
            response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]

        if _should_mask_offpolicy(args, sample):
            # Off-policy masking: all existing tokens are off-policy,
            # only newly generated tokens (added after this point) will be marked as on-policy (1).
            loss_masks = [0] * len(response_token_ids)
            sample.metadata["offpolicy_masked"] = True
        elif metadata.get("partial_rollout") and "loss_masks" in metadata:
            # Keep masks aligned to the exact saved token IDs. In particular,
            # tool_call_count == 0 is valid for an aborted first assistant turn
            # and must not force a decode -> re-tokenize reconstruction.
            loss_masks = list(metadata.get("loss_masks") or [])
            if len(loss_masks) != len(response_token_ids):
                print(f"[WARNING] Saved loss_masks length ({len(loss_masks)}) != response tokens ({len(response_token_ids)})")
                loss_masks = (loss_masks + [1] * len(response_token_ids))[: len(response_token_ids)]
        else:
            loss_masks = reconstruct_loss_masks(response, state.tokenizer)

        saved_tool_call_count = metadata.get("tool_call_count")
        tool_call_count = (
            int(saved_tool_call_count) if saved_tool_call_count is not None else count_tool_turns(response)
        )
        start_turn = int(metadata.get("current_turn", tool_call_count))

        # Legacy partial samples may lack exact saved token boundaries.
        if sample.rollout_log_probs is not None and len(sample.rollout_log_probs) != len(response_token_ids):
            print(f"[WARNING] rollout_log_probs length ({len(sample.rollout_log_probs)}) != response tokens ({len(response_token_ids)}), resetting to zeros")
            sample.rollout_log_probs = [0.0] * len(response_token_ids)

        # Update off-policy token count
        sample.total_off_policy_tokens += len(response_token_ids)
        assistant_start_token_idx = int(
            metadata.get("retool_assistant_start_token_idx", len(response_token_ids))
        )
        assistant_start_token_idx = min(max(assistant_start_token_idx, 0), len(response_token_ids))
        continuing_partial_assistant = bool(metadata.get("retool_continue_partial_assistant", False))
        # Carry over timing from previous attempt(s)
        _accrued_sample_time = getattr(sample, 'sample_time', 0.0) or 0.0
        _tool_time = getattr(sample, 'tool_time', 0.0) or 0.0
        _tool_token_count = int(getattr(sample, "tool_token_count", 0) or metadata.get("tool_token_count", 0) or 0)
    else:
        # Non-partial rollout: start fresh
        sample.rollout_log_probs = None
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = None

        prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
        prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response = ""
        response_token_ids = []
        loss_masks = []
        tool_call_count = 0  # Track actual tool call rounds
        start_turn = 0
        assistant_start_token_idx = 0
        continuing_partial_assistant = False
        _accrued_sample_time = 0.0
        _tool_time = 0.0
        _tool_token_count = 0

    _start_time = time.monotonic()
    attempt_tracker = AttemptTracker.begin(
        args,
        sample,
        is_partial_resume=is_partial_resume,
        start_response_length=len(response_token_ids),
        start_tool_call_count=tool_call_count,
        start_tool_time=_tool_time,
    )

    def _record_timing(reason: str) -> None:
        attempt_time = time.monotonic() - _start_time
        sample.sample_time = _accrued_sample_time + attempt_time
        attempt_tracker.finish(
            duration=attempt_time,
            cumulative_sample_time=sample.sample_time,
            status=sample.status,
            reason=reason,
            response_length=len(response_token_ids),
            tool_call_count=tool_call_count,
            tool_time=_tool_time,
        )

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
    max_context_length = min(max_context_length, len(prompt_tokens_ids) + _int_env("MIXED_RETOOL_MAX_RESPONSE_LEN", 8192))

    def _save_partial_for_resume(
        current_turn: int,
        status=Sample.Status.ABORTED,
        reason: str = "state_abort",
    ) -> Sample:
        sample.metadata.pop(PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY, None)
        sample.status = status
        sample.tokens = prompt_tokens_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.response = response
        sample.loss_mask = loss_masks
        sample.tool_call_count = tool_call_count
        sample.tool_token_count = _tool_token_count
        sample.code_call_count = tool_call_count
        sample.search_call_count = 0
        sample.payload_text = prompt + response
        sample.payload_has_system = "<|im_start|>system" in prompt + response
        sample.payload_has_tools = "# Tools" in prompt + response
        sample.metadata.update({
            "partial_rollout": True,
            "partial_rollout_prefix_length": len(response_token_ids),
            "current_turn": current_turn,
            "loss_masks": loss_masks,
            "tool_call_count": tool_call_count,
            "tool_token_count": _tool_token_count,
            "code_call_count": tool_call_count,
            "search_call_count": 0,
            "formatted_prompt": prompt,
            "formatted_prompt_token_count": len(prompt_tokens_ids),
            "retool_assistant_start_token_idx": assistant_start_token_idx,
            "retool_continue_partial_assistant": continuing_partial_assistant,
        })
        _record_timing(reason)
        sample.tool_time = _tool_time
        return sample

    output = None
    truncated_by_context = False
    truncated_by_empty_generation = False
    last_completed_generation = None
    final_generations_confirmed = set()
    tool_generations_confirmed = set()
    terminal_repair_events = sample.metadata.setdefault(
        "terminal_repair_events", []
    )
    pending_terminal_repair = None

    def _confirm_trajectory_final() -> None:
        if (
            last_completed_generation is None
            or last_completed_generation in final_generations_confirmed
        ):
            return
        if confirm_agentic_generation_final(
            sample.metadata,
            last_completed_generation,
            p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
        ):
            final_generations_confirmed.add(last_completed_generation)

    def _confirm_valid_tool() -> None:
        if (
            last_completed_generation is None
            or last_completed_generation in tool_generations_confirmed
        ):
            return
        if confirm_agentic_generation_tool(
            sample.metadata,
            last_completed_generation,
            p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
        ):
            tool_generations_confirmed.add(last_completed_generation)

    existing_overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
    if existing_overflow > 0:
        keep_response_tokens = max(0, max_context_length - len(prompt_tokens_ids))
        response_token_ids = response_token_ids[:keep_response_tokens]
        loss_masks = loss_masks[:keep_response_tokens]
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs = sample.rollout_log_probs[:keep_response_tokens]
        response = state.tokenizer.decode(response_token_ids)
        assistant_start_token_idx = min(assistant_start_token_idx, len(response_token_ids))
        truncated_by_context = True

    if sample.metadata.get(TOOL_DELAY_REMAINING_KEY) is not None:
        if not await _sleep_after_tool_delay(args, state, sample, abort_epoch=attempt_abort_epoch):
            return _save_partial_for_resume(start_turn, reason="tool_delay_abort")

    for turn in range(start_turn, TOOL_CONFIGS["max_turns"]):
        if _crossed_weight_update_boundary():
            return _save_partial_for_resume(turn, reason="weight_update_boundary")

        if not continuing_partial_assistant:
            assistant_start_token_idx = len(response_token_ids)

        # Check if total length exceeds max context length
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            truncated_by_context = True
            _confirm_trajectory_final()
            break

        # Clamp per-turn max_new_tokens to the remaining context budget so a
        # single turn cannot push total_length past max_context_length. Without
        # this, a turn can append up to rollout_max_response_len tokens on top
        # of a total that was just barely under the cap, producing samples
        # that exceed the training-side max_tokens_per_gpu * cp_size budget
        # and crash the partition/batch code (asserts or OOMs on an oversized
        # partition).
        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = min(
            sampling_params.get("max_new_tokens", remaining_budget),
            remaining_budget,
        )
        agentic_request_id = None
        if lifecycle_enabled():
            per_turn_sampling_params, agentic_request_id = add_agentic_kv_metadata(
                per_turn_sampling_params,
                trajectory_metadata=sample.metadata,
                generation=turn,
                tokenizer=state.tokenizer,
                tool_type="code_interpreter",
                tool_suffix_markers=("</tool_call>", "</code>"),
                terminal_markers=(r"\boxed{",),
            )

        # Use token IDs instead of text
        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,  # Request log probabilities for training
        }
        if agentic_request_id is not None:
            # The envelope carries a generation-scoped radix key plus the
            # trajectory metadata needed by the reverse-KV lifecycle.
            payload["extra_key"] = build_agentic_extra_key(
                agentic_request_id, per_turn_sampling_params
            )

        # Scheme 3: the first Retool model call uses strict PD.  Later calls
        # are sent to a colocated engine, where their prompt is prefetched on
        # the D-side GPU and decoding continues on that same engine.
        url = (
            pd_url
            if lifecycle_enabled()
            else (local_url if local_url is not None and turn >= 1 else pd_url)
        )
        route_mode = "d_local" if url == local_url else "strict_pd"
        repair_event = pending_terminal_repair
        pending_terminal_repair = None
        if repair_event is not None:
            repair_event["next_generation"] = turn
            repair_event["next_prompt_tokens"] = len(current_token_ids)
            repair_event["repair_attempted"] = True
        with dashboard_span(
            args,
            sample,
            "generation_turn",
            attrs={
                "task_type": "math",
                "turn": turn + 1,
                "max_new_tokens": per_turn_sampling_params["max_new_tokens"],
                "route_mode": route_mode,
                "terminal_repair": repair_event is not None,
            },
        ) as generation_span:
            with track_sglang_generation(sample):
                output = await post(url, payload)
            last_completed_generation = turn
            meta_info = output.get("meta_info", {})
            meta_attrs = sglang_meta_attrs(meta_info)
            meta_attrs["route_mode"] = route_mode
            generation_span.update(meta_attrs)
        if repair_event is not None:
            actual_cached = int(meta_info.get("cached_tokens") or 0)
            actual_prompt = int(
                meta_info.get("prompt_tokens") or len(current_token_ids)
            )
            reusable_parent = min(
                actual_prompt,
                int(repair_event["page_aligned_parent_kv_tokens"]),
            )
            repair_event.update(
                {
                    "actual_prompt_tokens": actual_prompt,
                    "actual_cached_tokens": actual_cached,
                    "actual_prefill_tokens": max(0, actual_prompt - actual_cached),
                    "counterfactual_prefill_tokens": max(
                        0, actual_prompt - reusable_parent
                    ),
                    "extra_prefill_tokens": max(
                        0, reusable_parent - actual_cached
                    ),
                }
            )

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            if not args.partial_rollout:
                _confirm_trajectory_final()
                sample.status = Sample.Status.ABORTED
                _record_timing("sglang_abort")
                sample.tool_time = _tool_time
                sample.tool_token_count = _tool_token_count
                sample.code_call_count = tool_call_count
                sample.search_call_count = 0
                return sample

            # Preserve the exact unfinished assistant turn. It may end in the
            # middle of a JSON/tool call, so never execute it here. The next
            # attempt prefills prompt + this prefix, generates the suffix, and
            # parses the complete assistant turn exactly once.
            if "output_token_logprobs" not in output["meta_info"]:
                return _save_partial_for_resume(turn, reason="sglang_abort_missing_logprobs")

            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs
            response_token_ids += cur_response_token_ids
            response = state.tokenizer.decode(response_token_ids)
            if _should_mask_offpolicy(args, sample):
                loss_masks += [0] * len(cur_response_token_ids)
                sample.metadata["offpolicy_masked"] = True
            else:
                loss_masks += [1] * len(cur_response_token_ids)
            if cur_response_token_ids:
                continuing_partial_assistant = True
            return _save_partial_for_resume(turn, reason="sglang_abort")

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs

        else:
            # sglang returned text but no output_token_logprobs — we cannot
            # recover per-token logprobs for this turn, which would desync
            # rollout_log_probs from response_token_ids and blow up
            # `slice_log_prob_with_cp` downstream. Abort the sample so the
            # fully_async rollout manager returns the whole group to the
            # buffer for retry instead of poisoning the trainer.
            sample.status = Sample.Status.ABORTED
            _confirm_trajectory_final()
            _record_timing("missing_logprobs")
            sample.tool_time = _tool_time
            sample.tool_token_count = _tool_token_count
            sample.code_call_count = tool_call_count
            sample.search_call_count = 0
            return sample

        response_token_ids += cur_response_token_ids
        response = state.tokenizer.decode(response_token_ids)
        loss_masks += [1] * len(cur_response_token_ids)

        # An empty generation does not pass through SGLang's normal Decode
        # finished-request hook, hence it cannot have a reusable parent KV
        # snapshot.  Never turn it into a repair/continuation request.
        if not generation_has_visible_content(
            cur_response_token_ids, state.tokenizer
        ):
            truncated_by_empty_generation = True
            _confirm_trajectory_final()
            break

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            _confirm_trajectory_final()
            break

        assistant_text = state.tokenizer.decode(response_token_ids[assistant_start_token_idx:])
        parsed_action, parsed_content = postprocess_predictions(assistant_text)
        explicit_terminal = r"\boxed{" in assistant_text
        if parsed_action == "code" and parsed_content.strip():
            # Publish before the potentially long tool execution.  D may use
            # Shared Arena/Mooncake only after this parser-level ACK exists.
            _confirm_valid_tool()
        elif explicit_terminal:
            _confirm_trajectory_final()
        _tool_start = time.monotonic()
        sample.metadata[PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY] = True
        with dashboard_span(
            args,
            sample,
            "tool_call",
            attrs={"task_type": "math", "turn": turn + 1},
        ) as tool_span:
            next_obs, done = await execute_predictions(assistant_text)
            is_tool_call = "<interpreter>" in (next_obs or "")
            tool_span.update(
                {
                    "done": done,
                    "observation_chars": len(next_obs or ""),
                    "tool_calls": int(is_tool_call),
                    "is_tool_call": is_tool_call,
                }
            )
        if done:
            continuing_partial_assistant = False
            sample.metadata.pop(PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY, None)
            # Native serving treats a normal stop with no parser-confirmed
            # tool call as terminal.  Publish that decision immediately so D
            # cannot move a provisional, tool-looking candidate into Host
            # staging while the rest of this function finalizes the sample.
            _confirm_trajectory_final()
            break
        if parsed_action != "code" and not explicit_terminal:
            # A malformed action receives a repair observation and therefore
            # continues the trajectory.  Confirm it before the next P call so
            # the provisional UNKNOWN snapshot can take the Direct path.
            _confirm_valid_tool()

        # Count tool calls (when we get interpreter output, it means a tool
        # was called)
        if "<interpreter>" in next_obs:
            tool_call_count += 1
            _tool_time += time.monotonic() - _tool_start

        assert next_obs != "", "Next observation should not be empty."
        (
            response,
            response_token_ids,
            loss_masks,
            sample.rollout_log_probs,
            obs_truncated,
            obs_token_count,
        ) = append_observation_with_budget(
            state,
            prompt_tokens_ids,
            response,
            response_token_ids,
            loss_masks,
            sample.rollout_log_probs,
            next_obs,
            max_context_length,
        )
        _tool_token_count += obs_token_count
        if explicit_terminal:
            parent_prompt_tokens = len(current_token_ids)
            parent_completion_tokens = len(cur_response_token_ids)
            parent_kv_end = parent_prompt_tokens + parent_completion_tokens - 1
            event = {
                "generation": turn,
                "repair_attempted": False,
                "parent_prompt_tokens": parent_prompt_tokens,
                "parent_completion_tokens": parent_completion_tokens,
                "repair_observation_tokens": obs_token_count,
                "page_aligned_parent_kv_tokens": max(
                    0, (parent_kv_end // 64) * 64
                ),
            }
            terminal_repair_events.append(event)
            pending_terminal_repair = event

        if sample.rollout_log_probs is not None:
            assert len(response_token_ids) == len(
                sample.rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        if obs_truncated:
            continuing_partial_assistant = False
            sample.metadata.pop(PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY, None)
            truncated_by_context = True
            _confirm_trajectory_final()
            break

        continuing_partial_assistant = False
        assistant_start_token_idx = len(response_token_ids)

        # A tool that was already running at the update boundary is allowed to
        # finish. Commit its observation, then return the partial trajectory so
        # the next dispatch performs a fresh prefill under the new weights.
        if _crossed_weight_update_boundary():
            return _save_partial_for_resume(turn + 1, reason="tool_completed_after_weight_update")

        sample.metadata.pop(PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY, None)
        if not await _sleep_after_tool_delay(args, state, sample, abort_epoch=attempt_abort_epoch):
            return _save_partial_for_resume(turn + 1, reason="tool_delay_abort")

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            _confirm_trajectory_final()
            break

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Store payload information for wandb logging
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count
    sample.tool_token_count = _tool_token_count
    sample.code_call_count = tool_call_count
    sample.search_call_count = 0
    sample.metadata.pop(PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY, None)

    # Set status based on finish reason. Context-budget truncation wins over
    # the last model finish reason, since tool observations can be the part
    # that fills the context.
    if truncated_by_context or truncated_by_empty_generation:
        sample.status = Sample.Status.TRUNCATED
    elif output is not None:
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.TRUNCATED

    if truncated_by_context:
        attempt_reason = "context"
    elif truncated_by_empty_generation:
        attempt_reason = "empty_generation"
    elif output is not None:
        attempt_reason = output["meta_info"]["finish_reason"]["type"]
    else:
        attempt_reason = "no_output"
    _record_timing(attempt_reason)
    sample.tool_time = _tool_time
    sample.metadata.update({
        "tool_call_count": tool_call_count,
        "tool_token_count": _tool_token_count,
        "code_call_count": tool_call_count,
        "search_call_count": 0,
        "tool_time": _tool_time,
        "sample_time": sample.sample_time,
        "tool_time_ratio": _tool_time / sample.sample_time if sample.sample_time > 0 else 0.0,
        "retool_assistant_start_token_idx": assistant_start_token_idx,
        "retool_continue_partial_assistant": False,
    })
    _confirm_trajectory_final()
    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = sample.prompt + sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = str(sample.label) if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    # use \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = num_turns * 0.1 #(num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
