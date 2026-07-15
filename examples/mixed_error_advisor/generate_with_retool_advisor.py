# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import asyncio
import math
import os
import random
import re
import time
from typing import Any

import httpx

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality
from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

TOOL_DELAY_REMAINING_KEY = "pending_tool_delay_remaining"

ADVISOR_FALLBACK = (
    "The tool call failed. Read the error, correct the tool arguments, "
    "and do not repeat the same failed call."
)


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _extract_system_prompt(formatted_prompt: str) -> str:
    match = re.search(r"<\|im_start\|>system\s*\n?(.*?)<\|im_end\|>", formatted_prompt, re.DOTALL)
    return match.group(1).strip() if match else formatted_prompt


def _extract_tool_call(prediction: str) -> str | None:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", prediction, re.DOTALL)
    return re.sub(r"\s+", "", match.group(1)) if match else None


def _strip_advisor_thinking(text: str) -> str:
    text = text or ""
    if "<think>" in text and "</think>" not in text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text.split("<|im_end|>", 1)[0].strip()


async def get_advisor_feedback(system_prompt: str, error: str, failed_action: str) -> str:
    """Ask the isolated local advisor for one short, actionable correction."""
    advisor_url = os.getenv("TOOL_ERROR_ADVISOR_URL", "").strip()
    if not advisor_url:
        return ADVISOR_FALLBACK

    max_tokens = _int_env("TOOL_ERROR_ADVISOR_MAX_TOKENS", 64)
    advisor_prompt = (
        "<|im_start|>system\n"
        "You diagnose failed tool calls. Reply with one concise actionable sentence only. "
        "State what is wrong and exactly how to fix it. Do not solve the original problem. "
        "Never state or repeat the original problem's numeric answer, even if it is visible. "
        "Do not use markdown fences or XML tags.<|im_end|>\n"
        "<|im_start|>user\n"
        "/no_think\n"
        f"Original system prompt:\n{system_prompt}\n\n"
        f"Failed action:\n{failed_action}\n\n"
        f"Tool error:\n{error}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    payload = {
        "text": advisor_prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_tokens,
            "stop": ["<|im_end|>"],
            "skip_special_tokens": False,
        },
    }
    try:
        timeout = float(os.getenv("TOOL_ERROR_ADVISOR_TIMEOUT", "30"))
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(advisor_url, json=payload)
            response.raise_for_status()
            output = response.json()
        feedback = _strip_advisor_thinking(output.get("text", ""))
        return feedback or ADVISOR_FALLBACK
    except Exception as exc:
        print(f"[tool-error-advisor] request failed: {exc}")
        return ADVISOR_FALLBACK


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
        system_content = system_prompt
    else:
        system_content = (
            "You are a helpful assistant that can use Python "
            "tools to solve mathematical problems. When you need "
            "to perform calculations, use the code_interpreter "
            "tool to execute code and get results."
        )

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


async def execute_predictions(
    prediction: str, system_prompt: str = "", previous_tool_call: str | None = None
) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        current_tool_call = _extract_tool_call(prediction)
        if current_tool_call is not None and current_tool_call == previous_tool_call:
            error = "Error: The exact same tool call was already executed; repeating it provides no new information."
            feedback = await get_advisor_feedback(system_prompt, error, prediction)
            return f"\n<advisor_feedback>\n{feedback}\n</advisor_feedback>\n\n", False
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content.strip()
        if code:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            if not result.strip():
                result = "Error: The Python code ran successfully but produced no stdout. Use print(...) to emit the result."
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            if result.lstrip().startswith("Error"):
                feedback = await get_advisor_feedback(system_prompt, result, prediction)
                next_obs += f"<advisor_feedback>\n{feedback}\n</advisor_feedback>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found" "\n</interpreter>\n\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        error = "No valid final answer or executable tool call could be parsed from the previous action."
        feedback = await get_advisor_feedback(system_prompt, error, prediction)
        next_obs = f"\n<advisor_feedback>\n{feedback}\n</advisor_feedback>\n\n"
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls with partial rollout support"""

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Initialize total_off_policy_tokens if it doesn't exist
    if not hasattr(sample, 'total_off_policy_tokens'):
        sample.total_off_policy_tokens = 0

    # Set up tool specs
    tool_specs = tool_registry.get_tool_specs()

    # Ensure metadata exists
    if not hasattr(sample, 'metadata') or sample.metadata is None:
        sample.metadata = {}

    # Check if this is a partial rollout resume
    if args.partial_rollout and sample.status == Sample.Status.ABORTED and sample.response:
        # Partial rollout: resume from existing response
        metadata = sample.metadata

        if metadata.get("formatted_prompt"):
            prompt = metadata["formatted_prompt"]
        else:
            prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

        prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Restore state from saved metadata if available
        response = sample.response
        response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]

        if _should_mask_offpolicy(args, sample):
            # Off-policy masking: all existing tokens are off-policy,
            # only newly generated tokens (added after this point) will be marked as on-policy (1).
            loss_masks = [0] * len(response_token_ids)
            sample.metadata["offpolicy_masked"] = True
            # Still need tool_call_count and start_turn to resume the multi-turn loop correctly
            if metadata.get("tool_call_count"):
                tool_call_count = metadata["tool_call_count"]
                start_turn = metadata.get("current_turn", tool_call_count)
            else:
                tool_call_count = count_tool_turns(response)
                start_turn = tool_call_count
        elif metadata.get("partial_rollout") and metadata.get("loss_masks") and metadata.get("tool_call_count"):
            loss_masks = metadata["loss_masks"]
            if len(loss_masks) != len(response_token_ids):
                print(f"[WARNING] Saved loss_masks length ({len(loss_masks)}) != response tokens ({len(response_token_ids)})")
                loss_masks = reconstruct_loss_masks(response, state.tokenizer)
            tool_call_count = metadata["tool_call_count"]
            start_turn = metadata.get("current_turn", tool_call_count)
        else:
            loss_masks = reconstruct_loss_masks(response, state.tokenizer)
            tool_call_count = count_tool_turns(response)
            start_turn = tool_call_count

        # Re-sync rollout_log_probs if tokenizer round-trip changed token count
        if sample.rollout_log_probs is not None and len(sample.rollout_log_probs) != len(response_token_ids):
            print(f"[WARNING] rollout_log_probs length ({len(sample.rollout_log_probs)}) != response tokens ({len(response_token_ids)}), resetting to zeros")
            sample.rollout_log_probs = [0.0] * len(response_token_ids)

        # Update off-policy token count
        sample.total_off_policy_tokens += len(response_token_ids)
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
        _accrued_sample_time = 0.0
        _tool_time = 0.0
        _tool_token_count = 0

    _start_time = time.monotonic()

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
    max_context_length = min(max_context_length, len(prompt_tokens_ids) + _int_env("MIXED_RETOOL_MAX_RESPONSE_LEN", 8192))
    advisor_system_prompt = _extract_system_prompt(prompt)
    previous_tool_call = None
    existing_calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
    if existing_calls:
        previous_tool_call = re.sub(r"\s+", "", existing_calls[-1])

    def _save_partial_for_resume(current_turn: int, status=Sample.Status.ABORTED) -> Sample:
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
            "current_turn": current_turn,
            "loss_masks": loss_masks,
            "tool_call_count": tool_call_count,
            "tool_token_count": _tool_token_count,
            "code_call_count": tool_call_count,
            "search_call_count": 0,
            "formatted_prompt": prompt,
        })
        sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
        sample.tool_time = _tool_time
        return sample

    output = None
    truncated_by_context = False

    existing_overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
    if existing_overflow > 0:
        keep_response_tokens = max(0, max_context_length - len(prompt_tokens_ids))
        response_token_ids = response_token_ids[:keep_response_tokens]
        loss_masks = loss_masks[:keep_response_tokens]
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs = sample.rollout_log_probs[:keep_response_tokens]
        response = state.tokenizer.decode(response_token_ids)
        truncated_by_context = True

    if sample.metadata.get(TOOL_DELAY_REMAINING_KEY) is not None:
        if not await _sleep_after_tool_delay(args, state, sample):
            return _save_partial_for_resume(start_turn)

    for turn in range(start_turn, TOOL_CONFIGS["max_turns"]):
        # Check if total length exceeds max context length
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            truncated_by_context = True
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

        # Use token IDs instead of text
        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,  # Request log probabilities for training
        }

        # Log payload to wandb for debugging
        try:
            import wandb

            if wandb.run is not None:
                # Count available tools (from tool_specs)
                available_tools = len(tool_specs)
                # Count tools used in the current response
                tools_used = response.count("<interpreter>")

                wandb.log(
                    {
                        "debug/payload_length": len(prompt + response),
                        "debug/available_tools": available_tools,
                        "debug/tools_used": tools_used,
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass  # wandb not available

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            if not args.partial_rollout:
                sample.status = Sample.Status.ABORTED
                sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
                sample.tool_time = _tool_time
                sample.tool_token_count = _tool_token_count
                sample.code_call_count = tool_call_count
                sample.search_call_count = 0
                return sample
            else:
                # Partial rollout enabled: process partial response and save state
                if "output_token_logprobs" in output["meta_info"]:
                    cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                    cur_response = state.tokenizer.decode(cur_response_token_ids)
                    cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                    if sample.rollout_log_probs is None:
                        sample.rollout_log_probs = []
                    sample.rollout_log_probs += cur_log_probs
                else:
                    sample.status = Sample.Status.ABORTED
                    sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
                    sample.tool_time = _tool_time
                    sample.tool_token_count = _tool_token_count
                    sample.code_call_count = tool_call_count
                    sample.search_call_count = 0
                    return sample

                if cur_response:
                    response += cur_response
                    response_token_ids += cur_response_token_ids
                    if _should_mask_offpolicy(args, sample):
                        # All existing + new tokens are off-policy; only tokens
                        # generated on the next complete resume will be on-policy.
                        loss_masks += [0] * len(cur_response_token_ids)
                        sample.metadata["offpolicy_masked"] = True
                    else:
                        loss_masks += [1] * len(cur_response_token_ids)

                    _tool_start = time.monotonic()
                    current_tool_call = _extract_tool_call(cur_response)
                    next_obs, done = await execute_predictions(
                        cur_response, advisor_system_prompt, previous_tool_call
                    )
                    if current_tool_call is not None:
                        previous_tool_call = current_tool_call
                    if next_obs:
                        if "<interpreter>" in next_obs:
                            tool_call_count += 1
                        _tool_time += time.monotonic() - _tool_start

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
                        if obs_truncated:
                            truncated_by_context = True
                        else:
                            await _sleep_after_tool_delay(args, state, sample)

                # Save state for resumption
                sample.status = Sample.Status.TRUNCATED if truncated_by_context else Sample.Status.ABORTED
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
                    "current_turn": turn,
                    "loss_masks": loss_masks,
                    "tool_call_count": tool_call_count,
                    "tool_token_count": _tool_token_count,
                    "code_call_count": tool_call_count,
                    "search_call_count": 0,
                    "formatted_prompt": prompt,
                })
                sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
                sample.tool_time = _tool_time
                return sample

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
            sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
            sample.tool_time = _tool_time
            sample.tool_token_count = _tool_token_count
            sample.code_call_count = tool_call_count
            sample.search_call_count = 0
            return sample

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        _tool_start = time.monotonic()
        current_tool_call = _extract_tool_call(cur_response)
        next_obs, done = await execute_predictions(cur_response, advisor_system_prompt, previous_tool_call)
        if current_tool_call is not None:
            previous_tool_call = current_tool_call
        if done:
            break

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

        if sample.rollout_log_probs is not None:
            assert len(response_token_ids) == len(
                sample.rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        if obs_truncated:
            truncated_by_context = True
            break

        if not await _sleep_after_tool_delay(args, state, sample):
            return _save_partial_for_resume(turn)

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
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

    # Set status based on finish reason. Context-budget truncation wins over
    # the last model finish reason, since tool observations can be the part
    # that fills the context.
    if truncated_by_context:
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

    sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
    sample.tool_time = _tool_time
    sample.metadata.update({
        "tool_call_count": tool_call_count,
        "tool_token_count": _tool_token_count,
        "code_call_count": tool_call_count,
        "search_call_count": 0,
        "tool_time": _tool_time,
        "sample_time": sample.sample_time,
        "tool_time_ratio": _tool_time / sample.sample_time if sample.sample_time > 0 else 0.0,
    })
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
