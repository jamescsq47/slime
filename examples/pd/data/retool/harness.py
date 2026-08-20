"""Retool inference harness.

This is intentionally serving-only.  The original copy came from an RL
rollout and carried partial-rollout resume, off-policy masks, token logprobs,
reward computation and trainer payloads.  None of those fields are consumed
by ``examples/pd/inference.py``; retaining them wasted host/HBM memory and made
the agent loop much harder to audit.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
from typing import Any

from jinja2 import Template

from agentic_kv_request import (
    add_agentic_kv_metadata,
    build_agentic_extra_key,
    confirm_agentic_generation_final,
    confirm_agentic_generation_tool,
    generation_has_visible_content,
    lifecycle_enabled,
)
from pd_metrics import sglang_meta_attrs
from slime.dashboard.api import span as dashboard_span
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .tools import TOOL_CONFIGS, tool_registry


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

_QWEN_MESSAGE_PATTERN = re.compile(
    r"<\|im_start\|>(system|user|assistant)\s*\n?(.*?)<\|im_end\|>",
    re.DOTALL,
)


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def extract_user_prompt(prompt: str) -> str:
    """Unwrap a single-turn Qwen chat-template prompt if necessary."""

    if not isinstance(prompt, str) or "<|im_start|>" not in prompt:
        return prompt
    messages = _QWEN_MESSAGE_PATTERN.findall(prompt)
    user_messages = [content.strip() for role, content in messages if role == "user"]
    return user_messages[0] if len(user_messages) == 1 else prompt


def format_conversation_with_tools(
    prompt: str,
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> str:
    system_content = (
        f"{system_prompt.rstrip()}\n\n{RETOOL_PROTOCOL}"
        if system_prompt
        else RETOOL_PROTOCOL
    )
    rendered_messages = [{"role": "system", "content": system_content}]
    if prompt:
        rendered_messages.append({"role": "user", "content": extract_user_prompt(prompt)})
    if messages:
        rendered_messages.extend(messages)
    return Template(TOOL_TEMPLATE).render(messages=rendered_messages, tools=tools or [])


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    answer_match = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", prediction, re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()

    tool_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", prediction, re.DOTALL)
    if tool_match:
        try:
            value = json.loads(tool_match.group(1).replace("\n", "\\n"))
            if value.get("name") == "code_interpreter":
                code = value.get("arguments", {}).get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, AttributeError):
            pass

    for pattern in (r"<code>(.*?)</code>", r"```python\s*(.*?)\s*```"):
        match = re.search(pattern, prediction, re.DOTALL)
        if match:
            return "code", match.group(1).strip()
    return None, ""


def postprocess_responses(response: str) -> str:
    """Retained public parser helper used by existing prompt tests."""

    patterns = (
        r"<tool_call>\s*\{.*?\}\s*</tool_call>",
        r"<code>.*?</code>",
        r"```python\s*.*?```",
        r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, response, re.DOTALL))
        if matches:
            return response[: matches[-1].end()]
    return response


async def execute_predictions(prediction: str) -> tuple[str, bool]:
    action, content = postprocess_predictions(prediction)
    if action == "code":
        result = await tool_registry.execute_tool(
            "code_interpreter", {"code": content.strip()}
        )
        return f"\n\n<interpreter>\n{result}\n</interpreter>\n\n", False
    if action == "answer":
        return "", True
    return (
        "\nThe previous response was not a valid action. Either continue reasoning and end with "
        "the line `#### \\boxed{answer}`, or call code_interpreter using exactly one valid "
        "<tool_call> JSON block with no text after it.\n",
        False,
    )


def _append_observation(
    state: GenerateState,
    prompt_tokens: list[int],
    response_tokens: list[int],
    observation: str,
    max_context_length: int,
) -> tuple[str, list[int], bool, int]:
    remaining = max_context_length - len(prompt_tokens) - len(response_tokens)
    if remaining <= 0:
        return "", response_tokens, True, 0
    observation_tokens = state.tokenizer(observation, add_special_tokens=False)["input_ids"]
    truncated = len(observation_tokens) > remaining
    observation_tokens = observation_tokens[:remaining]
    return state.tokenizer.decode(observation_tokens), response_tokens + observation_tokens, truncated, len(
        observation_tokens
    )


def _sample_tool_delay(args: Any) -> float:
    if not getattr(args, "enable_tool_delay", False):
        return 0.0
    mean = max(0.0, float(getattr(args, "tool_delay_mean", 25.0)))
    variance = max(0.0, float(getattr(args, "tool_delay_variance", 500.0)))
    if mean == 0 or variance == 0:
        return mean
    sigma2 = math.log1p(variance / (mean * mean))
    return random.lognormvariate(math.log(mean) - sigma2 / 2, math.sqrt(sigma2))


async def generate(
    args: Any, sample: Sample, sampling_params: dict[str, Any]
) -> Sample:
    """Run one complete Retool trajectory; no training state is produced."""

    state = GenerateState(args)
    sample.metadata = dict(sample.metadata or {})
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
    prompt_tokens = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_tokens: list[int] = []
    response = ""
    tool_call_count = 0
    tool_token_count = 0
    tool_seconds = 0.0
    started_at = time.monotonic()
    last_output: dict[str, Any] | None = None
    last_generation: int | None = None
    final_confirmed: set[int] = set()
    tool_confirmed: set[int] = set()
    terminal_repairs = sample.metadata.setdefault("terminal_repair_events", [])
    pending_repair: dict[str, Any] | None = None
    truncated_by_context = False
    truncated_by_empty = False

    pd_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    local_port = getattr(args, "retool_local_router_port", None)
    local_url = (
        f"http://{args.sglang_router_ip}:{local_port}/generate"
        if local_port is not None
        else None
    )
    configured_context = (
        args.rollout_max_context_len
        if args.rollout_max_context_len is not None
        else args.context_parallel_size * args.max_tokens_per_gpu
    )
    response_cap = int(
        sampling_params.get("max_new_tokens")
        or _int_env("MIXED_RETOOL_MAX_RESPONSE_LEN", 8192)
    )
    max_context_length = min(configured_context, len(prompt_tokens) + response_cap)

    def confirm_final() -> None:
        if last_generation is None or last_generation in final_confirmed:
            return
        if confirm_agentic_generation_final(
            sample.metadata,
            last_generation,
            p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
        ):
            final_confirmed.add(last_generation)

    def confirm_tool() -> None:
        if last_generation is None or last_generation in tool_confirmed:
            return
        if confirm_agentic_generation_tool(
            sample.metadata,
            last_generation,
            p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
        ):
            tool_confirmed.add(last_generation)

    for turn in range(TOOL_CONFIGS["max_turns"]):
        assistant_start = len(response_tokens)
        total_length = len(prompt_tokens) + len(response_tokens)
        if total_length >= max_context_length:
            truncated_by_context = True
            confirm_final()
            break

        remaining = max_context_length - total_length
        params = dict(sampling_params)
        params["max_new_tokens"] = min(int(params.get("max_new_tokens") or remaining), remaining)
        agentic_request_id = None
        if lifecycle_enabled():
            params, agentic_request_id = add_agentic_kv_metadata(
                params,
                trajectory_metadata=sample.metadata,
                generation=turn,
                tokenizer=state.tokenizer,
                tool_type="code_interpreter",
                tool_suffix_markers=("</tool_call>", "</code>"),
                terminal_markers=(r"\boxed{",),
            )

        current_tokens = prompt_tokens + response_tokens
        payload: dict[str, Any] = {
            "input_ids": current_tokens,
            "sampling_params": params,
            # Inference never consumes token logprobs.  Explicitly disabling
            # them avoids the large prefill-side allocation seen at high c.
            "return_logprob": False,
        }
        if agentic_request_id is not None:
            payload["extra_key"] = build_agentic_extra_key(agentic_request_id, params)

        url = (
            pd_url
            if lifecycle_enabled()
            else (local_url if local_url is not None and turn >= 1 else pd_url)
        )
        route_mode = "d_local" if url == local_url else "strict_pd"
        repair = pending_repair
        pending_repair = None
        if repair is not None:
            repair.update(
                {
                    "next_generation": turn,
                    "next_prompt_tokens": len(current_tokens),
                    "repair_attempted": True,
                }
            )

        with dashboard_span(
            args,
            sample,
            "generation_turn",
            attrs={
                "task_type": sample.metadata.get("task_type", "math"),
                "turn": turn + 1,
                "max_new_tokens": params["max_new_tokens"],
                "route_mode": route_mode,
                "terminal_repair": repair is not None,
            },
        ) as generation_span:
            last_output = await post(url, payload)
            last_generation = turn
            meta = last_output.get("meta_info", {})
            attrs = sglang_meta_attrs(meta)
            attrs["route_mode"] = route_mode
            generation_span.update(attrs)

        if repair is not None:
            actual_cached = int(meta.get("cached_tokens") or 0)
            actual_prompt = int(meta.get("prompt_tokens") or len(current_tokens))
            reusable_parent = min(actual_prompt, int(repair["page_aligned_parent_kv_tokens"]))
            repair.update(
                {
                    "actual_prompt_tokens": actual_prompt,
                    "actual_cached_tokens": actual_cached,
                    "actual_prefill_tokens": max(0, actual_prompt - actual_cached),
                    "counterfactual_prefill_tokens": max(0, actual_prompt - reusable_parent),
                    "extra_prefill_tokens": max(0, reusable_parent - actual_cached),
                }
            )

        finish_type = meta["finish_reason"]["type"]
        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            confirm_final()
            break

        new_tokens = list(last_output.get("output_ids") or [])
        response_tokens.extend(new_tokens)
        response = state.tokenizer.decode(response_tokens)
        if not generation_has_visible_content(new_tokens, state.tokenizer):
            truncated_by_empty = True
            confirm_final()
            break
        if finish_type == "length":
            confirm_final()
            break

        assistant_text = state.tokenizer.decode(response_tokens[assistant_start:])
        action, action_content = postprocess_predictions(assistant_text)
        explicit_terminal = r"\boxed{" in assistant_text
        if action == "code" and action_content.strip():
            confirm_tool()
        elif explicit_terminal:
            confirm_final()

        tool_started = time.monotonic()
        with dashboard_span(
            args,
            sample,
            "tool_call",
            attrs={"task_type": sample.metadata.get("task_type", "math"), "turn": turn + 1},
        ) as tool_span:
            observation, done = await execute_predictions(assistant_text)
            is_tool_call = "<interpreter>" in observation
            tool_span.update(
                {
                    "done": done,
                    "observation_chars": len(observation),
                    "tool_calls": int(is_tool_call),
                    "is_tool_call": is_tool_call,
                }
            )
        if done:
            confirm_final()
            break
        if action != "code" and not explicit_terminal:
            confirm_tool()
        if is_tool_call:
            tool_call_count += 1
            tool_seconds += time.monotonic() - tool_started

        appended_text, response_tokens, observation_truncated, observation_tokens = _append_observation(
            state,
            prompt_tokens,
            response_tokens,
            observation,
            max_context_length,
        )
        response += appended_text
        tool_token_count += observation_tokens
        if explicit_terminal:
            parent_kv_end = len(current_tokens) + len(new_tokens) - 1
            repair = {
                "generation": turn,
                "repair_attempted": False,
                "parent_prompt_tokens": len(current_tokens),
                "parent_completion_tokens": len(new_tokens),
                "repair_observation_tokens": observation_tokens,
                "page_aligned_parent_kv_tokens": max(0, (parent_kv_end // 64) * 64),
            }
            terminal_repairs.append(repair)
            pending_repair = repair
        if observation_truncated:
            truncated_by_context = True
            confirm_final()
            break

        delay = _sample_tool_delay(args)
        if delay:
            await asyncio.sleep(delay)
        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            confirm_final()
            break

    sample.tokens = prompt_tokens + response_tokens
    sample.response = response
    sample.response_length = len(response_tokens)
    sample.tool_call_count = tool_call_count
    sample.code_call_count = tool_call_count
    sample.search_call_count = 0
    sample.tool_time = tool_seconds
    if truncated_by_context or truncated_by_empty:
        sample.status = Sample.Status.TRUNCATED
    elif sample.status == Sample.Status.ABORTED:
        pass
    elif last_output is None:
        sample.status = Sample.Status.TRUNCATED
    elif last_output["meta_info"]["finish_reason"]["type"] == "length":
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.COMPLETED
    sample.metadata.update(
        {
            "tool_call_count": tool_call_count,
            "tool_token_count": tool_token_count,
            "code_call_count": tool_call_count,
            "search_call_count": 0,
            "tool_time": tool_seconds,
            "sample_time": time.monotonic() - started_at,
        }
    )
    confirm_final()
    return sample
