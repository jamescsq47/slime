"""Terminal-Bench 2 multi-turn inference harness."""

from __future__ import annotations

import os
import re
import time
from typing import Any

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

from .client import Tbench2Client


_BASH_BLOCK = re.compile(r"```(?:bash|sh)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OBSERVATION_CHAR_LIMIT = 8192


def command_from_reply(reply: str) -> str | None:
    """Return one executable shell block, never model reasoning or bare prose."""

    reply = reply.replace("<|im_end|>", "").strip()
    reply = _THINK_BLOCK.sub("", reply).strip()
    if "<think>" in reply or "</think>" in reply:
        return None
    match = _BASH_BLOCK.fullmatch(reply)
    if match is None:
        return None
    command = match.group(1).strip()
    return command or None


def _truncate_observation(value: str) -> str:
    if len(value) <= _OBSERVATION_CHAR_LIMIT:
        return value
    keep = (_OBSERVATION_CHAR_LIMIT - 80) // 2
    dropped = len(value) - 2 * keep
    return f"{value[:keep]}\n...[truncated {dropped} characters]...\n{value[-keep:]}"


def _render_initial_prompt(
    tokenizer: Any,
    base_messages: list[dict[str, Any]],
    instruction: str,
    working_directory: str,
) -> list[int]:
    messages = list(base_messages)
    messages.append(
        {
            "role": "user",
            "content": (
                f"Task instruction:\n{instruction}\n\n"
                f"Initial working directory: {working_directory or '(not reported)'}\n\n"
                "Each command runs in a fresh shell in that working directory; a standalone `cd` "
                "does not persist, so combine it with the command. Respond with exactly one command "
                "inside a ```bash block. When finished, respond with TASK_COMPLETE."
            ),
        }
    )
    kwargs: dict[str, Any] = {}
    enable_thinking = os.getenv("TERMINAL_ENABLE_THINKING")
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking.lower() in {"1", "true", "yes", "y"}
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def _observation_tokens(tokenizer: Any, output: str) -> list[int]:
    text = f"\n<|im_end|>\n<|im_start|>user\n{_truncate_observation(output)}<|im_end|>\n<|im_start|>assistant\n"
    return list(tokenizer.encode(text, add_special_tokens=False))


async def generate(
    args: Any, sample: Sample, sampling_params: dict[str, Any]
) -> Sample:
    metadata = dict(sample.metadata or {})
    sample.metadata = metadata
    task_id = str(metadata.get("task_id") or "")
    if not task_id:
        raise ValueError("Terminal-Bench sample requires metadata.task_id")
    dataset_id = str(metadata.get("dataset_id") or metadata.get("task_type") or "terminal")
    options = getattr(args, "workload_dataset_options", {}).get(dataset_id, {})
    base_url = str(options.get("environment_url") or os.getenv("TBENCH2_ENV_URL") or "")
    if not base_url:
        raise ValueError(
            "Terminal-Bench requires options.environment_url or TBENCH2_ENV_URL; "
            "the harness never starts the environment service itself"
        )

    state = GenerateState(args)
    client = Tbench2Client(
        base_url,
        message_timeout=float(options.get("environment_timeout_seconds", 4200)),
    )
    started_at = time.monotonic()
    prompt_tokens: list[int] = []
    response_tokens: list[int] = []
    command_count = 0
    tool_seconds = 0.0
    last_generation: int | None = None
    status = Sample.Status.COMPLETED
    stop_reason = "max_turns"
    reward = None
    try:
        await client.connect()
        reset = await client.reset(task_id)
        prompt_tokens = _render_initial_prompt(
            state.tokenizer,
            list(sample.prompt) if isinstance(sample.prompt, list) else [],
            reset.instruction,
            str((reset.info or {}).get("working_directory") or ""),
        )
        max_context = int(
            getattr(args, "max_seq_len", None)
            or getattr(args, "sglang_context_length", None)
            or 40960
        )
        max_turns = int(options.get("max_turns", 64))
        per_turn_cap = int(options.get("max_tokens_per_turn", 2048))
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        for turn in range(max_turns):
            input_ids = prompt_tokens + response_tokens
            remaining = max_context - len(input_ids)
            if remaining <= 0:
                status = Sample.Status.TRUNCATED
                stop_reason = "budget"
                break
            params = dict(sampling_params)
            params["max_new_tokens"] = min(per_turn_cap, remaining)
            request_id = None
            if lifecycle_enabled():
                params, request_id = add_agentic_kv_metadata(
                    params,
                    trajectory_metadata=metadata,
                    generation=turn,
                    tokenizer=state.tokenizer,
                    tool_type="shell",
                    tool_suffix_markers=("```",),
                    terminal_markers=("TASK_COMPLETE",),
                )
            payload: dict[str, Any] = {
                "input_ids": input_ids,
                "sampling_params": params,
                "return_logprob": False,
            }
            if request_id is not None:
                payload["extra_key"] = build_agentic_extra_key(request_id, params)

            with dashboard_span(
                args,
                sample,
                "generation_turn",
                attrs={
                    "task_type": metadata.get("task_type", "terminal"),
                    "turn": turn + 1,
                    "max_new_tokens": params["max_new_tokens"],
                    "route_mode": "strict_pd",
                },
            ) as span:
                output = await post(url, payload)
                span.update(sglang_meta_attrs(output.get("meta_info", {})))
            last_generation = turn
            meta = output["meta_info"]
            finish_type = meta["finish_reason"]["type"]
            new_tokens = list(output.get("output_ids") or [])
            response_tokens.extend(new_tokens)
            reply = state.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            if finish_type == "abort":
                status = Sample.Status.ABORTED
                stop_reason = "abort"
                break
            if not generation_has_visible_content(new_tokens, state.tokenizer):
                status = Sample.Status.TRUNCATED
                stop_reason = "empty_generation"
                break
            if "TASK_COMPLETE" in reply:
                confirm_agentic_generation_final(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                evaluated = await client.evaluate()
                reward = evaluated.reward
                stop_reason = "task_complete"
                break
            if finish_type == "length":
                status = Sample.Status.TRUNCATED
                stop_reason = "length"
                break

            command = command_from_reply(reply)
            if command is not None:
                confirm_agentic_generation_tool(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                tool_started = time.monotonic()
                result = await client.execute(command)
                tool_seconds += time.monotonic() - tool_started
                command_count += 1
                observation = result.output
            else:
                confirm_agentic_generation_tool(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                observation = (
                    "Invalid response. Return exactly one shell command in a single ```bash code block, "
                    "or TASK_COMPLETE when the task is finished."
                )
            observation_ids = _observation_tokens(state.tokenizer, observation)
            remaining = max_context - len(prompt_tokens) - len(response_tokens)
            if len(observation_ids) > remaining:
                observation_ids = observation_ids[: max(0, remaining)]
                status = Sample.Status.TRUNCATED
                stop_reason = "budget"
            response_tokens.extend(observation_ids)
            if status is Sample.Status.TRUNCATED:
                break
        else:
            status = Sample.Status.TRUNCATED

        if last_generation is not None:
            confirm_agentic_generation_final(
                metadata,
                last_generation,
                p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
            )
    except Exception as exc:
        status = Sample.Status.FAILED
        stop_reason = f"environment_error:{type(exc).__name__}"
        metadata["environment_error"] = str(exc)
    finally:
        await client.close()

    sample.status = status
    sample.tokens = prompt_tokens + response_tokens
    sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    sample.reward = reward
    sample.tool_time = tool_seconds
    sample.tool_call_count = command_count
    sample.code_call_count = 0
    sample.search_call_count = 0
    metadata.update(
        {
            "num_turns": (last_generation + 1) if last_generation is not None else 0,
            "stop_reason": stop_reason,
            "shell_call_count": command_count,
            "tool_time": tool_seconds,
            "sample_time": time.monotonic() - started_at,
        }
    )
    return sample
