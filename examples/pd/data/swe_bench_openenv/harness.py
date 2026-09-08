"""OpenEnv-style SWE-bench episode loop adapted to local SGLang serving.

The control contract follows the direct SWE-bench evaluator in Miles PR #51:

``reset -> {policy -> one shell action} -> capture patch -> hidden verifier``.

Only transports are local adapters. Model calls go through the experiment's
SGLang router and environment actions use the existing instrumented Docker or
Daytona sandbox. Hidden verifier material is unavailable until the policy has
irreversibly stopped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any

from agentic_kv_request import (
    add_agentic_kv_metadata,
    build_agentic_extra_key,
    confirm_agentic_generation_final,
    confirm_agentic_generation_tool,
    lifecycle_enabled,
)
from pd_metrics import sglang_meta_attrs
from slime.dashboard.api import span as dashboard_span
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from data.swe_bench.harness import _create_task, _verifier_semaphore
from data.swe_bench.verifier import (
    capture_repository_patch,
    patch_metadata,
    prepare_repository_baseline,
    run_inline_verifier,
)


_BASH_FENCE = re.compile(
    r"^[ \t]*```(?:bash|sh|shell)[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*(?:\r?\n|$)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_GENERIC_FENCE = re.compile(
    r"^[ \t]*```[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*(?:\r?\n|$)",
    re.DOTALL | re.MULTILINE,
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TASK_COMPLETE_LINE = re.compile(r"(?im)^\s*TASK_COMPLETE\s*$")
_TASK_COMPLETE = "TASK_COMPLETE"

_SHELL_TOOL = {
    "type": "function",
    "function": {
        "description": "Execute exactly one shell command in the /testbed repository.",
        # Keep the canonical field order emitted by pd_baseline's Pydantic
        # Tool model.  Qwen's template preserves JSON insertion order, so
        # changing this order changes the token sequence even at equal length.
        "name": "shell",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        # Match the canonical Tool schema used by the untouched pd_baseline
        # Chat server.  Without this explicit default, its Pydantic adapter
        # inserts ``strict=false`` before applying the Qwen template, so the
        # baseline prompt is five tokens longer than the harness rendering.
        "strict": False,
    },
}

_SYSTEM_PROMPT = """You are an autonomous software-engineering agent working on a
SWE-bench issue in a real repository at /testbed. Inspect the code, reproduce the
bug when useful, implement the smallest correct fix, and run focused tests. On
each turn respond with EXACTLY ONE shell command inside a single ```bash code
block and nothing else. Commands always start in /testbed, but shell state such
as `cd` does not persist between turns. Do not merely describe a patch: edit the
files in the repository. When the fix is complete, reply TASK_COMPLETE with no
code block. The benchmark verifier is unavailable during your work and must not
be searched for or modified."""

_TOOL_SYSTEM_PROMPT = """You are an autonomous software-engineering agent working
on a SWE-bench issue in a real repository at /testbed. Inspect the code,
reproduce the bug when useful, implement the smallest correct fix, and run
focused tests. On each turn call the shell tool exactly once with one command.
Commands always start in /testbed, but shell state such as `cd` does not persist
between turns. Do not merely describe a patch: edit the files in the repository.
When the fix is complete, reply TASK_COMPLETE without calling a tool. The
benchmark verifier is unavailable during your work and must not be searched for
or modified."""

_CONDA_PREFIX = (
    "if test -f /opt/miniconda3/bin/activate; then "
    "source /opt/miniconda3/bin/activate >/dev/null 2>&1; "
    "conda activate testbed >/dev/null 2>&1; fi; "
)


def extract_command(reply: str) -> str:
    """Return PR #51's accepted control token or one fenced shell command."""

    # Qwen3.5 may expose the closing </think> token while its opening marker is
    # consumed by the chat template. Treat everything before that boundary as
    # reasoning, matching the OpenAI reasoning-parser view used by Miles.
    normalized = reply.replace("<|im_end|>", "")
    visible = (
        normalized.rsplit("</think>", 1)[-1]
        if "</think>" in normalized
        else _THINK.sub("", normalized)
    )
    stripped = visible.strip()
    if stripped.upper().startswith(_TASK_COMPLETE):
        return _TASK_COMPLETE
    shell_matches = _BASH_FENCE.findall(visible)
    if len(shell_matches) == 1:
        return shell_matches[0].strip()
    if shell_matches:
        return ""
    generic_matches = _GENERIC_FENCE.findall(visible)
    if len(generic_matches) == 1:
        return generic_matches[0].strip()
    return ""


def extract_tool_command(tool_calls: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Accept exactly one well-formed OpenAI ``shell`` function call."""

    if not tool_calls:
        return "", "missing_tool_call"
    if len(tool_calls) != 1:
        return "", "multiple_tool_calls"
    function = tool_calls[0].get("function") or {}
    function_name = str(function.get("name") or "")
    # Qwen3.5 occasionally repeats the XML attribute label in the parsed
    # function name. The call is otherwise fully structured and the harness
    # exposes only one executable tool, so these two observed spellings are
    # unambiguous compatibility aliases rather than free-form command parsing.
    if function_name not in {"shell", "function=shell", "shell=shell"}:
        return "", "unexpected_tool_name"
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return "", "invalid_tool_arguments_json"
    if not isinstance(arguments, dict):
        return "", "invalid_tool_arguments"
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return "", "missing_shell_command"
    return command.strip(), None


def structured_terminal_reason(
    reply: str,
    tool_calls: list[dict[str, Any]],
    reasoning_content: str = "",
) -> str | None:
    """Classify a tool-protocol response that does not request another action.

    In OpenAI's tool-calling protocol, a non-empty assistant response without a
    tool call is a valid final answer. ``TASK_COMPLETE`` remains a stronger,
    explicit completion marker, but it may follow a short human-readable
    summary. Only an empty response without a tool call is a protocol failure.
    """

    if tool_calls:
        return None
    visible = reply.replace("<|im_end|>", "").strip()
    reasoning = reasoning_content.replace("<|im_end|>", "").strip()
    if _TASK_COMPLETE_LINE.search(visible) or _TASK_COMPLETE_LINE.search(reasoning):
        return "task_complete"
    if visible:
        return "final_answer"
    return "no_command"


def bounded_observation(output: str, cap: int) -> tuple[str, bool]:
    if not output:
        return "(no output)", False
    if len(output) <= cap:
        return output, False
    head = cap // 2
    tail = cap - head
    omitted = len(output) - cap
    return (
        output[:head]
        + f"\n\n... [{omitted} characters omitted from model context] ...\n\n"
        + output[-tail:],
        True,
    )


def compact_messages(
    messages: list[dict[str, Any]], history_messages: int
) -> list[dict[str, Any]]:
    """Keep the public task, durable repository state, and recent transcript."""

    if len(messages) <= history_messages + 3:
        return messages
    return messages[:2] + [
        {
            "role": "user",
            "content": (
                "Earlier shell interaction was compacted to stay within the model "
                "context. Continue from the current repository state."
            ),
        }
    ] + messages[-history_messages:]


def _next_identical_outcome(
    previous: tuple[str, int, str] | None,
    count: int,
    command: str,
    exit_code: int,
    output: str,
) -> tuple[tuple[str, int, str], int]:
    fingerprint = (
        command,
        exit_code,
        hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest(),
    )
    return fingerprint, count + 1 if fingerprint == previous else 1


def _render_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    # OpenAI history keeps function.arguments as a JSON string. Hugging Face
    # chat templates expect a mapping, so normalize only this local rendering
    # copy; the request sent through the Rust router remains protocol-valid.
    template_messages = []
    for message in messages:
        rendered_message = dict(message)
        if isinstance(message.get("tool_calls"), list):
            rendered_calls = []
            for call in message["tool_calls"]:
                rendered_call = dict(call)
                function = dict(rendered_call.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                rendered_call["function"] = function
                rendered_calls.append(rendered_call)
            rendered_message["tool_calls"] = rendered_calls
        template_messages.append(rendered_message)
    rendered = tokenizer.apply_chat_template(
        template_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tools=tools,
    )
    return list(tokenizer.encode(rendered, add_special_tokens=False))


async def _model_turn(
    *,
    args: Any,
    sample: Sample,
    tokenizer: Any,
    metadata: dict[str, Any],
    messages: list[dict[str, Any]],
    sampling_params: dict[str, Any],
    options: dict[str, Any],
    turn: int,
) -> tuple[str, list[int], dict[str, Any], list[dict[str, Any]] | None]:
    max_context = int(getattr(args, "max_seq_len", None) or 81920)
    max_tokens = int(options.get("max_tokens_per_turn", 8192))
    history_messages = int(options.get("history_messages", 12))
    enable_thinking = bool(options.get("enable_thinking", True))
    action_protocol = str(options.get("action_protocol", "fenced_shell")).strip().lower()
    if action_protocol not in {"fenced_shell", "openai_tools"}:
        raise ValueError(f"unsupported OpenEnv SWE-bench action_protocol {action_protocol!r}")
    tools = [_SHELL_TOOL] if action_protocol == "openai_tools" else None
    request_messages = messages
    compacted_messages: list[dict[str, Any]] | None = None

    input_ids = _render_prompt(
        tokenizer,
        request_messages,
        enable_thinking=enable_thinking,
        tools=tools,
    )
    remaining = max_context - len(input_ids) - 1
    if remaining < 16:
        candidate = compact_messages(messages, history_messages)
        if candidate is messages:
            raise RuntimeError(
                f"OpenEnv SWE-bench prompt reached the {max_context}-token context limit"
            )
        request_messages = candidate
        compacted_messages = candidate
        input_ids = _render_prompt(
            tokenizer,
            request_messages,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        remaining = max_context - len(input_ids) - 1
    if remaining < 16:
        raise RuntimeError(
            f"OpenEnv SWE-bench compacted prompt exceeds {max_context} tokens"
        )

    params = dict(sampling_params)
    params["max_new_tokens"] = min(max_tokens, remaining)
    model_api = str(options.get("model_api", "generate")).strip().lower()
    if model_api not in {"generate", "chat_completions"}:
        raise ValueError(f"unsupported OpenEnv SWE-bench model_api {model_api!r}")
    request_id = None
    if lifecycle_enabled():
        params, request_id = add_agentic_kv_metadata(
            params,
            trajectory_metadata=metadata,
            generation=turn,
            tokenizer=tokenizer,
            tool_type="shell",
            tool_suffix_markers=("```",),
            terminal_markers=(_TASK_COMPLETE,),
        )
    if model_api == "chat_completions":
        # The official SGLang reasoning parser is applied only by the OpenAI
        # chat entrypoint. It returns private chain-of-thought separately from
        # the actionable assistant content; no harness-authored recovery or
        # format-correction prompt is involved.
        payload = {
            "model": str(
                options.get("model")
                or getattr(args, "hf_checkpoint", "")
                or getattr(args, "model", "default")
            ),
            "messages": request_messages,
            # SGLang's Chat endpoint accepts pre-tokenized input while still
            # using ``messages``/``tools`` for stop-token and structured-tool
            # parsing.  Reuse the exact prompt rendered above so Router
            # admission, the P Direct workset, and the model server all agree
            # on one token sequence.  In particular, Pydantic-normalized tool
            # schemas can otherwise make server-side Chat tokenization a few
            # tokens longer than the harness rendering.
            "input_ids": input_ids,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),
            "min_p": params.get("min_p"),
            "max_completion_tokens": params["max_new_tokens"],
            "stop": params.get("stop"),
            "stop_token_ids": params.get("stop_token_ids"),
            "skip_special_tokens": params.get("skip_special_tokens", True),
            "no_stop_trim": params.get("no_stop_trim", True),
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "separate_reasoning": True,
            "stream": False,
        }
        if tools is not None:
            payload.update(
                {
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            )
        if request_id is not None:
            # Chat Completions accepts the same SGLang extensions as
            # /generate.  Carry both fields: custom_params reaches the
            # scheduler directly, while the generation-scoped extra_key is
            # also understood by PD routers that only preserve that field.
            # Keep the count for compatibility with routers that do not yet
            # preserve Chat ``input_ids``.  The current Router gives the real
            # token list precedence over this hint.
            params["custom_params"]["agentic_prompt_token_count"] = len(input_ids)
            payload["custom_params"] = params["custom_params"]
            payload["extra_key"] = build_agentic_extra_key(request_id, params)
        payload = {key: value for key, value in payload.items() if value is not None}
        url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
            "/v1/chat/completions"
        )
    else:
        payload = {
            "input_ids": input_ids,
            "sampling_params": params,
            "return_logprob": False,
        }
        if request_id is not None:
            payload["extra_key"] = build_agentic_extra_key(request_id, params)
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    with dashboard_span(
        args,
        sample,
        "generation_turn",
        attrs={
            "task_type": metadata.get("task_type", "swe_bench"),
            "turn": turn + 1,
            "max_new_tokens": params["max_new_tokens"],
            "agent_harness": "openenv-swebench-pr51",
        },
    ) as span:
        started = time.monotonic()
        output = await post(url, payload)
        generation_seconds = time.monotonic() - started
        if model_api == "generate":
            span.update(sglang_meta_attrs(output.get("meta_info", {})))

    reasoning_content = ""
    tool_calls: list[dict[str, Any]] = []
    if model_api == "chat_completions":
        choices = output.get("choices") or []
        if not choices:
            raise RuntimeError("SGLang Chat Completions returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        reply = str(message.get("content") or "").strip()
        reasoning_content = str(message.get("reasoning_content") or "")
        tool_calls = list(message.get("tool_calls") or [])
        output_ids = list(tokenizer.encode(reply, add_special_tokens=False))
        usage = output.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        prompt_tokens = int(usage.get("prompt_tokens", len(input_ids)) or 0)
        completion_tokens = int(usage.get("completion_tokens", len(output_ids)) or 0)
        reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
        cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
        finish_type = str(choice.get("finish_reason") or "stop")
    else:
        meta = output.get("meta_info", {})
        output_ids = list(output.get("output_ids") or [])
        reply = tokenizer.decode(output_ids, skip_special_tokens=False).strip()
        prompt_tokens = int(meta.get("prompt_tokens", len(input_ids)) or 0)
        completion_tokens = len(output_ids)
        reasoning_tokens = 0
        cached_tokens = int(meta.get("cached_tokens", 0) or 0)
        finish_type = str((meta.get("finish_reason") or {}).get("type") or "stop")
    metric = {
        "turn": turn + 1,
        "input_tokens": len(input_ids),
        "output_tokens": completion_tokens,
        "visible_output_tokens": len(output_ids),
        "reasoning_tokens": reasoning_tokens,
        "generation_seconds": generation_seconds,
        "finish_type": finish_type,
        "cached_tokens": cached_tokens,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": params["max_new_tokens"],
        "context_compacted": compacted_messages is not None,
        "model_api": model_api,
        "action_protocol": action_protocol,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "model_reply": reply,
    }
    return reply, output_ids, metric, compacted_messages


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    metadata = dict(sample.metadata or {})
    sample.metadata = metadata
    dataset_id = str(
        metadata.get("dataset_id")
        or metadata.get("task_type")
        or "swe_bench_openenv"
    )
    options = dict(getattr(args, "workload_dataset_options", {}).get(dataset_id, {}))
    action_protocol = str(options.get("action_protocol", "fenced_shell")).strip().lower()
    if action_protocol == "openai_tools" and str(
        options.get("model_api", "generate")
    ).strip().lower() != "chat_completions":
        raise ValueError("openai_tools requires model_api=chat_completions")
    state = GenerateState(args)
    task = _create_task(metadata, options)
    verifier_mode = str(options.get("verifier_mode", "inline")).strip().lower()
    if verifier_mode not in {"inline", "capture", "disabled"}:
        raise ValueError(f"unsupported SWE-bench verifier_mode {verifier_mode!r}")

    default_system_prompt = (
        _TOOL_SYSTEM_PROMPT if action_protocol == "openai_tools" else _SYSTEM_PROMPT
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": str(options.get("system_prompt", default_system_prompt)),
        },
        {"role": "user", "content": str(metadata["problem_statement"])},
    ]
    turn_events: list[dict[str, Any]] = []
    output_token_ids: list[int] = []
    raw_replies: list[str] = []
    command_count = 0
    completion_token_count = 0
    model_seconds = 0.0
    tool_seconds = 0.0
    preserved_commit = ""
    final_patch = ""
    reward: int | None = None
    status = Sample.Status.COMPLETED
    stop_reason = "max_turns"
    last_generation: int | None = None
    last_outcome: tuple[str, int, str] | None = None
    identical_outcomes = 0
    last_failing_command: str | None = None
    repeated_failing_commands = 0
    started_at = time.monotonic()

    try:
        await task.start()
        baseline = await prepare_repository_baseline(
            task, str(metadata.get("base_commit", ""))
        )
        preserved_commit = baseline.image_commit
        metadata["repository_baseline"] = {
            "official_base_commit": baseline.official_base_commit,
            "official_tree": baseline.official_tree,
            "image_commit": baseline.image_commit,
            "image_tree": baseline.image_tree,
            "kind": baseline.kind,
            "image_commits_ahead": baseline.image_commits_ahead,
            "fingerprint": baseline.fingerprint,
            "source_image_commit": baseline.source_image_commit,
            "initial_worktree_status": baseline.initial_worktree_status,
        }

        max_turns = int(options.get("max_turns", 64))
        observation_chars = int(options.get("max_observation_chars", 12000))
        identical_limit = int(options.get("max_identical_command_outcomes", 4))
        repeated_failure_limit = int(options.get("max_repeated_failing_commands", 0))
        command_timeout = float(options.get("command_timeout_seconds", 600))

        for turn in range(max_turns):
            last_generation = turn
            reply, output_ids, metric, request_messages = await _model_turn(
                args=args,
                sample=sample,
                tokenizer=state.tokenizer,
                metadata=metadata,
                messages=messages,
                sampling_params=sampling_params,
                options=options,
                turn=turn,
            )
            output_token_ids.extend(output_ids)
            completion_token_count += int(metric["output_tokens"])
            raw_replies.append(reply)
            model_seconds += float(metric["generation_seconds"])
            tool_calls = list(metric.get("tool_calls") or [])
            structured_error: str | None = None
            if action_protocol == "openai_tools":
                if tool_calls:
                    command, structured_error = extract_tool_command(tool_calls)
                else:
                    # A response without a tool call is the standard terminal
                    # form when ``tool_choice=auto``; it is not a malformed
                    # action by itself.
                    command = ""
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": reply or None,
                    "reasoning_content": metric.get("reasoning_content") or None,
                }
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
            else:
                messages.append({"role": "assistant", "content": reply})
                command = extract_command(reply)
            event: dict[str, Any] = {
                **metric,
                "assistant": reply,
                "command": command or None,
                "structured_action_error": structured_error,
                "structured_action_repaired": bool(
                    tool_calls
                    and not structured_error
                    and str((tool_calls[0].get("function") or {}).get("name") or "")
                    != "shell"
                ),
                "request_messages": request_messages,
            }
            turn_events.append(event)

            if action_protocol == "openai_tools":
                terminal_reason = structured_terminal_reason(
                    reply,
                    tool_calls,
                    str(metric.get("reasoning_content") or ""),
                )
            else:
                terminal_reason = (
                    "task_complete" if command == _TASK_COMPLETE else None
                )
            if terminal_reason is not None:
                stop_reason = terminal_reason
                confirm_agentic_generation_final(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                break
            if not command:
                stop_reason = "no_command"
                confirm_agentic_generation_final(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                break

            confirm_agentic_generation_tool(
                metadata,
                turn,
                p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
            )
            command_started = time.monotonic()
            exit_code, output = await task.execute(
                _CONDA_PREFIX + command,
                timeout=command_timeout,
                phase="agent_tool",
            )
            command_seconds = time.monotonic() - command_started
            tool_seconds += command_seconds
            command_count += 1
            observation, truncated = bounded_observation(output, observation_chars)
            model_observation = (
                f"<shell_result exit_code={exit_code}>\n"
                f"{observation}\n</shell_result>"
            )
            if action_protocol == "openai_tools":
                tool_call_id = str(tool_calls[0].get("id") or "functions.shell:0")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "shell",
                        "content": model_observation,
                    }
                )
            else:
                messages.append({"role": "user", "content": model_observation})
            event.update(
                {
                    "command_seconds": command_seconds,
                    "exit_code": exit_code,
                    "observation": output,
                    "observation_chars": len(output),
                    "observation_tokens": len(
                        state.tokenizer.encode(
                            model_observation, add_special_tokens=False
                        )
                    ),
                    "observation_sent_to_model": model_observation,
                    "observation_truncated_for_model": truncated,
                }
            )

            if exit_code != 0:
                if command == last_failing_command:
                    repeated_failing_commands += 1
                else:
                    last_failing_command = command
                    repeated_failing_commands = 1
            else:
                last_failing_command = None
                repeated_failing_commands = 0
            last_outcome, identical_outcomes = _next_identical_outcome(
                last_outcome, identical_outcomes, command, exit_code, output
            )
            event["repeated_failing_commands"] = repeated_failing_commands
            event["identical_command_outcomes"] = identical_outcomes

            if exit_code == 124:
                stop_reason = "command_timeout"
                break
            if repeated_failure_limit and repeated_failing_commands >= repeated_failure_limit:
                stop_reason = "repeated_failing_command"
                break
            if identical_limit and identical_outcomes >= identical_limit:
                stop_reason = "repeated_command_outcome"
                break
        else:
            # Reaching the configured turn budget is a normal OpenEnv episode
            # terminal state. The resulting repository is still canonically
            # graded, unlike mini-SWE-agent's submission-gated contract.
            if last_generation is not None:
                confirm_agentic_generation_final(
                    metadata,
                    last_generation,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )

        # PR #51 grades the durable repository state after the agent stops; it
        # does not require a harness-specific submission marker.
        final_patch = await capture_repository_patch(task, preserved_commit)
        metadata.update(patch_metadata(final_patch))
        metadata["model_patch"] = final_patch

        if verifier_mode == "inline":
            verifier_limit = int(options.get("verifier_max_concurrent", 16))
            queued = time.monotonic()
            async with _verifier_semaphore(args, verifier_limit):
                metadata["verifier_queue_seconds"] = time.monotonic() - queued
                verifier = await run_inline_verifier(
                    task,
                    metadata,
                    final_patch,
                    timeout_seconds=float(options.get("verifier_timeout_seconds", 2400)),
                    output_tail_chars=int(
                        options.get("verifier_output_tail_chars", 12000)
                    ),
                )
            metadata["swe_bench_verifier"] = verifier.to_metadata()
            reward = verifier.reward
            if verifier.status == "infrastructure_error":
                status = Sample.Status.FAILED
                stop_reason = "verifier_infrastructure_error"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        status = Sample.Status.FAILED
        stop_reason = f"environment_error:{type(exc).__name__}"
        metadata["environment_error"] = str(exc)
        if last_generation is not None:
            confirm_agentic_generation_final(
                metadata,
                last_generation,
                p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
            )
    finally:
        await task.close()

    sample.status = status
    sample.tokens = output_token_ids
    sample.response = "\n\n".join(raw_replies)
    sample.response_length = completion_token_count
    sample.reward = reward
    sample.tool_time = tool_seconds
    sample.tool_call_count = command_count
    sample.code_call_count = 0
    sample.search_call_count = 0
    metadata.update(
        {
            "num_turns": len(turn_events),
            "stop_reason": stop_reason,
            "shell_call_count": command_count,
            "tool_time": tool_seconds,
            "model_time": model_seconds,
            "sample_time": time.monotonic() - started_at,
            "turn_metrics": turn_events,
            "container_image": task.image,
            "sandbox_metrics": task.metrics,
            "sandbox_backend": str(options.get("sandbox_backend", "docker")).lower(),
            "verifier_mode": verifier_mode,
            "agent_harness": "openenv-swebench-pr51",
            "model_api": str(options.get("model_api", "generate")).strip().lower(),
            "action_protocol": action_protocol,
            "openenv_episode_schema_version": 1,
            "openenv_trajectory": {
                "system_prompt": messages[0]["content"],
                "instruction": messages[1]["content"],
                "messages": messages,
                "turn_events": turn_events,
                "terminal_reason": stop_reason,
                "patch": final_patch,
                "reward": reward,
            },
        }
    )
    progress_path = os.environ.get("PD_SWE_PROGRESS_FILE", "").strip()
    if progress_path:
        record = json.dumps(
            {
                "instance_id": str(metadata.get("instance_id", "")),
                "status": str(sample.status),
                "reward": sample.reward,
                "completed_at": time.time(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
        fd = os.open(progress_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, record.encode("utf-8"))
        finally:
            os.close(fd)
    return sample
