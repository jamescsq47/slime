"""BrowseComp multi-turn caller with request-generation KV metadata.

This is the serving/profile entry point for ``examples/pd``.  It deliberately
keeps only the closed-loop inference behavior needed by the PD experiments;
the removed training-oriented ``examples/mixed`` implementation is not
restored or modified.
"""

from __future__ import annotations

import os
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
from .env import BrowseCompEnv, SearchBackendError, extract_fn_call
from pd_metrics import sglang_meta_attrs
from slime.dashboard.api import span as dashboard_span
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


BUDGET_MARGIN = 512
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _render(tokenizer, messages: list[dict[str, Any]], *, generation: bool) -> str:
    kwargs: dict[str, Any] = {}
    value = os.getenv("BROWSECOMP_ENABLE_THINKING")
    if value is not None:
        kwargs["enable_thinking"] = value.lower() in {"1", "true", "yes", "y"}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=generation,
        **kwargs,
    )


def _initial_tokens(tokenizer, prompt: Any) -> list[int]:
    if isinstance(prompt, list):
        prompt = _render(tokenizer, prompt, generation=True)
    if not isinstance(prompt, str):
        raise TypeError(f"BrowseComp prompt must be str or message list, got {type(prompt)}")
    return list(tokenizer.encode(prompt, add_special_tokens=False))


def _observation_tokens(tokenizer, observation: str) -> list[int]:
    base = _render(tokenizer, DUMMY_MESSAGES, generation=False)
    full = _render(
        tokenizer,
        DUMMY_MESSAGES + [{"role": "user", "content": observation}],
        generation=True,
    )
    base_ids = list(tokenizer.encode(base, add_special_tokens=False))
    full_ids = list(tokenizer.encode(full, add_special_tokens=False))
    if full_ids[: len(base_ids)] == base_ids:
        result = full_ids[len(base_ids) :]
    else:
        result = list(tokenizer.encode(observation, add_special_tokens=False))
    if result and tokenizer.bos_token_id is not None and result[0] == tokenizer.bos_token_id:
        result = result[1:]
    return result


async def _generate_turn(
    *,
    url: str,
    input_ids: list[int],
    sampling_params: dict[str, Any],
    tokenizer: Any,
    trajectory_metadata: dict[str, Any],
    generation: int,
) -> tuple[str, list[int], str, dict[str, Any]]:
    params = dict(sampling_params)
    request_id = None
    if lifecycle_enabled():
        params, request_id = add_agentic_kv_metadata(
            params,
            trajectory_metadata=trajectory_metadata,
            generation=generation,
            tokenizer=tokenizer,
            tool_type="browsecomp",
            tool_suffix_markers=("</function>", "</tool_call>"),
            terminal_markers=("<function=finish>", "<answer>"),
        )
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": params,
        "return_logprob": False,
    }
    if request_id is not None:
        payload["extra_key"] = build_agentic_extra_key(request_id, params)
    output = await post(url, payload)
    meta = output["meta_info"]
    tokens = [int(item) for item in output.get("output_ids") or []]
    return (
        output.get("text", ""),
        tokens,
        meta["finish_reason"]["type"],
        meta,
    )


async def generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """Run one real BrowseComp trajectory through the strict PD router."""

    del evaluation
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    metadata = sample.metadata
    question = metadata.get("question")
    label = metadata.get("answer") or sample.label
    if not question or not label:
        raise ValueError("BrowseComp sample requires metadata.question and answer")

    state = GenerateState(args)
    prompt_tokens = _initial_tokens(state.tokenizer, sample.prompt)
    response_tokens: list[int] = []
    dataset_id = str(metadata.get("dataset_id") or metadata.get("task_type") or "qa")
    options = getattr(args, "workload_dataset_options", {}).get(dataset_id, {})
    max_turns = int(options.get("max_turns", os.getenv("BROWSECOMP_MAX_TURNS", "100")))
    per_turn_cap = min(
        int(
            options.get(
                "max_tokens_per_turn",
                os.getenv("BROWSECOMP_TURN_MAX_NEW_TOKENS", "2048"),
            )
        ),
        int(sampling_params.get("max_new_tokens") or 2048),
    )
    max_seq_len = (
        getattr(args, "max_seq_len", None)
        or getattr(args, "sglang_context_length", None)
        or int(os.getenv("BROWSECOMP_MAX_SEQ_LEN", "40960"))
    )
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    env = BrowseCompEnv(
        question=question,
        label_answer=label,
        must_search=os.getenv("BROWSECOMP_MUST_SEARCH", "1") == "1",
        base_url=options.get("search_url"),
    )
    started_at = time.monotonic()
    tool_seconds = 0.0
    tool_tokens = 0
    stop_reason = "max_turns"
    status = Sample.Status.COMPLETED
    completed_turns = 0
    terminal_repair_events: list[dict[str, Any]] = []
    metadata["terminal_repair_events"] = terminal_repair_events
    pending_terminal_repair: dict[str, Any] | None = None

    try:
        for turn in range(max_turns):
            all_tokens = prompt_tokens + response_tokens
            if len(all_tokens) + per_turn_cap + BUDGET_MARGIN >= max_seq_len:
                stop_reason = "budget"
                status = Sample.Status.TRUNCATED
                break
            params = dict(sampling_params)
            params["max_new_tokens"] = per_turn_cap
            repair_event = pending_terminal_repair
            pending_terminal_repair = None
            if repair_event is not None:
                repair_event["next_generation"] = turn
                repair_event["next_prompt_tokens"] = len(all_tokens)
                repair_event["repair_attempted"] = True
            with dashboard_span(
                args,
                sample,
                "generation_turn",
                attrs={
                    "task_type": "qa",
                    "turn": turn + 1,
                    "max_new_tokens": per_turn_cap,
                    "route_mode": "strict_pd",
                    "terminal_repair": repair_event is not None,
                },
            ) as generation_span:
                text, new_tokens, finish_type, meta = await _generate_turn(
                    url=url,
                    input_ids=all_tokens,
                    sampling_params=params,
                    tokenizer=state.tokenizer,
                    trajectory_metadata=metadata,
                    generation=turn,
                )
                generation_span.update(sglang_meta_attrs(meta))
            if repair_event is not None:
                actual_cached = int(meta.get("cached_tokens") or 0)
                actual_prompt = int(meta.get("prompt_tokens") or len(all_tokens))
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
            response_tokens.extend(new_tokens)
            completed_turns = turn + 1
            # SGLang can return an empty generation without advancing the
            # Decode finished-request hook (for example, an immediate stop on
            # an already-complete assistant boundary).  There is no generated
            # KV snapshot for such a turn, so a repair turn would necessarily
            # reference a parent that cannot exist.  Treat it as a terminal
            # serving outcome instead of fabricating continuation state.
            if not generation_has_visible_content(new_tokens, state.tokenizer):
                confirm_agentic_generation_final(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                stop_reason = "empty_generation"
                status = Sample.Status.TRUNCATED
                break
            if finish_type == "abort":
                stop_reason = "abort"
                status = Sample.Status.ABORTED
                break
            if finish_type == "length":
                stop_reason = "length"
                status = Sample.Status.TRUNCATED
                break

            parsed_calls = extract_fn_call(text) or []
            explicit_terminal = (
                any(call.get("function") == "finish" for call in parsed_calls)
                or "<function=finish>" in text
                or "<answer>" in text
            )
            valid_tool_call = any(
                (
                    call.get("function") == "search"
                    and bool(str(call.get("arguments", {}).get("query", "")).strip())
                )
                or (
                    call.get("function") == "open_page"
                    and bool(
                        call.get("arguments", {}).get("url")
                        or call.get("arguments", {}).get("docid")
                    )
                )
                for call in parsed_calls
            )
            if valid_tool_call:
                # ACK before search/open_page performs network I/O.  Only
                # parser-accepted executable calls may use slow KV storage.
                confirm_agentic_generation_tool(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
            elif explicit_terminal:
                # A malformed finish/answer is terminal only for this KV
                # generation.  Baseline serving gives the model the normal
                # parser repair observation and starts another generation.
                # Marking this parent final makes P safely recompute that
                # repair turn instead of attempting to reuse released KV.
                confirm_agentic_generation_final(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
            tool_started = time.monotonic()
            result = await env.run_action(text)
            tool_seconds += time.monotonic() - tool_started
            if result.get("action") == "finish":
                confirm_agentic_generation_final(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                stop_reason = "finish"
                break
            if not valid_tool_call and not explicit_terminal:
                # Malformed/no-tool output can receive an immediate repair
                # observation.  It is therefore a continuation, not final;
                # reuse the existing continuation ACK so D may serve it via
                # Direct but still cannot spill it without application ACK.
                confirm_agentic_generation_tool(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
            observation = _observation_tokens(state.tokenizer, result["observation"])
            remaining = max_seq_len - len(prompt_tokens) - len(response_tokens)
            if remaining <= 0:
                stop_reason = "budget"
                status = Sample.Status.TRUNCATED
                break
            if len(observation) > remaining:
                observation = observation[:remaining]
                status = Sample.Status.TRUNCATED
                stop_reason = "budget"
            response_tokens.extend(observation)
            tool_tokens += len(observation)
            if explicit_terminal:
                parent_kv_end = len(all_tokens) + len(new_tokens) - 1
                event = {
                    "generation": turn,
                    "repair_attempted": False,
                    "parent_prompt_tokens": len(all_tokens),
                    "parent_completion_tokens": len(new_tokens),
                    "repair_observation_tokens": len(observation),
                    "page_aligned_parent_kv_tokens": max(
                        0, (parent_kv_end // 64) * 64
                    ),
                }
                terminal_repair_events.append(event)
                pending_terminal_repair = event
            if status is Sample.Status.TRUNCATED:
                break
    except SearchBackendError:
        status = Sample.Status.ABORTED
        stop_reason = "search_backend_error"
    finally:
        await env.close()

    sample.status = status
    sample.tokens = prompt_tokens + response_tokens
    sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    sample.sample_time = time.monotonic() - started_at
    sample.tool_time = tool_seconds
    sample.tool_token_count = tool_tokens
    sample.tool_call_count = int(env.stats.get("search", 0)) + int(
        env.stats.get("open_page", 0)
    )
    if completed_turns > 0:
        # The application, unlike D, knows there will be no next model call
        # for finish, length, budget, or abort exits.  Confirm the last
        # generation so D can release provisional KV without spilling it.
        confirm_agentic_generation_final(
            metadata,
            completed_turns - 1,
            p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
        )
    predicted, explanation, confidence = env.predicted_answer or (None, None, None)
    metadata.update(
        {
            "predicted_answer": predicted,
            "explanation": explanation,
            "confidence": confidence,
            "num_turns": completed_turns,
            "stop_reason": stop_reason,
            "tool_stats": dict(env.stats),
            "tool_token_count": tool_tokens,
            "tool_time": tool_seconds,
        }
    )
    return sample
