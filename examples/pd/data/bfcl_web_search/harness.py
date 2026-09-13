"""Native Qwen tool calling with exact model token history; inference only."""

import asyncio
import json
import logging
import re
import time

from agentic_kv_request import lifecycle_enabled
from pd_metrics import sglang_meta_attrs
from slime.dashboard.api import span as dashboard_span
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .tools import WebBackendError, WebTools, validate_call

LOG = logging.getLogger(__name__)


def parse_calls(text):
    # Qwen reasoning may discuss tool syntax; only parse the answer portion.
    text = text.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
    if "<tool_call>" in text and not blocks:
        raise ValueError("Unclosed tool_call; return a complete JSON tool call")
    if len(blocks) > 8:
        raise ValueError("At most 8 tool calls per model turn")
    calls = [json.loads(block) for block in blocks]
    for call in calls:
        validate_call(call)
    return calls


def initial_tokens(tokenizer, sample):
    rendered = tokenizer.apply_chat_template(
        sample.prompt, tools=sample.metadata["bfcl_tools"],
        tokenize=False, add_generation_prompt=True,
    )
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def observation_tokens(tokenizer, observations):
    # Append template boundaries; never decode/re-encode model-produced IDs.
    # No dummy assistant: Qwen's template changes the last assistant's thinking
    # block when a later tool message is appended, breaking prefix equality.
    base = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "x"}]
    prefix_text = tokenizer.apply_chat_template(base, tokenize=False, add_generation_prompt=False)
    full_text = tokenizer.apply_chat_template(
        base + [{"role": "tool", "content": json.dumps(x, ensure_ascii=False)} for x in observations],
        tokenize=False, add_generation_prompt=True,
    )
    prefix = list(tokenizer.encode(prefix_text, add_special_tokens=False))
    full = list(tokenizer.encode(full_text, add_special_tokens=False))
    if full[:len(prefix)] != prefix:
        raise ValueError("Tokenizer tool template is not append-only")
    return full[len(prefix):]


async def generate(args, sample, sampling_params):
    if lifecycle_enabled():
        raise RuntimeError("BFCL harness is colocated-only until PD lifecycle integration is audited")
    metadata = sample.metadata
    options = getattr(args, "workload_dataset_options", {}).get(metadata.get("dataset_id", "bfcl"), {})
    tokenizer = GenerateState(args).tokenizer
    initial = initial_tokens(tokenizer, sample)
    response = []
    tools = WebTools(
        backend=options.get("search_backend", "auto"),
        snippets=metadata["bfcl_variant"] == "base",
        timeout=int(options.get("tool_timeout_seconds", 20)),
        max_chars=int(options.get("page_max_chars", 40000)),
    )
    started = time.monotonic()
    completed_turns = tool_tokens = tool_errors = 0
    max_tool_errors = max(1, int(options.get("max_tool_errors", 3)))
    stop_reason, status = "max_turns", Sample.Status.TRUNCATED
    params = dict(sampling_params)
    params["max_new_tokens"] = min(int(params.get("max_new_tokens") or 8192), int(options.get("max_tokens_per_turn", 8192)))
    total_budget = int(options.get("max_response_tokens", 32768))
    context_limit = int(getattr(args, "max_seq_len", None) or 40960)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    try:
        for turn in range(int(options.get("max_turns", 20))):
            remaining = min(total_budget - len(response), context_limit - len(initial) - len(response) - 512)
            if remaining <= 0:
                stop_reason = "budget"
                break
            turn_params = {**params, "max_new_tokens": min(params["max_new_tokens"], remaining)}
            with dashboard_span(args, sample, "generation_turn", attrs={
                "task_type": "bfcl", "turn": turn + 1, "route_mode": "colocated",
                "max_new_tokens": turn_params["max_new_tokens"],
            }) as span:
                output = await post(url, {"input_ids": initial + response,
                                          "sampling_params": turn_params, "return_logprob": False})
                meta = output["meta_info"]
                span.update(sglang_meta_attrs(meta))
            ids = output.get("output_ids")
            if ids is None:
                raise RuntimeError("Model server must return output_ids for exact multi-turn history")
            response.extend(int(x) for x in ids)
            completed_turns += 1
            finish = meta["finish_reason"]["type"]
            if finish in {"length", "abort"} or not ids:
                stop_reason = finish if ids else "empty_generation"
                status = Sample.Status.ABORTED if finish == "abort" else Sample.Status.TRUNCATED
                break
            text = output.get("text", "")
            try:
                calls = parse_calls(text)
            except (ValueError, TypeError) as exc:
                calls = None
                observations = [{"error": str(exc), "instruction": "Correct the tool call JSON and retry."}]
                metadata["format_repairs"] = metadata.get("format_repairs", 0) + 1
            if calls == []:
                metadata["predicted_answer"] = text.rsplit("</think>", 1)[-1].strip()
                stop_reason, status = "final", Sample.Status.COMPLETED
                break
            if calls:
                # Execute model-requested calls in order, as in the reference
                # harness. Concurrency is between agents, not invented tool calls.
                observations = []
                for call in calls:
                    try:
                        observations.append(await tools.call(call))
                    except WebBackendError as exc:
                        tool_errors += 1
                        if tool_errors >= max_tool_errors:
                            raise
                        # The reference web tool returns errors to the agent,
                        # allowing a different query/URL. Bound failed calls so
                        # an unavailable backend cannot create an endless run.
                        observations.append({"error": str(exc)})
            suffix = observation_tokens(tokenizer, observations)
            # SGLang returns the stop token but not the template's newline
            # following it. Preserve generated IDs and append that delimiter.
            if ids and ids[-1] == getattr(tokenizer, "eos_token_id", None):
                suffix = list(tokenizer.encode("\n", add_special_tokens=False)) + suffix
            available = min(total_budget - len(response), context_limit - len(initial) - len(response) - 512)
            if len(suffix) > available:
                stop_reason = "tool_context_budget"
                break
            response.extend(suffix)
            tool_tokens += len(suffix)
    except WebBackendError as exc:
        stop_reason, status = "web_backend_error", Sample.Status.ABORTED
        metadata["web_backend_error"] = str(exc)
        # inference.run_one uses exceptions to mark failed serving records.
        # Do not count an aborted web trajectory as successful agent throughput.
        raise
    except asyncio.CancelledError:
        stop_reason, status = "cancelled", Sample.Status.ABORTED
        raise
    except Exception:
        stop_reason, status = "harness_or_model_error", Sample.Status.ABORTED
        raise
    finally:
        metadata.update({"num_turns": completed_turns, "stop_reason": stop_reason,
                         "web_tool_events": tools.events, "search_backend": tools.backend,
                         "tool_error_count": tool_errors,
                         "official_bfcl_score": False})
        sample.status = status
        sample.tokens = initial + response
        sample.response = tokenizer.decode(response, skip_special_tokens=False)
        sample.response_length = len(response)
        sample.sample_time = time.monotonic() - started
        sample.tool_time = sum(x["elapsed_seconds"] for x in tools.events)
        sample.tool_token_count = tool_tokens
        sample.tool_call_count = len(tools.events)
        sample.search_call_count = sum(x["tool"] == "search_engine_query" for x in tools.events)
        sample.code_call_count = 0
        LOG.info("BFCL sample=%s turns=%d stop=%s tools=%d tool_seconds=%.3f",
                 sample.index, completed_turns, stop_reason, len(tools.events), sample.tool_time)
    return sample
