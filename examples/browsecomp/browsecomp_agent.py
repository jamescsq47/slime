"""ReAct generate function for BrowseComp-Plus RL training.

Plugged into slime via:
  --custom-generate-function-path browsecomp_agent.generate

The model emits reasoning plus a text-format function call
(`<function=search|open_page|finish>...</function>`, as specified in the
system prompt shipped with the data); the call is executed against the
BrowseComp-Plus local search server and the observation is appended as a
user message. Assistant tokens receive loss mask 1; tool observation tokens
receive loss mask 0.

Environment variables:
  LOCAL_SEARCH_URL                   BrowseComp-Plus search server (required)
  BROWSECOMP_MAX_TURNS               max ReAct turns (default 100)
  BROWSECOMP_TURN_MAX_NEW_TOKENS     per-turn completion cap (default 2048)
  BROWSECOMP_MUST_SEARCH             require >=1 search before finish counts (default 1)
  BROWSECOMP_DO_NOT_GIVE_UP          nudge on "insufficient" answers (default 0)
  BROWSECOMP_ENABLE_THINKING         pass Qwen3 enable_thinking to chat template when set
"""

import logging
import os
import time
from copy import deepcopy
from typing import Any

import httpx

from browsecomp_env import BrowseCompEnv, SearchBackendError
from slime.dashboard.api import span as dashboard_span
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Reserve room for the next turn's completion plus template overhead when
# deciding whether another turn fits in the max_seq_len budget.
BUDGET_MARGIN = 512

DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool_env(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _render_messages(tokenizer, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    kwargs = {}
    enable_thinking = _bool_env("BROWSECOMP_ENABLE_THINKING")
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def _encode_initial_prompt(tokenizer, prompt: str | list[dict[str, Any]]) -> list[int]:
    if isinstance(prompt, list):
        rendered = _render_messages(tokenizer, prompt, add_generation_prompt=True)
        return tokenizer.encode(rendered, add_special_tokens=False)
    if isinstance(prompt, str):
        return tokenizer.encode(prompt, add_special_tokens=False)
    raise TypeError(f"prompt must be a string or messages list, got {type(prompt)}")


def _encode_user_observation(tokenizer, message: dict[str, str]) -> list[int]:
    """Encode one appended user observation plus the next assistant prompt."""
    dummy = _render_messages(tokenizer, DUMMY_MESSAGES, add_generation_prompt=False)
    rendered = _render_messages(tokenizer, DUMMY_MESSAGES + [message], add_generation_prompt=True)
    dummy_ids = tokenizer.encode(dummy, add_special_tokens=False)
    rendered_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if rendered_ids[: len(dummy_ids)] == dummy_ids:
        obs_ids = rendered_ids[len(dummy_ids) :]
    else:
        # Template prefix mismatch should be rare; fall back to a conservative
        # text encoding so the rollout can continue instead of crashing.
        obs_ids = tokenizer.encode(message["content"], add_special_tokens=False)

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_ids and obs_ids[0] == bos_id:
        obs_ids = obs_ids[1:]
    return obs_ids


def _append_tokens(
    sample: Sample,
    response_tokens: list[int],
    tokens: list[int],
    logprobs: list[float],
    loss_mask_value: int,
) -> None:
    sample.tokens.extend(tokens)
    response_tokens.extend(tokens)
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.rollout_log_probs.extend(logprobs)
    sample.loss_mask = sample.loss_mask or []
    sample.loss_mask.extend([loss_mask_value] * len(tokens))
    sample.response_length = len(response_tokens)


async def _generate_step(
    url: str, tokens: list[int], sampling_params: dict[str, Any]
) -> tuple[str, list[int], list[float], str]:
    output = await post(
        url,
        {
            "input_ids": tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
        },
    )
    meta_info = output["meta_info"]
    if "output_token_logprobs" in meta_info:
        new_tokens = [item[1] for item in meta_info["output_token_logprobs"]]
        new_logprobs = [item[0] for item in meta_info["output_token_logprobs"]]
    else:
        new_tokens, new_logprobs = [], []
    return output["text"], new_tokens, new_logprobs, meta_info["finish_reason"]["type"]


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    """slime custom generate entry point for BrowseComp ReAct rollouts."""
    metadata = sample.metadata or {}
    question = metadata.get("question")
    label_answer = metadata.get("answer") or sample.label
    assert question and label_answer, (
        "sample.metadata must contain 'question' and 'answer'; "
        "did you prepare the data with examples/browsecomp/prepare_data.py?"
    )

    max_turns = _int_env("BROWSECOMP_MAX_TURNS", 100)
    turn_max_new_tokens = _int_env("BROWSECOMP_TURN_MAX_NEW_TOKENS", 2048)
    must_search = os.getenv("BROWSECOMP_MUST_SEARCH", "1") == "1"
    max_seq_len = getattr(args, "max_seq_len", None)
    if max_seq_len is None and os.getenv("BROWSECOMP_MAX_SEQ_LEN"):
        max_seq_len = int(os.getenv("BROWSECOMP_MAX_SEQ_LEN"))
    if max_seq_len is None:
        max_seq_len = getattr(args, "sglang_context_length", None)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    per_turn_max_tokens = min(turn_max_new_tokens, sampling_params.get("max_new_tokens") or turn_max_new_tokens)

    resume_state = metadata.get("browsecomp_partial_state") if args.partial_rollout else None
    is_resume = sample.status == Sample.Status.ABORTED and isinstance(resume_state, dict)
    if not sample.tokens:
        sample.tokens = _encode_initial_prompt(state.tokenizer, sample.prompt)
        metadata["browsecomp_prompt_length"] = len(sample.tokens)
    if is_resume:
        prompt_length = int(metadata.get("browsecomp_prompt_length", len(sample.tokens) - sample.response_length))
        response_tokens = list(sample.tokens[prompt_length:])
        sample.loss_mask = list(sample.loss_mask or [])
        sample.rollout_log_probs = list(sample.rollout_log_probs or [])
        sample.status = Sample.Status.PENDING
    else:
        response_tokens: list[int] = []
        sample.loss_mask = []
        sample.rollout_log_probs = []

    messages = deepcopy(sample.prompt) if isinstance(sample.prompt, list) else None
    env = BrowseCompEnv(question=question, label_answer=label_answer, must_search=must_search)
    stop_reason = "max_turns"
    num_turns = int(resume_state.get("num_turns", 0)) if is_resume else 0
    accrued_sample_time = float(getattr(sample, "sample_time", 0.0) or 0.0) if is_resume else 0.0
    tool_time = float(getattr(sample, "tool_time", 0.0) or 0.0) if is_resume else 0.0
    pending_response_text = str(resume_state.get("pending_response_text", "")) if is_resume else ""
    segment_started = time.monotonic()
    if is_resume:
        env.visited_pages = set(resume_state.get("visited_pages", []))
        env.stats.update(resume_state.get("tool_stats", {}))
        predicted = resume_state.get("predicted_answer")
        env.predicted_answer = tuple(predicted) if predicted else None

    def save_partial_state() -> None:
        metadata["browsecomp_partial_state"] = {
            "num_turns": num_turns,
            "visited_pages": sorted(env.visited_pages),
            "tool_stats": dict(env.stats),
            "predicted_answer": list(env.predicted_answer) if env.predicted_answer else None,
            "pending_response_text": pending_response_text,
        }
        sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(response_tokens)
        sample.sample_time = accrued_sample_time + (time.monotonic() - segment_started)
        sample.tool_time = tool_time

    try:
        for _turn in range(num_turns, max_turns):
            if max_seq_len is not None and len(sample.tokens) + per_turn_max_tokens + BUDGET_MARGIN >= max_seq_len:
                sample.status = Sample.Status.TRUNCATED
                stop_reason = "budget"
                break

            cur_sampling_params = sampling_params.copy()
            cur_sampling_params["max_new_tokens"] = per_turn_max_tokens
            with dashboard_span(
                args,
                sample,
                "generation_turn",
                attrs={"task_type": "qa", "turn": num_turns + 1, "max_new_tokens": per_turn_max_tokens},
            ) as generation_span:
                response_text, new_tokens, new_logprobs, finish_type = await _generate_step(
                    url, sample.tokens, cur_sampling_params
                )
                generation_span.update(
                    {"finish_reason": finish_type, "completion_tokens": len(new_tokens)}
                )
            num_turns += 1
            _append_tokens(sample, response_tokens, new_tokens, new_logprobs, loss_mask_value=1)
            action_response_text = pending_response_text + response_text

            if finish_type == "abort":
                pending_response_text = action_response_text
                sample.status = Sample.Status.ABORTED
                stop_reason = "abort"
                save_partial_state()
                break
            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                stop_reason = "length"
                break

            pending_response_text = ""
            if messages is not None:
                messages.append({"role": "assistant", "content": action_response_text})

            tool_started = time.monotonic()
            tool_stats_before = dict(env.stats)
            with dashboard_span(
                args,
                sample,
                "tool_call",
                attrs={"task_type": "qa", "turn": num_turns},
            ) as tool_span:
                result = await env.run_action(action_response_text)
                tool_actions = {
                    name: int(env.stats.get(name, 0)) - int(tool_stats_before.get(name, 0))
                    for name in ("search", "open_page")
                }
                tool_actions = {name: count for name, count in tool_actions.items() if count > 0}
                tool_span.update(
                    {
                        "action": result.get("action", ",".join(tool_actions) or "invalid"),
                        "tool_calls": sum(tool_actions.values()),
                        "is_tool_call": bool(tool_actions),
                    }
                )
            tool_time += time.monotonic() - tool_started
            if result.get("action") == "finish":
                sample.status = Sample.Status.COMPLETED
                stop_reason = "finish"
                break

            observation_msg = {"role": "user", "content": result["observation"]}
            if messages is not None:
                messages.append(observation_msg)
            obs_tokens = _encode_user_observation(state.tokenizer, observation_msg)
            _append_tokens(sample, response_tokens, obs_tokens, [0.0] * len(obs_tokens), loss_mask_value=0)
        else:
            sample.status = Sample.Status.COMPLETED
    except SearchBackendError:
        raise
    finally:
        await env.close()

    if sample.status != Sample.Status.ABORTED:
        metadata.pop("browsecomp_partial_state", None)
        sample.sample_time = accrued_sample_time + (time.monotonic() - segment_started)
        sample.tool_time = tool_time

    predicted_answer, explanation, confidence = env.predicted_answer or (None, None, None)
    sample.metadata.update(
        {
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "confidence": confidence,
            "num_turns": num_turns,
            "stop_reason": stop_reason,
            "tool_stats": dict(env.stats),
            "visited_pages": len(env.visited_pages),
            "tool_time": sample.tool_time,
            "sample_time": sample.sample_time,
        }
    )
    sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    return sample


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    metadata = metadata or {}
    request_kwargs = dict(request_kwargs or {})

    question = metadata.get("question")
    label_answer = metadata.get("answer")
    assert question and label_answer, (
        "sample.metadata must contain 'question' and 'answer' — "
        "did you prepare the data with examples/browsecomp/prepare_data.py?"
    )

    max_turns = _int_env("BROWSECOMP_MAX_TURNS", 100)
    turn_max_new_tokens = _int_env("BROWSECOMP_TURN_MAX_NEW_TOKENS", 2048)
    must_search = os.getenv("BROWSECOMP_MUST_SEARCH", "1") == "1"
    max_seq_len = metadata.get("max_seq_len") or os.getenv("BROWSECOMP_MAX_SEQ_LEN")
    if max_seq_len is not None:
        max_seq_len = int(max_seq_len)

    max_tokens = request_kwargs.pop("max_tokens", None)
    per_turn_max_tokens = min(turn_max_new_tokens, max_tokens) if max_tokens else turn_max_new_tokens

    assert isinstance(prompt, list), f"prompt must be a messages list, got {type(prompt)}"
    messages = deepcopy(prompt)

    env = BrowseCompEnv(question=question, label_answer=label_answer, must_search=must_search)
    num_turns = 0
    stop_reason = "max_turns"

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            for _turn in range(max_turns):
                payload = {
                    "messages": messages,
                    "max_tokens": per_turn_max_tokens,
                    **request_kwargs,
                }
                resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
                if resp.status_code != 200:
                    # Typically 400 from sglang when the accumulated context no
                    # longer fits; end the rollout so the caller can
                    # finalize the partial trajectory.
                    logger.warning(
                        "chat/completions returned %d, ending rollout: %s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    stop_reason = f"http_{resp.status_code}"
                    break
                data = resp.json()
                num_turns += 1

                choice = data["choices"][0]
                assistant_msg = choice["message"]
                finish_reason = choice.get("finish_reason")
                messages.append(assistant_msg)

                if finish_reason == "abort":
                    stop_reason = "abort"
                    break

                result = await env.run_action(assistant_msg.get("content") or "")
                if result.get("action") == "finish":
                    stop_reason = "finish"
                    break

                messages.append({"role": "user", "content": result["observation"]})

                # Stop when the next turn cannot fit in the sequence budget.
                usage = data.get("usage") or {}
                total_tokens = usage.get("total_tokens")
                if (
                    max_seq_len is not None
                    and total_tokens is not None
                    and total_tokens + per_turn_max_tokens + BUDGET_MARGIN >= max_seq_len
                ):
                    stop_reason = "budget"
                    break
    except SearchBackendError:
        # The search backend is down or persistently failing. Re-raise so
        # Keep backend failures explicit; reward scoring only applies to completed
        # rollouts with a valid submitted answer.
        raise
    finally:
        await env.close()

    predicted_answer, explanation, confidence = env.predicted_answer or (None, None, None)
    return {
        "predicted_answer": predicted_answer,
        "explanation": explanation,
        "confidence": confidence,
        "num_turns": num_turns,
        "stop_reason": stop_reason,
        "tool_stats": dict(env.stats),
        "visited_pages": len(env.visited_pages),
    }
