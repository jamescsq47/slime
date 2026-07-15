"""SFT-only BrowseComp trajectory generator, isolated from the RL agent."""

import os
from copy import deepcopy
from typing import Any

from browsecomp_agent import (
    BUDGET_MARGIN,
    _append_tokens,
    _encode_initial_prompt,
    _encode_user_observation,
    _generate_step,
    _int_env,
)
from browsecomp_env import BrowseCompEnv, SearchBackendError
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    """Generate one searched train trajectory and retain structured messages."""
    assert not evaluation, "SFT mining only accepts the BrowseComp train split"
    assert not args.partial_rollout, "BrowseComp SFT mining does not support partial rollout"
    assert isinstance(sample.prompt, list), "SFT mining requires a structured messages prompt"

    metadata = sample.metadata or {}
    sample.metadata = metadata
    question = metadata.get("question")
    label_answer = metadata.get("answer") or sample.label
    assert question and label_answer, "sample.metadata must contain question and answer"

    max_turns = _int_env("BROWSECOMP_MAX_TURNS", 60)
    turn_max_new_tokens = _int_env("BROWSECOMP_TURN_MAX_NEW_TOKENS", 1536)
    must_search = _int_env("BROWSECOMP_MUST_SEARCH", 1) == 1
    max_seq_len = getattr(args, "max_seq_len", None) or int(os.getenv("BROWSECOMP_MAX_SEQ_LEN", "36864"))

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    per_turn_max_tokens = min(turn_max_new_tokens, sampling_params.get("max_new_tokens") or turn_max_new_tokens)
    if not sample.tokens:
        sample.tokens = _encode_initial_prompt(state.tokenizer, sample.prompt)
    response_tokens: list[int] = []
    sample.loss_mask = []
    sample.rollout_log_probs = []
    messages = deepcopy(sample.prompt)
    env = BrowseCompEnv(question=question, label_answer=label_answer, must_search=must_search)
    stop_reason = "max_turns"
    num_turns = 0

    try:
        for _turn in range(max_turns):
            if len(sample.tokens) + per_turn_max_tokens + BUDGET_MARGIN >= max_seq_len:
                sample.status = Sample.Status.TRUNCATED
                stop_reason = "budget"
                break
            params = sampling_params.copy()
            params["max_new_tokens"] = per_turn_max_tokens
            text, tokens, logprobs, finish_type = await _generate_step(url, sample.tokens, params)
            num_turns += 1
            _append_tokens(sample, response_tokens, tokens, logprobs, loss_mask_value=1)
            messages.append({"role": "assistant", "content": text})
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                stop_reason = "abort"
                break
            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                stop_reason = "length"
                break
            result = await env.run_action(text)
            if result.get("action") == "finish":
                sample.status = Sample.Status.COMPLETED
                stop_reason = "finish"
                break
            observation = {"role": "user", "content": result["observation"]}
            messages.append(observation)
            observation_tokens = _encode_user_observation(state.tokenizer, observation)
            _append_tokens(sample, response_tokens, observation_tokens, [0.0] * len(observation_tokens), 0)
        else:
            sample.status = Sample.Status.COMPLETED
    except SearchBackendError:
        raise
    finally:
        await env.close()

    predicted_answer, explanation, confidence = env.predicted_answer or (None, None, None)
    metadata.update(
        trajectory_messages=messages,
        predicted_answer=predicted_answer,
        explanation=explanation,
        confidence=confidence,
        num_turns=num_turns,
        stop_reason=stop_reason,
        tool_stats=dict(env.stats),
        visited_pages=len(env.visited_pages),
    )
    sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    return sample
