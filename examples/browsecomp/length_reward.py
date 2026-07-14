# Group-relative length penalty for BrowseComp GRPO training.
# Ported from examples/search-r1/length_reward.py (exp/search-r1-length-penalty)
# with env vars renamed to BROWSECOMP_LENGTH_PENALTY_* and an ENABLED gate.
#
# Wire with:
#   --custom-reward-post-process-path length_reward.post_process_rewards
# Raw rewards stay the BrowseComp judge correctness score; only the shaped
# rewards fed to GRPO advantages are modified, and only for successful samples
# in groups with >= 2 successes.

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from slime.utils.types import Sample


logger = logging.getLogger(__name__)


# Running EMA of the reference length when GLOBAL_REF mode is on. Lives for
# the duration of the rollout-manager process, which persists across batches.
_GLOBAL_REF_STATE: dict = {"ref_len": None}


@dataclass(frozen=True)
class LengthPenaltyConfig:
    enabled: bool
    global_ref: bool
    global_ref_alpha: float
    trunc_penalty: float
    success_threshold: float
    beta: float
    cap: float
    ref_quantile: float
    relative_slack: float
    absolute_slack: float
    success_floor: float
    require_completed: bool
    log_stats: bool


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _config() -> LengthPenaltyConfig:
    return LengthPenaltyConfig(
        enabled=_env_bool("BROWSECOMP_LENGTH_PENALTY_ENABLED", True),
        # GLOBAL_REF: use a cross-batch EMA of successful-sample lengths as the
        # reference instead of the within-group quantile, so groups with a
        # single success are penalized too (within-group mode needs >= 2).
        global_ref=_env_bool("BROWSECOMP_LENGTH_PENALTY_GLOBAL_REF", False),
        global_ref_alpha=_env_float("BROWSECOMP_LENGTH_PENALTY_GLOBAL_REF_ALPHA", 0.1),
        # TRUNC_PENALTY > 0 subtracts a flat penalty from FAILED samples that
        # wasted the budget: token-truncated rollouts, or rollouts that ended
        # without ever submitting a finish answer. ABORTED samples (backend
        # failures, not the policy's fault) are exempt. Teaches "wrap up
        # within budget" — failures otherwise carry no length signal at all.
        trunc_penalty=_env_float("BROWSECOMP_LENGTH_PENALTY_TRUNC_PENALTY", 0.0),
        success_threshold=_env_float("BROWSECOMP_LENGTH_PENALTY_SUCCESS_THRESHOLD", 1.0),
        beta=_env_float("BROWSECOMP_LENGTH_PENALTY_BETA", 0.10),
        cap=_env_float("BROWSECOMP_LENGTH_PENALTY_CAP", 0.20),
        ref_quantile=_env_float("BROWSECOMP_LENGTH_PENALTY_REF_QUANTILE", 0.25),
        relative_slack=_env_float("BROWSECOMP_LENGTH_PENALTY_REL_SLACK", 0.10),
        absolute_slack=_env_float("BROWSECOMP_LENGTH_PENALTY_ABS_SLACK", 16.0),
        success_floor=_env_float("BROWSECOMP_LENGTH_PENALTY_SUCCESS_FLOOR", 0.80),
        require_completed=_env_bool("BROWSECOMP_LENGTH_PENALTY_REQUIRE_COMPLETED", True),
        log_stats=_env_bool("BROWSECOMP_LENGTH_PENALTY_LOG_STATS", True),
    )


def _flatten_and_group(samples: list[Sample] | list[list[Sample]], args) -> tuple[list[Sample], list[list[Sample]]]:
    if not samples:
        return [], []

    if isinstance(samples[0], list):
        groups = samples
        flat = [sample for group in groups for sample in group]
        return flat, groups

    flat = samples
    if all(getattr(sample, "group_index", None) is not None for sample in flat):
        groups_by_index: dict[Any, list[Sample]] = {}
        for sample in flat:
            groups_by_index.setdefault(sample.group_index, []).append(sample)
        return flat, list(groups_by_index.values())

    group_size = max(1, int(getattr(args, "n_samples_per_prompt", 1) or 1))
    groups = [flat[i : i + group_size] for i in range(0, len(flat), group_size)]
    return flat, groups


def _effective_length(sample: Sample) -> int:
    if sample.loss_mask is not None:
        return int(sum(sample.loss_mask))
    return int(sample.response_length)


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    q = min(1.0, max(0.0, q))
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _is_success(sample: Sample, reward: float, cfg: LengthPenaltyConfig) -> bool:
    if reward < cfg.success_threshold:
        return False
    if cfg.require_completed and sample.status != Sample.Status.COMPLETED:
        return False
    return True


def _normalize_rewards(
    args,
    groups: list[list[Sample]],
    flat_samples: list[Sample],
    rewards: list[float],
) -> list[float]:
    if not rewards:
        return []
    if len(flat_samples) != len(rewards):
        raise ValueError(f"Length mismatch: got {len(flat_samples)} samples but {len(rewards)} rewards.")
    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        shaped_by_sample_id = {id(sample): reward for sample, reward in zip(flat_samples, rewards, strict=True)}
        normalized_by_sample_id = {}

        for group in groups:
            group_rewards = [shaped_by_sample_id[id(sample)] for sample in group]
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float)
            rewards_tensor = rewards_tensor - rewards_tensor.mean()

            if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
                if len(group) > 1:
                    std = rewards_tensor.std()
                    rewards_tensor = rewards_tensor / (std + 1e-6)
                else:
                    rewards_tensor = torch.zeros_like(rewards_tensor)

            for sample, reward in zip(group, rewards_tensor.tolist(), strict=True):
                normalized_by_sample_id[id(sample)] = reward

        return [normalized_by_sample_id[id(sample)] for sample in flat_samples]

    return rewards


def _apply_length_penalty(
    args,
    flat_samples: list[Sample],
    groups: list[list[Sample]],
    raw_rewards: list[float],
    cfg: LengthPenaltyConfig,
) -> list[float]:
    raw_by_sample_id = {id(sample): reward for sample, reward in zip(flat_samples, raw_rewards, strict=True)}
    shaped_by_sample_id = dict(raw_by_sample_id)
    penalty_by_sample_id = {id(sample): 0.0 for sample in flat_samples}
    success_by_sample_id = {id(sample): 0.0 for sample in flat_samples}
    eligible_group_by_sample_id = {id(sample): 0.0 for sample in flat_samples}

    eligible_groups = 0
    penalized_samples = 0
    total_penalty = 0.0
    max_penalty = 0.0

    successes_by_group: list[list[tuple[Sample, int, float]]] = []
    for group in groups:
        successes = []
        for sample in group:
            reward = raw_by_sample_id[id(sample)]
            if _is_success(sample, reward, cfg):
                success_by_sample_id[id(sample)] = 1.0
                successes.append((sample, _effective_length(sample), reward))
        successes_by_group.append(successes)

    global_ref_len = None
    if cfg.global_ref:
        batch_lengths = [length for successes in successes_by_group for _, length, _ in successes]
        if batch_lengths:
            batch_ref = max(1.0, _quantile(batch_lengths, cfg.ref_quantile))
            prev = _GLOBAL_REF_STATE["ref_len"]
            global_ref_len = (
                batch_ref if prev is None else cfg.global_ref_alpha * batch_ref + (1.0 - cfg.global_ref_alpha) * prev
            )
            _GLOBAL_REF_STATE["ref_len"] = global_ref_len
        else:
            global_ref_len = _GLOBAL_REF_STATE["ref_len"]

    for group, successes in zip(groups, successes_by_group, strict=True):
        if cfg.global_ref:
            # Any group with at least one success is eligible; the reference
            # comes from the cross-batch EMA (None only before any success
            # has ever been seen).
            if not successes or global_ref_len is None:
                continue
            ref_len = global_ref_len
        else:
            if len(successes) < 2:
                continue
            ref_len = max(1.0, _quantile([length for _, length, _ in successes], cfg.ref_quantile))

        eligible_groups += 1
        for sample in group:
            eligible_group_by_sample_id[id(sample)] = 1.0

        allowed_len = ref_len * (1.0 + cfg.relative_slack) + cfg.absolute_slack

        for sample, length, reward in successes:
            excess = max(0.0, float(length) - allowed_len)
            penalty = min(cfg.cap, cfg.beta * excess / ref_len) if cfg.beta > 0 and cfg.cap > 0 else 0.0
            floor = min(cfg.success_floor, reward)
            shaped_reward = max(floor, reward - penalty)
            shaped_by_sample_id[id(sample)] = shaped_reward
            penalty_by_sample_id[id(sample)] = penalty

            if penalty > 0:
                penalized_samples += 1
                total_penalty += penalty
                max_penalty = max(max_penalty, penalty)

    # Flat truncation penalty for failed samples that wasted the budget.
    trunc_penalized = 0
    trunc_by_sample_id = {id(sample): 0.0 for sample in flat_samples}
    if cfg.trunc_penalty > 0:
        for sample in flat_samples:
            raw_reward = raw_by_sample_id[id(sample)]
            if raw_reward >= cfg.success_threshold:
                continue
            if sample.status == Sample.Status.ABORTED:
                continue
            no_answer = not (sample.metadata or {}).get("predicted_answer")
            if sample.status == Sample.Status.TRUNCATED or no_answer:
                shaped_by_sample_id[id(sample)] = raw_reward - cfg.trunc_penalty
                trunc_by_sample_id[id(sample)] = cfg.trunc_penalty
                trunc_penalized += 1

    for sample in flat_samples:
        sample_id = id(sample)
        raw_reward = raw_by_sample_id[sample_id]
        shaped_reward = shaped_by_sample_id[sample_id]
        penalty = penalty_by_sample_id[sample_id]
        sample.metadata["length_trunc_penalty"] = trunc_by_sample_id[sample_id]
        sample.metadata["length_penalty"] = penalty
        sample.metadata["length_penalty_applied"] = 1.0 if penalty > 0 else 0.0
        sample.metadata["length_base_reward"] = raw_reward
        sample.metadata["length_shaped_reward"] = shaped_reward
        sample.metadata["length_reward_delta"] = raw_reward - shaped_reward
        sample.metadata["length_effective_response_length"] = _effective_length(sample)
        sample.metadata["length_success"] = success_by_sample_id[sample_id]
        sample.metadata["length_group_eligible"] = eligible_group_by_sample_id[sample_id]

    if cfg.log_stats:
        mean_penalty = total_penalty / penalized_samples if penalized_samples else 0.0
        logger.info(
            "browsecomp_length_penalty groups=%d eligible_groups=%d penalized_samples=%d "
            "mean_penalty=%.6f max_penalty=%.6f trunc_penalized=%d",
            len(groups),
            eligible_groups,
            penalized_samples,
            mean_penalty,
            max_penalty,
            trunc_penalized,
        )

    return [shaped_by_sample_id[id(sample)] for sample in flat_samples]


def post_process_rewards(args, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    """Apply group-relative length penalty before GRPO reward normalization.

    Raw rewards stay as the base EM rewards for logging and pass@k metrics. The
    processed rewards returned here are the shaped rewards used for advantages.
    """
    flat_samples, groups = _flatten_and_group(samples, args)
    raw_rewards = [float(sample.get_reward_value(args)) for sample in flat_samples]
    cfg = _config()
    if cfg.enabled:
        shaped_rewards = _apply_length_penalty(args, flat_samples, groups, raw_rewards, cfg)
    else:
        shaped_rewards = raw_rewards
    return raw_rewards, _normalize_rewards(args, groups, flat_samples, shaped_rewards)
