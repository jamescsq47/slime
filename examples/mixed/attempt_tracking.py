from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample


MAX_ATTEMPT_HISTORY = 32


def _as_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status or "unknown"))


@dataclass
class AttemptTracker:
    sample: Sample
    attempt_count: int
    partial_resume_count: int
    restart_count: int
    resume_kind: str
    start_timestamp: float
    start_response_length: int
    start_tool_call_count: int
    start_tool_time: float
    previous_lifetime_attempt_time: float
    policy_version: int | None
    dispatch_version: int | None
    rollout_id: int | None

    @classmethod
    def begin(
        cls,
        args: Any,
        sample: Sample,
        *,
        is_partial_resume: bool,
        start_response_length: int,
        start_tool_call_count: int,
        start_tool_time: float,
    ) -> "AttemptTracker":
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = metadata
        history = metadata.setdefault("attempt_history", [])
        if not isinstance(history, list):
            history = []
            metadata["attempt_history"] = history

        previous_attempt_count = max(
            _as_nonnegative_int(getattr(sample, "attempt_count", 0)),
            _as_nonnegative_int(metadata.get("attempt_count")),
            len(history),
        )
        if is_partial_resume and previous_attempt_count == 0 and float(getattr(sample, "sample_time", 0.0) or 0.0) > 0:
            # Backward compatibility for partial samples produced before the
            # explicit attempt counter existed.
            previous_attempt_count = 1

        previous_partial_count = max(
            _as_nonnegative_int(getattr(sample, "partial_resume_count", 0)),
            _as_nonnegative_int(metadata.get("partial_resume_count")),
        )
        previous_restart_count = max(
            _as_nonnegative_int(getattr(sample, "restart_count", 0)),
            _as_nonnegative_int(metadata.get("restart_count")),
        )
        attempt_count = previous_attempt_count + 1
        partial_resume_count = previous_partial_count + int(is_partial_resume)
        is_restart = previous_attempt_count > 0 and not is_partial_resume
        restart_count = previous_restart_count + int(is_restart)
        resume_kind = "partial_resume" if is_partial_resume else ("group_restart" if is_restart else "initial")

        previous_lifetime_attempt_time = float(
            getattr(sample, "lifetime_attempt_time", 0.0)
            or metadata.get("lifetime_attempt_time", 0.0)
            or (getattr(sample, "sample_time", 0.0) if previous_attempt_count > 0 else 0.0)
            or 0.0
        )
        policy_version = metadata.get("policy_version")
        if policy_version is None:
            policy_version = getattr(args, "current_policy_version", None)
        dispatch_version = metadata.get("dispatch_version")
        rollout_id = getattr(args, "current_rollout_id", None)

        sample.attempt_count = attempt_count
        sample.partial_resume_count = partial_resume_count
        sample.restart_count = restart_count
        metadata.update(
            {
                "attempt_count": attempt_count,
                "partial_resume_count": partial_resume_count,
                "restart_count": restart_count,
            }
        )

        if getattr(args, "use_slime_dashboard", False) and previous_attempt_count > 0:
            from slime.utils.trace_utils import trace_next_attempt

            trace_next_attempt(
                sample,
                attrs={
                    "attempt_count": attempt_count,
                    "partial_resume_count": partial_resume_count,
                    "restart_count": restart_count,
                    "resume_kind": resume_kind,
                    "policy_version": policy_version,
                    "rollout_id": rollout_id,
                },
            )

        return cls(
            sample=sample,
            attempt_count=attempt_count,
            partial_resume_count=partial_resume_count,
            restart_count=restart_count,
            resume_kind=resume_kind,
            start_timestamp=time.time(),
            start_response_length=max(0, int(start_response_length)),
            start_tool_call_count=max(0, int(start_tool_call_count)),
            start_tool_time=max(0.0, float(start_tool_time)),
            previous_lifetime_attempt_time=previous_lifetime_attempt_time,
            policy_version=policy_version,
            dispatch_version=dispatch_version,
            rollout_id=rollout_id,
        )

    def finish(
        self,
        *,
        duration: float,
        cumulative_sample_time: float,
        status: Any,
        reason: str,
        response_length: int,
        tool_call_count: int,
        tool_time: float,
    ) -> dict[str, Any]:
        duration = max(0.0, float(duration))
        lifetime_attempt_time = self.previous_lifetime_attempt_time + duration
        response_length = max(0, int(response_length))
        tool_call_count = max(0, int(tool_call_count))
        tool_time = max(0.0, float(tool_time))
        entry = {
            "attempt_count": self.attempt_count,
            "resume_kind": self.resume_kind,
            "start_ts": self.start_timestamp,
            "duration": duration,
            "status": _status_value(status),
            "reason": str(reason),
            "policy_version": self.policy_version,
            "dispatch_version": self.dispatch_version,
            "rollout_id": self.rollout_id,
            "response_tokens_added": max(0, response_length - self.start_response_length),
            "tool_calls_added": max(0, tool_call_count - self.start_tool_call_count),
            "tool_time_added": max(0.0, tool_time - self.start_tool_time),
        }

        metadata = self.sample.metadata
        history = metadata.setdefault("attempt_history", [])
        history.append(entry)
        if len(history) > MAX_ATTEMPT_HISTORY:
            dropped = len(history) - MAX_ATTEMPT_HISTORY
            del history[:dropped]
            metadata["attempt_history_dropped"] = (
                _as_nonnegative_int(metadata.get("attempt_history_dropped")) + dropped
            )

        self.sample.attempt_time = duration
        self.sample.attempt_count = self.attempt_count
        self.sample.partial_resume_count = self.partial_resume_count
        self.sample.restart_count = self.restart_count
        self.sample.lifetime_attempt_time = lifetime_attempt_time
        metadata.update(
            {
                "attempt_time": duration,
                "attempt_count": self.attempt_count,
                "partial_resume_count": self.partial_resume_count,
                "restart_count": self.restart_count,
                "cumulative_sample_time": float(cumulative_sample_time),
                "lifetime_attempt_time": lifetime_attempt_time,
                "last_attempt_status": entry["status"],
                "last_attempt_reason": entry["reason"],
            }
        )
        return entry
