"""Small request-side helpers for SGLang's agentic KV lifecycle.

The feature is opt-in through ``SGLANG_AGENTIC_KV_LIFECYCLE``.  Keeping the
wire metadata construction here avoids coupling application agents to SGLang
internals and leaves existing profiling scripts unchanged when the flag is off.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any


REQUEST_ID_KEY = "agentic_request_id"
GENERATION_KEY = "agentic_generation"
PARENT_GENERATION_KEY = "agentic_parent_generation"
TOOL_TYPE_KEY = "agentic_tool_type"
TOOL_SUFFIXES_KEY = "agentic_tool_suffix_token_ids"
TERMINAL_MARKERS_KEY = "agentic_terminal_marker_token_ids"
TOOL_SUFFIX_STRINGS_KEY = "agentic_tool_suffix_strings"
TERMINAL_MARKER_STRINGS_KEY = "agentic_terminal_marker_strings"
EXTRA_KEY_ENVELOPE_PREFIX = "agentic-v1e:"
AGENTIC_WIRE_KEYS = frozenset(
    {
        REQUEST_ID_KEY,
        GENERATION_KEY,
        PARENT_GENERATION_KEY,
        TOOL_TYPE_KEY,
        TOOL_SUFFIXES_KEY,
        TERMINAL_MARKERS_KEY,
        TOOL_SUFFIX_STRINGS_KEY,
        TERMINAL_MARKER_STRINGS_KEY,
    }
)


def generation_has_visible_content(token_ids: Iterable[int], tokenizer: Any) -> bool:
    """Return whether a model turn contains non-special, visible text.

    In disaggregated serving, an EOS-only first token can finish on P before
    the request ever reaches D.  Such a turn has a non-empty token list but no
    D-side KV snapshot, so it must not be followed by a repair generation.
    """

    ids = list(token_ids)
    if not ids:
        return False
    try:
        text = tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        text = tokenizer.decode(ids)
    return bool(str(text).strip())


def _urlsafe_json(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _normalize_markers(markers: Iterable[str] | str, *, field: str) -> tuple[str, ...]:
    """Accept a single marker or an iterable without splitting strings.

    Dataset adapters frequently load one delimiter directly from JSON/YAML,
    where it is naturally a string rather than a list.  Treating ``str`` as a
    generic iterable would silently turn it into one marker per character.
    """

    if isinstance(markers, str):
        values = (markers,)
    else:
        try:
            values = tuple(markers)
        except TypeError as exc:
            raise TypeError(f"{field} must be a string or iterable of strings") from exc
    if any(not isinstance(marker, str) for marker in values):
        raise TypeError(f"{field} must contain only strings")
    if any(not marker for marker in values):
        raise ValueError(f"{field} must not contain empty strings")
    return values


def build_agentic_extra_key(
    request_id: str, sampling_params: Mapping[str, Any]
) -> str:
    """Carry lifecycle metadata through routers that drop ``custom_params``.

    The radix namespace is generation-scoped.  A P->D ACK from generation N
    is allowed to release that P branch while generation N+1 is concurrently
    receiving reverse KV; using a trajectory-only key would let the older ACK
    delete the newly inserted branch.  Cross-turn locality comes from the
    explicit complete-snapshot handoff, not from keeping an ambiguous local
    branch alive.
    """

    custom_params = sampling_params.get("custom_params")
    if not isinstance(custom_params, Mapping):
        raise ValueError("agentic sampling_params must contain custom_params")
    if str(custom_params.get(REQUEST_ID_KEY)) != str(request_id):
        raise ValueError("agentic request id does not match custom_params")
    generation = custom_params.get(GENERATION_KEY)
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ValueError("agentic generation must be an integer")
    if generation < 0:
        raise ValueError("agentic generation must be non-negative")
    stable_key = f"agentic-v1:{request_id}:g{generation}"
    stable = base64.urlsafe_b64encode(stable_key.encode("utf-8")).decode("ascii")
    stable = stable.rstrip("=")
    wire_params = {
        key: value for key, value in custom_params.items() if key in AGENTIC_WIRE_KEYS
    }
    envelope = f"{EXTRA_KEY_ENVELOPE_PREFIX}{stable}:{_urlsafe_json(wire_params)}"
    if len(envelope) > 32768:
        raise ValueError("agentic extra_key envelope is too large")
    return envelope


def lifecycle_enabled() -> bool:
    return os.getenv("SGLANG_AGENTIC_KV_LIFECYCLE", "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def confirm_agentic_generation_final(
    trajectory_metadata: Mapping[str, Any],
    generation: int,
    *,
    p_ready_dir: str,
) -> bool:
    """Best-effort application ACK that this generation ended the trajectory."""

    p_ready_dir = p_ready_dir or os.getenv("PD_P_READY_DIR", "")
    if not lifecycle_enabled() or not p_ready_dir:
        return False
    request_id = trajectory_metadata.get(REQUEST_ID_KEY)
    if request_id is None:
        return False
    try:
        from sglang.srt.disaggregation.agentic_early_claim import (
            AgenticEarlyClaimStore,
        )
        from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration

        store = AgenticEarlyClaimStore(os.path.join(p_ready_dir, "early-claims"))
        store.publish_final(RequestGeneration(str(request_id), generation))
        return True
    except (OSError, TypeError, ValueError):
        # Missing an optimization ACK must not fail an otherwise valid sample.
        return False


def confirm_agentic_generation_tool(
    trajectory_metadata: Mapping[str, Any],
    generation: int,
    *,
    p_ready_dir: str,
) -> bool:
    """ACK that the application parser accepted this generation's tool call."""

    p_ready_dir = p_ready_dir or os.getenv("PD_P_READY_DIR", "")
    if not lifecycle_enabled() or not p_ready_dir:
        return False
    request_id = trajectory_metadata.get(REQUEST_ID_KEY)
    if request_id is None:
        return False
    try:
        from sglang.srt.disaggregation.agentic_early_claim import (
            AgenticEarlyClaimStore,
        )
        from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration

        store = AgenticEarlyClaimStore(os.path.join(p_ready_dir, "early-claims"))
        store.publish_tool(RequestGeneration(str(request_id), generation))
        return True
    except (OSError, TypeError, ValueError):
        # Missing this optimization ACK safely disables slow-path preservation
        # for the generation; correctness falls back to Prefill recomputation.
        return False


def ensure_request_id(metadata: MutableMapping[str, Any]) -> str:
    """Return the trajectory-stable id, creating it once if necessary."""

    request_id = metadata.get(REQUEST_ID_KEY)
    if request_id is None:
        request_id = uuid.uuid4().hex
        metadata[REQUEST_ID_KEY] = request_id
    request_id = str(request_id)
    if not request_id:
        raise ValueError("agentic request id must be non-empty")
    return request_id


def encode_tool_suffixes(
    tokenizer: Any, markers: Iterable[str] | str, *, field: str = "markers"
) -> list[list[int]]:
    suffixes: list[list[int]] = []
    for marker in _normalize_markers(markers, field=field):
        tokens = tokenizer.encode(marker, add_special_tokens=False)
        if tokens:
            suffixes.append([int(token) for token in tokens])
    return suffixes


def add_agentic_kv_metadata(
    sampling_params: Mapping[str, Any],
    *,
    trajectory_metadata: MutableMapping[str, Any],
    generation: int,
    tokenizer: Any,
    tool_type: str,
    tool_suffix_markers: Iterable[str] | str,
    terminal_markers: Iterable[str] | str = (),
) -> tuple[dict[str, Any], str]:
    """Copy sampling params and add minimal request-generation wire metadata."""

    if generation < 0:
        raise ValueError("generation must be non-negative")
    result = dict(sampling_params)
    request_id = ensure_request_id(trajectory_metadata)
    tool_suffix_markers = _normalize_markers(
        tool_suffix_markers, field="tool_suffix_markers"
    )
    terminal_markers = _normalize_markers(
        terminal_markers, field="terminal_markers"
    )
    custom_params = dict(result.get("custom_params") or {})
    custom_params.update(
        {
            REQUEST_ID_KEY: request_id,
            GENERATION_KEY: generation,
            TOOL_TYPE_KEY: tool_type,
            TOOL_SUFFIXES_KEY: encode_tool_suffixes(
                tokenizer, tool_suffix_markers, field="tool_suffix_markers"
            ),
            TERMINAL_MARKERS_KEY: encode_tool_suffixes(
                tokenizer, terminal_markers, field="terminal_markers"
            ),
            # Keep the original text as the canonical, tokenizer-independent
            # representation.  Token ids remain for backwards compatibility
            # and the allocation-free fast path.
            TOOL_SUFFIX_STRINGS_KEY: [str(marker) for marker in tool_suffix_markers],
            TERMINAL_MARKER_STRINGS_KEY: [str(marker) for marker in terminal_markers],
        }
    )
    if generation > 0:
        custom_params[PARENT_GENERATION_KEY] = generation - 1
    else:
        custom_params.pop(PARENT_GENERATION_KEY, None)
    result["custom_params"] = custom_params
    return result, request_id
