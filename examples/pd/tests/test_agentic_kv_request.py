import base64
import json
import tempfile

import numpy as np
import pytest

from agentic_kv_request import (
    add_agentic_kv_metadata,
    build_agentic_extra_key,
    confirm_agentic_generation_final,
    confirm_agentic_generation_tool,
    ensure_request_id,
    generation_has_visible_content,
)
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
from sglang.srt.disaggregation.utils import kv_to_page_indices


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        values = list(token_ids)
        if skip_special_tokens:
            values = [value for value in values if value != 0]
        return "".join(chr(value) for value in values)


def test_page_indices_use_int32_wire_format_for_torch_style_int64_input():
    token_indices = np.arange(14 * 64, 24 * 64, dtype=np.int64)
    pages = kv_to_page_indices(token_indices, page_size=64)
    assert pages.dtype == np.int32
    assert pages.tolist() == list(range(14, 24))


def test_generation_visible_content_rejects_empty_and_special_only_tokens():
    tokenizer = FakeTokenizer()
    assert not generation_has_visible_content([], tokenizer)
    assert not generation_has_visible_content([0], tokenizer)
    assert generation_has_visible_content([0, ord("x")], tokenizer)


def test_request_id_is_created_once():
    metadata = {}
    first = ensure_request_id(metadata)
    assert ensure_request_id(metadata) == first
    assert metadata["agentic_request_id"] == first


def test_final_confirmation_uses_environment_ready_dir(monkeypatch):
    with tempfile.TemporaryDirectory(dir="/dev/shm") as ready_dir:
        monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", "true")
        monkeypatch.setenv("PD_P_READY_DIR", ready_dir)
        assert confirm_agentic_generation_final(
            {"agentic_request_id": "done"}, 3, p_ready_dir=""
        )
        store = AgenticEarlyClaimStore(f"{ready_dir}/early-claims")
        assert store.read_final(
            RequestGeneration("done", 3),
            not_before=0.0,
            max_age_seconds=5.0,
        ) is not None


def test_tool_confirmation_uses_environment_ready_dir(monkeypatch):
    with tempfile.TemporaryDirectory(dir="/dev/shm") as ready_dir:
        monkeypatch.setenv("SGLANG_AGENTIC_KV_LIFECYCLE", "true")
        monkeypatch.setenv("PD_P_READY_DIR", ready_dir)
        assert confirm_agentic_generation_tool(
            {"agentic_request_id": "tool"}, 2, p_ready_dir=""
        )
        store = AgenticEarlyClaimStore(f"{ready_dir}/early-claims")
        assert store.read_tool(
            RequestGeneration("tool", 2),
            not_before=0.0,
            max_age_seconds=5.0,
        ) is not None


def test_add_agentic_metadata_preserves_existing_custom_params():
    trajectory = {"agentic_request_id": "trajectory-1"}
    params, request_id = add_agentic_kv_metadata(
        {"temperature": 0, "custom_params": {"application": "keep-me"}},
        trajectory_metadata=trajectory,
        generation=2,
        tokenizer=FakeTokenizer(),
        tool_type="code_interpreter",
        tool_suffix_markers=("</tool_call>",),
        terminal_markers=("<answer>",),
    )
    custom = params["custom_params"]
    assert request_id == "trajectory-1"
    assert custom["application"] == "keep-me"
    assert custom["agentic_generation"] == 2
    assert custom["agentic_parent_generation"] == 1
    assert custom["agentic_tool_suffix_token_ids"] == [
        [ord(char) for char in "</tool_call>"]
    ]
    assert custom["agentic_terminal_marker_token_ids"] == [
        [ord(char) for char in "<answer>"]
    ]
    assert custom["agentic_tool_suffix_strings"] == ["</tool_call>"]
    assert custom["agentic_terminal_marker_strings"] == ["<answer>"]


def test_agentic_extra_key_is_router_safe_and_excludes_application_metadata():
    params = {
        "custom_params": {
            "agentic_request_id": "请求/with:specials",
            "agentic_generation": 4,
            "agentic_parent_generation": 3,
            "agentic_tool_suffix_strings": ["🔧</tool_call>"],
            "application": "must-not-cross-the-wire",
        }
    }
    key = build_agentic_extra_key("请求/with:specials", params)
    prefix, stable_raw, payload_raw = key.split(":", 2)
    assert prefix == "agentic-v1e"
    stable = base64.urlsafe_b64decode(stable_raw + "=" * (-len(stable_raw) % 4))
    payload = json.loads(
        base64.urlsafe_b64decode(payload_raw + "=" * (-len(payload_raw) % 4))
    )
    assert stable.decode() == "agentic-v1:请求/with:specials:g4"
    assert payload["agentic_generation"] == 4
    assert payload["agentic_tool_suffix_strings"] == ["🔧</tool_call>"]
    assert "application" not in payload


def test_single_unicode_marker_is_not_split_into_characters():
    params, _ = add_agentic_kv_metadata(
        {},
        trajectory_metadata={"agentic_request_id": "数据集:\n/🔧/e\u0301"},
        generation=0,
        tokenizer=FakeTokenizer(),
        tool_type="检索/工具",
        tool_suffix_markers="🔧</tool_call>\n",
        terminal_markers="<答案>\\end",
    )
    custom = params["custom_params"]
    assert custom["agentic_tool_suffix_strings"] == ["🔧</tool_call>\n"]
    assert custom["agentic_terminal_marker_strings"] == ["<答案>\\end"]
    key = build_agentic_extra_key("数据集:\n/🔧/e\u0301", params)
    _, _, payload_raw = key.split(":", 2)
    payload = json.loads(
        base64.urlsafe_b64decode(payload_raw + "=" * (-len(payload_raw) % 4))
    )
    assert payload["agentic_request_id"] == "数据集:\n/🔧/e\u0301"
    assert payload["agentic_tool_type"] == "检索/工具"


@pytest.mark.parametrize("markers", [["ok", ""], ["ok", 3], None])
def test_invalid_dataset_markers_fail_early(markers):
    with pytest.raises((TypeError, ValueError)):
        add_agentic_kv_metadata(
            {},
            trajectory_metadata={},
            generation=0,
            tokenizer=FakeTokenizer(),
            tool_type="tool",
            tool_suffix_markers=markers,
        )
