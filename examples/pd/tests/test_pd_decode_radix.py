from __future__ import annotations

import types

import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    prepare_pd_decode_radix_key,
    reconcile_pd_decode_imported_prefix,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)


class _ReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.tensor([[10, 11, 12, 13, 14, 15]])

    def write(self, index, value):
        self.req_to_token[index] = value


def test_full_pd_import_deduplicates_only_resident_prefix():
    freed = []
    locked = []
    scheduler = types.SimpleNamespace(
        tree_cache=types.SimpleNamespace(
            disable=False,
            inc_lock_ref=lambda node: locked.append(node),
        ),
        req_to_token_pool=_ReqToTokenPool(),
        token_to_kv_pool_allocator=types.SimpleNamespace(
            free=lambda indices: freed.append(indices.clone())
        ),
    )
    req = types.SimpleNamespace(
        rid="req-1",
        fill_ids=[1, 2, 3, 4, 5, 6],
        req_pool_idx=0,
        prefix_indices=torch.tensor([101, 102, 103]),
        last_node=object(),
    )

    assert reconcile_pd_decode_imported_prefix(scheduler, req) == 3
    assert scheduler.req_to_token_pool.req_to_token[0].tolist() == [
        101,
        102,
        103,
        13,
        14,
        15,
    ]
    assert freed[0].tolist() == [10, 11, 12]
    assert locked == [req.last_node]


def test_chunk_cache_keeps_complete_import_private():
    scheduler = types.SimpleNamespace(
        tree_cache=types.SimpleNamespace(disable=True),
        req_to_token_pool=_ReqToTokenPool(),
    )
    req = types.SimpleNamespace(prefix_indices=torch.tensor([101, 102]))

    assert reconcile_pd_decode_imported_prefix(scheduler, req) == 0
    assert scheduler.req_to_token_pool.req_to_token[0].tolist() == [
        10,
        11,
        12,
        13,
        14,
        15,
    ]


def test_agentic_wire_key_is_normalized_only_inside_decode_radix():
    scheduler = types.SimpleNamespace(tree_cache=types.SimpleNamespace(disable=False))
    req = types.SimpleNamespace(
        extra_key="agentic-v1:req-7:g3",
        lora_id=None,
        sampling_params=types.SimpleNamespace(
            custom_params={"agentic_request_id": "req-7"}
        ),
    )

    prepare_pd_decode_radix_key(scheduler, req)

    assert req._pd_transport_extra_key == "agentic-v1:req-7:g3"
    assert req.extra_key == "agentic-pd-decode-v1:"


def test_non_agentic_cache_salt_is_not_normalized():
    scheduler = types.SimpleNamespace(tree_cache=types.SimpleNamespace(disable=False))
    req = types.SimpleNamespace(
        extra_key="tenant-cache-salt",
        lora_id=None,
        sampling_params=types.SimpleNamespace(custom_params={}),
    )

    prepare_pd_decode_radix_key(scheduler, req)

    assert req.extra_key == "tenant-cache-salt"
    assert not hasattr(req, "_pd_transport_extra_key")


def test_radix_terminal_release_is_one_complete_lifecycle_commit():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.tree_cache = types.SimpleNamespace(disable=False)
    calls = []
    manager._release_finished_req = lambda req, start: calls.append((req, start))
    req = types.SimpleNamespace(req_pool_idx=7)

    manager.finalize_release_on_finish(req)

    assert calls == [(req, 0)]


def test_pd_prealloc_reclaims_only_radix_deficit():
    evictions = []
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.token_to_kv_pool_allocator = types.SimpleNamespace(
        available_size=lambda: 1_000
    )
    queue.tree_cache = types.SimpleNamespace(
        is_chunk_cache=lambda: False,
        evict=lambda params: (
            evictions.append(params.num_tokens)
            or types.SimpleNamespace(num_tokens_evicted=params.num_tokens)
        ),
    )

    assert queue._ensure_prealloc_kv_available(1_600) == 600
    assert evictions == [600]


def test_pd_prealloc_does_not_evict_when_free_pages_are_sufficient():
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.token_to_kv_pool_allocator = types.SimpleNamespace(
        available_size=lambda: 2_000
    )
    queue.tree_cache = types.SimpleNamespace(
        is_chunk_cache=lambda: False,
        evict=lambda _params: (_ for _ in ()).throw(AssertionError("unexpected evict")),
    )

    assert queue._ensure_prealloc_kv_available(1_600) == 0
