from __future__ import annotations

import types
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    prepare_pd_decode_radix_key,
    reconcile_pd_decode_imported_prefix,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    cache_pd_decode_committed_req,
)


class _ReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.tensor([[10, 11, 12, 13, 14, 15]])

    def write(self, index, value):
        self.req_to_token[index] = value


def test_failed_uncommitted_pd_import_releases_private_pages_idempotently():
    freed = []

    class Pool:
        def __init__(self):
            self.req_to_token = torch.tensor([[10, 11, 12, 13]])

        def free(self, req):
            req.req_pool_idx = None

    pool = Pool()
    tree_cache = types.SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=types.SimpleNamespace(
            free=lambda indices: freed.append(indices.clone())
        ),
        supports_mamba=lambda: False,
    )
    req = types.SimpleNamespace(
        req_pool_idx=0,
        last_node=None,
        mamba_pool_idx=None,
        pop_committed_kv_cache=lambda: 3,
        pop_overallocated_kv_cache=lambda: (3, 4),
    )

    release_kv_cache(req, tree_cache, is_insert=False)
    release_kv_cache(req, tree_cache, is_insert=False)

    assert req.req_pool_idx is None
    assert [indices.tolist() for indices in freed] == [[10, 11, 12, 13]]


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


def test_full_pd_import_never_returns_dummy_kv_slot_to_allocator():
    freed = []
    pool = _ReqToTokenPool()
    pool.req_to_token[0] = torch.tensor([0, 0, 128, 129, 130, 131])
    scheduler = types.SimpleNamespace(
        tree_cache=types.SimpleNamespace(
            disable=False,
            inc_lock_ref=lambda _node: None,
        ),
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=types.SimpleNamespace(
            free=lambda indices: freed.append(indices.clone())
        ),
    )
    req = types.SimpleNamespace(
        rid="req-dummy",
        fill_ids=[1, 2, 3, 4, 5, 6],
        req_pool_idx=0,
        prefix_indices=torch.tensor([101, 102, 103]),
        last_node=object(),
    )

    assert reconcile_pd_decode_imported_prefix(scheduler, req) == 3
    assert freed[0].tolist() == [128]


def test_prebuilt_decode_caches_only_committed_kv_not_buffered_output_token():
    seen = []
    tree_cache = types.SimpleNamespace(
        cache_unfinished_req=lambda req: seen.append(list(req.fill_ids))
    )
    req = types.SimpleNamespace(
        fill_ids=[1, 2, 3, 4],
        kv_committed_len=3,
    )

    cache_pd_decode_committed_req(tree_cache, req)

    assert seen == [[1, 2, 3]]
    assert req.fill_ids == [1, 2, 3, 4]


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


def test_agentic_decode_release_drops_generation_instead_of_caching_history():
    releases = []
    tree_cache = types.SimpleNamespace(
        disable=False,
        release_request_generation_cache=lambda req, **kwargs: (
            releases.append((req, kwargs)) or 123
        ),
    )
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.tree_cache = tree_cache
    manager.offloaded_state = {"req-1": object()}
    req = types.SimpleNamespace(
        rid="req-1",
        req_pool_idx=7,
        kv_committed_len=640,
        extra_key="agentic-pd-decode-v1:",
        _pd_transport_extra_key="agentic-v1:req-1:g2",
    )

    target = (
        "sglang.srt.disaggregation.decode_kvcache_offload_manager."
        "release_kv_cache"
    )
    with patch(target) as release_kv:
        manager._release_finished_req(req, 0)

    release_kv.assert_called_once_with(req, tree_cache, is_insert=False)
    assert releases == [
        (
            req,
            {
                "committed_len": 640,
                "event_prefix": "d_generation_release",
                "allow_shared_ancestors": True,
            },
        )
    ]
    assert "req-1" not in manager.offloaded_state


def test_non_agentic_decode_release_preserves_native_radix_cache():
    tree_cache = types.SimpleNamespace(disable=False)
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.tree_cache = tree_cache
    manager.offloaded_state = {}
    req = types.SimpleNamespace(
        rid="ordinary",
        req_pool_idx=3,
        kv_committed_len=128,
    )

    target = (
        "sglang.srt.disaggregation.decode_kvcache_offload_manager."
        "release_kv_cache"
    )
    with patch(target) as release_kv:
        manager._release_finished_req(req, 0)

    release_kv.assert_called_once_with(req, tree_cache, is_insert=True)


def test_generation_release_keeps_shared_prefix_until_last_live_owner_leaves():
    cache = RadixCache.create_simulated(page_size=1)
    freed = []
    cache.token_to_kv_pool_allocator = types.SimpleNamespace(
        free=lambda indices: freed.extend(indices.tolist())
    )
    for tokens, indices in (
        ([1, 2, 3, 4], [10, 11, 12, 13]),
        ([1, 2, 5, 6], [20, 21, 22, 23]),
    ):
        cache.insert(
            InsertParams(
                key=RadixKey(tokens, "shared"),
                value=torch.tensor(indices),
            )
        )

    def match(tokens):
        return cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens, "shared"))
        )

    a_tokens = [1, 2, 3, 4]
    b_tokens = [1, 2, 5, 6]
    a_node = match(a_tokens).last_device_node
    b_node = match(b_tokens).last_device_node
    cache.inc_lock_ref(a_node)
    cache.inc_lock_ref(b_node)
    req_a = types.SimpleNamespace(
        extra_key="shared",
        origin_input_ids=a_tokens,
        output_ids=[],
        kv_committed_len=4,
    )
    req_b = types.SimpleNamespace(
        extra_key="shared",
        origin_input_ids=b_tokens,
        output_ids=[],
        kv_committed_len=4,
    )

    cache.dec_lock_ref(a_node)
    assert cache.release_request_generation_cache(req_a, committed_len=4) == 2
    assert len(match(a_tokens).device_indices) == 2
    assert len(match(b_tokens).device_indices) == 4
    assert cache.total_size() == 4

    cache.dec_lock_ref(b_node)
    assert cache.release_request_generation_cache(req_b, committed_len=4) == 4
    assert cache.total_size() == 0
    assert sorted(freed) == [10, 11, 12, 13, 22, 23]


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
