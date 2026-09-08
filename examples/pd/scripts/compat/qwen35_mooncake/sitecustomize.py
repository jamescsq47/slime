"""Run-local SGLang 0.5.10 compatibility for hybrid Mamba Mooncake HiCache.

This module is loaded only when its directory is explicitly prepended to
PYTHONPATH.  It intentionally avoids modifying the pd_baseline environment.
The compatibility methods are a narrow backport of the upstream SGLang
hybrid-pool Mooncake implementation for PoolName.MAMBA.
"""

from __future__ import annotations

import logging
import json
import threading
import time
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    MambaPoolHost,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    PoolEntry,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeBaseStore,
    MooncakeStore,
)


logger = logging.getLogger(__name__)


def _mamba_get_hybrid_pool_buffer(self: MambaPoolHost):
    return [self.temporal_buffer, *self.conv_buffer]


def _mamba_get_page_buffer_meta(self: MambaPoolHost, indices):
    """Return zero-copy Mooncake pointers for page-first Mamba state."""
    assert len(indices) % self.page_size == 0
    if self.layout not in ("page_first", "page_first_direct"):
        raise ValueError(
            f"Mamba Mooncake zero-copy requires page_first layout, got {self.layout}"
        )

    temporal_elems = int(np.prod(self.temporal_state_shape))
    conv_elems = [int(np.prod(shape)) for shape in self.conv_state_shapes]
    temporal_bytes = (
        self.page_size
        * self.num_mamba_layers
        * self.temporal_dtype.itemsize
        * temporal_elems
    )
    conv_bytes = [
        self.page_size
        * self.num_mamba_layers
        * self.conv_dtype.itemsize
        * elems
        for elems in conv_elems
    ]
    temporal_base = self.temporal_buffer.data_ptr()
    conv_bases = [buffer.data_ptr() for buffer in self.conv_buffer]

    ptrs = []
    sizes = []
    for index in indices.tolist()[:: self.page_size]:
        if temporal_elems > 0:
            ptrs.append(
                temporal_base
                + index
                * self.num_mamba_layers
                * temporal_elems
                * self.temporal_dtype.itemsize
            )
            sizes.append(temporal_bytes)
        for component, elems in enumerate(conv_elems):
            ptrs.append(
                conv_bases[component]
                + index
                * self.num_mamba_layers
                * elems
                * self.conv_dtype.itemsize
            )
            sizes.append(conv_bytes[component])
    return ptrs, sizes


_original_mamba_init = MambaPoolHost.__init__
_original_copy_tensor_all_layers_lf_pf = (
    MambaPoolHost._copy_tensor_all_layers_lf_pf
)
_original_copy_tensor_pf_lf = MambaPoolHost._copy_tensor_pf_lf


def _mamba_init_with_component_sizes(self: MambaPoolHost, *args, **kwargs):
    _original_mamba_init(self, *args, **kwargs)
    self.temporal_state_elem_size = int(np.prod(self.temporal_state_shape))
    self.conv_state_elem_sizes = [
        int(np.prod(shape)) for shape in self.conv_state_shapes
    ]


def _copy_tensor_all_layers_lf_pf_with_cuda_dst_indices(
    src_layers,
    dst,
    src_indices,
    dst_indices,
    num_layers,
    device,
    io_backend,
):
    """Backport the hybrid-cache kernel's CUDA destination-index fix.

    SGLang 0.5.10 keeps host-pool indices on CPU, while the fused
    layer-first -> page-first Mamba copy kernel requires both index tensors on
    CUDA.  Upstream now performs this asynchronous device copy immediately
    before launching the kernel.
    """
    if io_backend == "kernel" and dst_indices.device.type != "cuda":
        dst_indices = dst_indices.to(src_indices.device, non_blocking=True)
    return _original_copy_tensor_all_layers_lf_pf(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        num_layers,
        device,
        io_backend,
    )


def _copy_tensor_pf_lf_with_cuda_src_indices(
    src,
    dst,
    src_indices,
    dst_indices,
    layer_id,
    num_layers,
    io_backend,
):
    """Backport the symmetric Host->GPU Mamba index-device fix."""
    if io_backend == "kernel" and src_indices.device.type != "cuda":
        src_indices = src_indices.to(dst_indices.device, non_blocking=True)
    return _original_copy_tensor_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        num_layers,
        io_backend,
    )


_original_register_mem_pool_host = MooncakeStore.register_mem_pool_host


def _register_mem_pool_host(self: MooncakeStore, mem_pool_host):
    if isinstance(mem_pool_host, HostPoolGroup):
        # The legacy v1 Mooncake path remains responsible for the KV anchor.
        mem_pool_host = mem_pool_host.anchor_entry.host_pool
    return _original_register_mem_pool_host(self, mem_pool_host)


def _register_mem_host_pool_v2(
    self: MooncakeStore, host_pool, host_pool_name
):
    if host_pool_name == PoolName.KV:
        return
    if host_pool_name != PoolName.MAMBA:
        raise ValueError(
            f"Qwen3.5 compatibility only supports the Mamba side pool, got {host_pool_name}"
        )
    if not hasattr(self, "registered_pools"):
        self.registered_pools = {}
    self.registered_pools[host_pool_name] = host_pool
    for buffer in host_pool.get_hybrid_pool_buffer():
        if buffer.numel() > 0:
            MooncakeBaseStore.register_buffer(self, buffer)


def _mamba_component_keys(self: MooncakeStore, page_keys: List[str], transfer):
    host_pool = self.registered_pools[transfer.name]
    suffixes = []
    if host_pool.temporal_state_elem_size > 0:
        suffixes.append(f"_{self.mha_suffix}_temporal")
    suffixes.extend(
        f"_{self.mha_suffix}_conv_{index}"
        for index in range(len(host_pool.conv_buffer))
    )
    return [f"{key}{suffix}" for key in page_keys for suffix in suffixes], len(
        suffixes
    )


def _batch_exists_v2(
    self: MooncakeStore,
    keys: List[str],
    pool_transfers: Optional[List[PoolTransfer]] = None,
    extra_info: Optional[HiCacheStorageExtraInfo] = None,
) -> PoolTransferResult:
    kv_pages = self.batch_exists(keys, extra_info)
    hit_count = {PoolName.KV: kv_pages} if kv_pages else {}
    final_pages = kv_pages

    for transfer in pool_transfers or []:
        if final_pages == 0:
            break
        if transfer.name != PoolName.MAMBA:
            raise ValueError(f"Unsupported hybrid pool: {transfer.name}")
        component_keys, multiplier = _mamba_component_keys(self, keys, transfer)
        exists = self._batch_exist(component_keys)
        page_exists = [
            all(
                value == 1
                for value in exists[index * multiplier : (index + 1) * multiplier]
            )
            for index in range(kv_pages)
        ]
        boundary = 0
        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            try:
                boundary = page_exists.index(False)
            except ValueError:
                boundary = kv_pages
        elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
            trailing = max(1, len(transfer.keys) if transfer.keys else 1)
            for prefix_len in range(kv_pages, 0, -1):
                if all(
                    page_exists[index]
                    for index in range(max(0, prefix_len - trailing), prefix_len)
                ):
                    boundary = prefix_len
                    break
        if boundary:
            hit_count[transfer.name] = boundary
        final_pages = min(final_pages, boundary)
    return PoolTransferResult(final_pages, hit_count)


def _batch_io_v2(self: MooncakeStore, transfers: List[PoolTransfer], is_set: bool):
    results = {}
    for transfer in transfers:
        if transfer.name != PoolName.MAMBA:
            raise ValueError(f"Unsupported hybrid pool: {transfer.name}")
        host_pool = self.registered_pools[transfer.name]
        keys = transfer.keys or []
        host_indices = transfer.host_indices
        assert keys
        assert len(keys) == len(host_indices) // host_pool.page_size
        component_keys, multiplier = _mamba_component_keys(self, keys, transfer)
        ptrs, sizes = host_pool.get_page_buffer_meta(host_indices)
        if is_set:
            exists = self._batch_exist(component_keys)
            io_results = [0 if value == 1 else -1 for value in exists]
            missing = [index for index, value in enumerate(exists) if value != 1]
            if missing:
                put_results = self._put_batch_zero_copy_impl(
                    [component_keys[index] for index in missing],
                    [ptrs[index] for index in missing],
                    [sizes[index] for index in missing],
                )
                for index, value in zip(missing, put_results):
                    io_results[index] = value
        else:
            io_results = self._get_batch_zero_copy_impl(component_keys, ptrs, sizes)
        # The legacy helper assumes ordinary MHA K/V pairs.  Regroup using the
        # actual number of Mamba components (temporal + conv buffers) instead.
        results[transfer.name] = [
            (
                all(value == 0 for value in group)
                if is_set
                else all(value > 0 for value in group)
            )
            for group in (
                io_results[index : index + multiplier]
                for index in range(0, len(io_results), multiplier)
            )
        ]
    return results


def _batch_get_v2(self: MooncakeStore, transfers, extra_info=None):
    return _batch_io_v2(self, transfers, is_set=False)


def _batch_set_v2(self: MooncakeStore, transfers, extra_info=None):
    return _batch_io_v2(self, transfers, is_set=True)


# Decode-side offload in SGLang 0.5.10 only accepts a plain MHA/MLA pool.
# Qwen3.5 exposes HybridLinearKVPool, so build the same paired KV + Mamba host
# pools used by HiMambaRadixCache and archive both under the same final hash.
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (  # noqa: E402
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.kv_events import OffloadedState  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402


_original_decode_offload_init = DecodeKVCacheOffloadManager.__init__
_original_decode_offload = DecodeKVCacheOffloadManager.offload_kv_cache
_original_decode_check_offload = DecodeKVCacheOffloadManager._check_offload_progress
_original_decode_check_backup = DecodeKVCacheOffloadManager._check_backup_progress
_original_decode_release = DecodeKVCacheOffloadManager._release_finished_req


def _decode_offload_init_hybrid(
    self,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    tp_group,
    tree_cache,
    server_args,
):
    hybrid_kv = token_to_kv_pool_allocator.get_kvcache()
    if not isinstance(hybrid_kv, HybridLinearKVPool):
        return _original_decode_offload_init(
            self,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            tp_group,
            tree_cache,
            server_args,
        )
    if not isinstance(req_to_token_pool, HybridReqToTokenPool):
        raise ValueError("Hybrid decode offload requires HybridReqToTokenPool")

    self.req_to_token_pool = req_to_token_pool
    self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    self.page_size = server_args.page_size
    self.server_args = server_args
    self.request_counter = 0
    self.tree_cache = tree_cache
    env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
    self.offload_stride = (
        self.page_size
        if env_stride is None or env_stride <= 0
        else max(self.page_size, (env_stride // self.page_size) * self.page_size)
    )

    full_kv = hybrid_kv.full_kv_pool
    kv_host_cls = MLATokenToKVPoolHost if hybrid_kv.use_mla else MHATokenToKVPoolHost
    full_host = kv_host_cls(
        full_kv,
        server_args.hicache_ratio,
        server_args.hicache_size,
        self.page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )
    mamba_host = MambaPoolHost(
        req_to_token_pool.mamba_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        allocator_type=server_args.hicache_storage_backend,
        layout=server_args.hicache_mem_layout,
    )
    full_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
    mamba_mapping = dict(req_to_token_pool.mamba_map)
    transfer_layer_num = len(set(full_mapping) | set(mamba_mapping))

    def kv_layer_mapper(layer_id):
        return full_mapping.get(layer_id) if 0 <= layer_id < transfer_layer_num else None

    def mamba_layer_mapper(layer_id):
        return mamba_mapping.get(layer_id) if 0 <= layer_id < transfer_layer_num else None

    host_group = HostPoolGroup(
        [
            PoolEntry(
                name=PoolName.KV,
                host_pool=full_host,
                device_pool=full_kv,
                layer_mapper=kv_layer_mapper,
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.MAMBA,
                host_pool=mamba_host,
                device_pool=req_to_token_pool.mamba_pool,
                layer_mapper=mamba_layer_mapper,
            ),
        ]
    )
    self.decode_host_mem_pool = host_group
    self._hybrid_mamba_host_pool = mamba_host
    self._qwen35_hybrid_decode_offload = True
    self.tp_group = tp_group
    self.tp_world_size = torch.distributed.get_world_size(group=tp_group)

    extra_config = {}
    if server_args.hicache_storage_backend_extra_config:
        extra_config = json.loads(server_args.hicache_storage_backend_extra_config)
    self.cache_controller = HybridCacheController(
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        mem_pool_host=host_group,
        page_size=self.page_size,
        tp_group=tp_group,
        io_backend=server_args.hicache_io_backend,
        load_cache_event=threading.Event(),
        storage_backend=server_args.hicache_storage_backend,
        model_name=server_args.served_model_name,
        storage_backend_extra_config=extra_config,
        transfer_layer_num=transfer_layer_num,
    )
    req_to_token_pool.register_layer_transfer_counter(
        self.cache_controller.layer_done_counter
    )
    hybrid_kv.register_layer_transfer_counter(self.cache_controller.layer_done_counter)
    self.ongoing_offload = {}
    self.ongoing_backup = {}
    self.offloaded_state = {}
    self.offload_inflight = {}
    self._hybrid_mamba_by_ack = {}
    logger.info(
        "Enabled run-local hybrid decode offload: stride=%s, KV+Mamba storage",
        self.offload_stride,
    )


def _decode_offload_hybrid(self, req):
    if not getattr(self, "_qwen35_hybrid_decode_offload", False):
        return _original_decode_offload(self, req)
    if req.req_pool_idx == -1 or req.req_pool_idx is None or not req.output_ids:
        return False
    if req.mamba_pool_idx is None:
        return False
    token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
    if token_indices.dim() == 0 or token_indices.numel() == 0:
        return False

    all_tokens = req.origin_input_ids + req.output_ids[:-1]
    prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
    state = self.offloaded_state.get(req.rid)
    if state is None:
        hashes = self._compute_prefix_hash(req.origin_input_ids[:prefill_len])
        state = OffloadedState(
            prefill_len=prefill_len,
            inc_len=0,
            last_hash=hashes[-1] if hashes else None,
        )
        self.offloaded_state[req.rid] = state
    new_count = len(all_tokens) - state.prefill_len - state.inc_len
    aligned = new_count // self.offload_stride * self.offload_stride
    if aligned == 0:
        return False

    start = state.prefill_len + state.inc_len
    end = start + aligned
    incremental_tokens = all_tokens[start:end]
    incremental_indices = token_indices[start:end]
    self.request_counter += 1
    ack_id = self.request_counter
    mamba_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        device_indices=req.mamba_pool_idx.unsqueeze(0),
    )
    host_indices = self.cache_controller.write(
        device_indices=incremental_indices.long(),
        node_id=ack_id,
        extra_pools=[mamba_transfer],
    )
    if host_indices is None:
        logger.warning("Not enough hybrid host memory for request %s", req.rid)
        return False

    self._mark_offload_started(req.rid)
    self.ongoing_offload[ack_id] = (
        req,
        host_indices,
        incremental_tokens,
        time.time(),
        start,
        end,
    )
    self._hybrid_mamba_by_ack[ack_id] = mamba_transfer
    state.inc_len += aligned
    return True


def _decode_check_offload_hybrid(self, finish_count):
    if not getattr(self, "_qwen35_hybrid_decode_offload", False):
        return _original_decode_check_offload(self, finish_count)
    while finish_count > 0:
        ack = self.cache_controller.ack_write_queue.pop(0)
        ack.finish_event.synchronize()
        for ack_id in ack.node_ids:
            req, host_indices, tokens, started, start, _end = self.ongoing_offload.pop(
                ack_id
            )
            mamba_transfer = self._hybrid_mamba_by_ack.pop(ack_id)
            self._mark_offload_finished(req.rid)
            state = self.offloaded_state.get(req.rid)
            prior_hash = state.last_hash if state is not None else None
            page_hashes = self._compute_prefix_hash(tokens, prior_hash)
            mamba_transfer.keys = [page_hashes[-1]]
            mamba_transfer.hit_policy = PoolHitPolicy.TRAILING_PAGES
            backup_id = self.cache_controller.write_storage(
                host_indices,
                tokens,
                hash_value=page_hashes,
                extra_pools=[mamba_transfer],
            )
            self.ongoing_backup[backup_id] = (
                req.rid,
                host_indices,
                started,
                mamba_transfer.host_indices,
            )
            if state is not None:
                state.last_hash = page_hashes[-1]
            if req.finished() and not self._has_inflight_offload(req.rid):
                self._release_finished_req(req, state.prefill_len if state else start)
        finish_count -= 1


def _decode_check_backup_hybrid(self, finish_count):
    if not getattr(self, "_qwen35_hybrid_decode_offload", False):
        return _original_decode_check_backup(self, finish_count)
    for _ in range(finish_count):
        op = self.cache_controller.ack_backup_queue.get()
        _rid, host_indices, _started, mamba_host_indices = self.ongoing_backup.pop(
            op.id
        )
        self.decode_host_mem_pool.free(host_indices)
        if mamba_host_indices is not None:
            self._hybrid_mamba_host_pool.free(mamba_host_indices)


def _decode_release_hybrid(self, req, start_offset):
    if (
        getattr(self, "_qwen35_hybrid_decode_offload", False)
        and req.req_pool_idx is not None
        and req.req_pool_idx != -1
        and req.mamba_pool_idx is not None
    ):
        self.req_to_token_pool.free_mamba_cache(req)
    return _original_decode_release(self, req, start_offset)


MambaPoolHost.__init__ = _mamba_init_with_component_sizes
MambaPoolHost._copy_tensor_all_layers_lf_pf = staticmethod(
    _copy_tensor_all_layers_lf_pf_with_cuda_dst_indices
)
MambaPoolHost._copy_tensor_pf_lf = staticmethod(
    _copy_tensor_pf_lf_with_cuda_src_indices
)
MambaPoolHost.get_hybrid_pool_buffer = _mamba_get_hybrid_pool_buffer
MambaPoolHost.get_page_buffer_meta = _mamba_get_page_buffer_meta
MooncakeStore.register_mem_pool_host = _register_mem_pool_host
MooncakeStore.register_mem_host_pool_v2 = _register_mem_host_pool_v2
MooncakeStore.batch_exists_v2 = _batch_exists_v2
MooncakeStore.batch_get_v2 = _batch_get_v2
MooncakeStore.batch_set_v2 = _batch_set_v2
DecodeKVCacheOffloadManager.__init__ = _decode_offload_init_hybrid
DecodeKVCacheOffloadManager.offload_kv_cache = _decode_offload_hybrid
DecodeKVCacheOffloadManager._check_offload_progress = _decode_check_offload_hybrid
DecodeKVCacheOffloadManager._check_backup_progress = _decode_check_backup_hybrid
DecodeKVCacheOffloadManager._release_finished_req = _decode_release_hybrid

logger.warning("Loaded run-local Qwen3.5 hybrid Mooncake compatibility layer")
