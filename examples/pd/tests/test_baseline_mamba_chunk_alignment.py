"""CPU regressions execute installed engine methods without starting CUDA.

Only resumed Mamba admission changes. Cache insertion and GPU state copies
are intentionally untouched. BASELINE_ENGINE_ROOT can select another install.
"""
import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(os.environ.get('BASELINE_ENGINE_ROOT',
    '/homes/siqic/anaconda3/envs/pd_mamba_baseline/lib/python3.12/site-packages/sglang/srt'))
ORIGINAL = Path('/tmp/pd-persist/baseline-mamba-chunk-alignment-fix-20260912/original')


def method(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    source = 'from __future__ import annotations\n' + ast.unparse(node)
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace[name]


def settings(chunk=64):
    return NS(mamba_cache_chunk_size=chunk, enable_mamba_extra_buffer_lazy=lambda: False)


def admission(prefix, total, budget, *, page=64, chunk=64, hybrid=True,
              original=False, physical_budget=None):
    req = NS(prefix_indices=range(prefix), extend_input_len=total-prefix,
             fill_len=prefix, sampling_params=NS(max_new_tokens=8192), retracted_stain=False)
    req.set_extend_input_len = lambda value: setattr(req, 'extend_input_len', value)
    updates = []
    adder = NS(dllm_config=None, is_hybrid_swa=False, is_hybrid_ssm_cache=hybrid,
               rem_chunk_tokens=8192, rem_total_tokens=budget, page_size=page,
               cur_rem_tokens=budget if physical_budget is None else physical_budget,
               can_run_list=[], _update_prefill_budget=lambda *args: updates.append(args))
    path = (ORIGINAL/'schedule_policy.py' if original else ROOT/'managers/schedule_policy.py')
    add = method(path, 'add_chunked_req', dict(math=math,
                 get_global_server_args=lambda: settings(chunk), CLIP_MAX_NEW_TOKENS=4096))
    pending = add(adder, req)
    return req, adder, pending, updates


def track(req, chunk=64):
    req.mamba_ping_pong_track_buffer = [NS(item=lambda: 0), NS(item=lambda: 1)]
    req.mamba_next_track_idx = 0
    req.mamba_branching_seqlen = None
    req.mamba_last_track_seqlen = None
    batch = NS(req_to_token_pool=NS(get_mamba_ping_pong_other_idx=lambda i: 1-i))
    fn = method(ROOT/'managers/schedule_batch.py', '_mamba_radix_cache_v2_req_prepare_for_extend',
                dict(get_global_server_args=lambda: settings(chunk), _MambaRadixCacheV2TrackEntry=NS))
    fn(batch, req)
    return req.mamba_last_track_seqlen


def test_old_two_chunk_sequence_reproduces_exact_21412_failure():
    if not ORIGINAL.exists():
        pytest.skip('Original engine source archived with this run')
    first, _, _, _ = admission(8192, 21448, 7972, original=True)
    assert first.fill_len == 16164  # Middle chunk leaves a 36-token tail.
    final, _, _, _ = admission(first.fill_len, 21448, 8192, original=True)
    assert track(final) == 21412
    assert track(final) // 64 * 64 == 21376


def test_fixed_sequence_tracks_real_aligned_state_and_keeps_final_tail():
    first, _, pending, _ = admission(8192, 21448, 7972)
    assert pending is first and first.fill_len == 16064
    assert track(first) == first.fill_len
    final, _, pending, _ = admission(first.fill_len, 21448, 8192)
    assert pending is None and final.fill_len == 21448
    assert track(final) == 21440  # Eight final tokens remain uncached, not deleted.


@pytest.mark.parametrize('budget', [1, 36, 63, 64, 65, 127, 4132, 7972, 8192])
@pytest.mark.parametrize('page,chunk', [(64,64), (128,64), (64,128), (1,64)])
def test_capacity_alignment_and_park_are_non_mutating(budget, page, chunk):
    req, adder, pending, updates = admission(8192, 30000, budget, page=page, chunk=chunk)
    alignment = math.lcm(page, chunk)
    assert pending is req
    if budget - page < alignment:
        assert adder.can_run_list == [] and updates == []
        assert req.fill_len == 8192 and req.extend_input_len == 21808
    else:
        assert adder.can_run_list == [req]
        assert 0 < req.extend_input_len <= budget
        assert math.ceil(req.extend_input_len / page) * page + page <= budget
        assert req.fill_len % alignment == 0
        assert track(req, chunk) % page == 0


@pytest.mark.parametrize('tail', [1, 36, 63, 64, 65])
def test_final_tail_not_rounded_or_dropped(tail):
    req, adder, pending, _ = admission(8192, 8192+tail, math.ceil(tail/64)*64+64)
    assert pending is None and req.fill_len == 8192+tail
    assert adder.can_run_list == [req]


def test_attention_only_path_unchanged():
    req, _, pending, _ = admission(8192, 30000, 7972, hybrid=False)
    assert pending is req and req.extend_input_len == 7972


@pytest.mark.parametrize('admitted', [False, True])
def test_parked_chunk_does_not_get_phantom_forward_or_abort_debt(admitted):
    tree = ast.parse((ROOT/'managers/scheduler.py').read_text())
    assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'batch_chunked_req' for t in n.targets))
    guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'batch_chunked_req is not None')
    class Req:
        inflight_middle_chunks = 0
    req, other = Req(), Req()
    scope = dict(self=NS(chunked_req=req), can_run_set={req} if admitted else {other})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assign,guard], type_ignores=[])),
                 '<real scheduler admission>', 'exec'), scope)
    assert req.inflight_middle_chunks == int(admitted)
    assert (scope['batch_chunked_req'] is req) == admitted
    if not admitted:
        assert req.inflight_middle_chunks == 0  # Cancellation need not await a nonexistent result.
    source = (ROOT/'managers/scheduler.py').read_text()
    assert 'chunked_req=batch_chunked_req' in source
    assert 'batch_chunked_req.extend_input_len' in source


def test_tp_ranks_choose_identical_chunks():
    for budget in [36, 64, 4132, 7972]:
        a, aa, _, _ = admission(8192, 30000, budget)
        b, bb, _, _ = admission(8192, 30000, budget)
        assert (a.fill_len, len(aa.can_run_list)) == (b.fill_len, len(bb.can_run_list))


@pytest.mark.parametrize('budget,physical', [(-704,3392), (0,3392), (8192,0),
                                          (8192,63), (100,100), (8192,100)])
def test_no_full_chunk_fallback_when_budget_exhausted(budget, physical):
    req, adder, pending, updates = admission(16384, 21344, budget,
                                           physical_budget=physical)
    assert pending is req and not adder.can_run_list and not updates
    assert req.fill_len == 16384 and req.extend_input_len == 4960


def test_original_reproduces_4960_admission_despite_3392_available():
    if not ORIGINAL.exists():
        pytest.skip('Original source unavailable')
    req, adder, pending, _ = admission(16384, 21344, -704,
                                      physical_budget=3392, original=True)
    assert adder.can_run_list == [req] and pending is None
    assert req.extend_input_len == 4960 > adder.cur_rem_tokens


@pytest.mark.parametrize('budget,admitted', [(100,False), (127,False), (128,True)])
def test_final_subpage_tail_reserves_rounding_and_overhead(budget, admitted):
    req, adder, pending, _ = admission(8192, 8227, budget)
    assert bool(adder.can_run_list) == admitted
    assert req.extend_input_len == 35
    assert req.fill_len == (8227 if admitted else 8192)


def test_parked_request_resumes_when_capacity_returns():
    req, adder, pending, updates = admission(8192, 13152, -704,
                                           physical_budget=3392)
    assert pending is req and not adder.can_run_list
    # Same request and owner, not a replacement request or lost cache prefix.
    adder.rem_total_tokens = adder.cur_rem_tokens = 8192
    add = method(ROOT/'managers/schedule_policy.py', 'add_chunked_req',
                 dict(math=math, get_global_server_args=settings, CLIP_MAX_NEW_TOKENS=4096))
    assert add(adder, req) is None
    assert adder.can_run_list == [req] and req.fill_len == 13152
    assert len(updates) == 1


@pytest.mark.parametrize('total', [128, 512, 4096, 8192])
@pytest.mark.parametrize('physical', [100, 128, 256, 3392, 8192])
def test_admission_charge_never_exceeds_either_budget(total, physical):
    req, adder, _, _ = admission(8192, 13152, total, physical_budget=physical)
    if adder.can_run_list:
        charge = math.ceil(req.extend_input_len/64)*64+64
        assert charge <= min(total, physical)
