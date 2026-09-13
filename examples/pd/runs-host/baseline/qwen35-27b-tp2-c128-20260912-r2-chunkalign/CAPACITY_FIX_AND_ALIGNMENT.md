# Capacity correction and proposed paired experiment

2026-09-12. No GPU experiment launched for this correction. R2 ended with
47/500 task records after model0 Prefill allocation failed (4960 input tokens,
3392 available tokens, zero evictable tokens). It is not a completed baseline.

## Installed baseline correction

`pd_mamba_baseline` only: `managers/schedule_policy.py::add_chunked_req` now
caps resumed SSM chunks by both `rem_total_tokens` and `cur_rem_tokens`,
subtracts a page of allocation overhead, and page-floors the input budget.
It parks before the legacy full-chunk fallback on exhausted capacity.
Previous Mamba chunk alignment and parked-batch accounting fix remains.
No KV/state contents, prefix insertion assertions, harness or PD code changed.

Physical allocation charge is `ceil(input/page)*page + page`; the bound is
applied to final chunks too. Example total100/physical100 cannot admit a
35-token final tail with page64; budget128 can. Exact old4960/3392 unsafe
fallback is reproduced on original code and blocked by the fixed code.
Future Decode reservations and normal retraction are not eliminated.

Validation:87 tests passed (78 admission/alignment,9 launcher/report).
Independent `audit_baseline_mamba_alignment` reran78 and returned GO for code.
Ownership remains on the request; pending/inflight/abort bookkeeping preserved;
no blocking I/O, new timeout, new transfer or collective. PD ownership criteria
are unchanged/not exercised. Full500 GPU correctness remains unvalidated.
Patch follows the earlier alignment patch:
`patches/sglang_0_5_14_mamba_resumed_chunk_capacity.patch`.

## Actual recent settings

PD record:
`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c256-c500-tool1s-20260912-r4/c256`.

Both recent PD and colocated use Triton Attention, deterministic inference,
PyTorch sampling, disabled custom all-reduce, page64/track64, extra_buffer,
static0.80, Mamba/full ratio0.9, chunk8192. PD has NUMA `[0,1]` binding;
colocated omitted explicit binding. PD P max-prefill8192; D16384; colocated8192.
PD requestedc256; latest colocatedc128: concurrency must also match for a pair.

Runtime discrepancy: baseline Torch2.11.0+cu128/Triton3.6.0 vs PD Torch2.9.1/
Triton3.5.1. FlashInfer packages both0.6.7.post3. PD imports
`sglang-qwen35-integration/python`, not installed sgLang sources: installed
metadata0.5.10.post1 does NOT identify the overlay source version. Current
overlay HEAD2fa0c98124 describes as v0.5.14-61-g2fa0c98124. Pin actual source
commit plus diff and dependency versions for any comparison.

## Proposed next paired performance setting (NOT applied)

Preserve page64/track64 and extra_buffer for reverse-compatible Mamba state.
Use the same upstream source base and necessary correctness fixes; baseline
has no agentic transport, PD alone adds the tested reverse-KV/state mechanism.
Unify Torch/CUDA/Triton/FlashInfer dependencies in isolated environments,
without changing another agent's shared PD install.

For serving-throughput evaluation, jointly use FlashInfer Attention/sampling,
deterministic mode OFF, allow native custom all-reduce, matching NUMA binding,
overlap ON, seed2026, chunk/max-prefill8192, static0.80, Mamba ratio0.9.
Verify actual effective backend and Qwen3.5 Mamba return correctness with a
small paired smoke before full500; do not assert compatibility/performance
merely from flags. Fixed sampling seed does not imply batch-invariant outputs.
If this backend combination fails compatibility validation, both arms must
use the same verified fallback, not FlashInfer in one and Triton in the other.

First pair c128: colocated4TP2 groups vs PD4P:4D (2P groups +2D groups), eight
A100s in both. Same GPU groups0,4;1,5;2,6;3,7, same500 tasks and order, same
OpenEnv structured tools and8K/64 turns, same sampling/context/tool/verifier
settings. Later c256 must be paired on both arms. Do not compare historical
page1/0.5.10 baseline with this new pair as a controlled experiment.

PD-only treatment: request-owned stable-prefix KV+Mamba return, fast-tool1s,
Direct handshake1s, failed Direct to Shared Host, native HiCache/Mooncake OFF,
content hashes OFF, complete Host registration before workload. Host capacity,
transfer traffic, and pool use must be reported as additional system resources.
These are method differences; do not enable the PD transport in colocated
merely to make settings appear identical.
