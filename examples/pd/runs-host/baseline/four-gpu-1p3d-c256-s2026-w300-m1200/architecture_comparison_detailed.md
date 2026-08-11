# Four-GPU mixed-agent architecture comparison

> **Correctness warning:** the native Mooncake Decode-offload case is invalid as
> a serving-performance baseline in the installed SGLang 0.5.10.post1. Its
> Decode offload manager frees page-aligned KV for requests that are still
> decoding. Under allocator churn, those slots are reused while the old
> requests still reference them. A deterministic reduced-capacity test retained
> only 64 accounted KV tokens per active anchor and produced token divergence in
> 8 of 12 anchors. Therefore its 1,574 Decode token/s/GPU and 0.999 Agent/s must
> not be compared with the two correctness-preserving cases.

All three cases use the same seed-2026 randomized Mixed 1:1 dispatch sequence,
256 closed-loop concurrent agents, 300 seconds of warmup, and a 1200-second
measurement window. `sglang_token_usage` below is logical KV allocator-slot
occupancy, not GPU compute utilization and not physical HBM allocation.

## Throughput and compute

| Metric | 4x colocated | 1P:3D, no reverse KV | 1P:3D, native Mooncake |
|---|---:|---:|---:|
| Completed agents | 1,465 | 471 | 1,199 |
| Completed agents/s | 1.221 | 0.392 | 0.999 |
| Retool / BrowseComp completed | 734 / 731 | 207 / 264 | 600 / 599 |
| Prefill compute token/s | 10,068 | 11,548 | 3,430 |
| Prefill cache-hit token/s | 30,819 | 0 | 8,349 |
| Prefill active token/s | 9,484 | 11,552 | 8,709 |
| Actual Prefill tokens/completed agent | 8,232 | 29,398 | 3,427 |
| Decode token/s, total | 4,826 | 1,729 | 4,721 |
| Decode token/s/GPU | 1,207 | 576 | 1,574 |
| Decode active token/s/GPU | 1,644 | 581 | 1,575 |
| Decode tokens/completed agent | 3,946 | 4,402 | 4,717 |
| P busy | 106.2% aggregate (26.5%/GPU) | 100.0% | 39.4% |
| D busy/GPU | 73.4% | 99.2% | 99.9% |
| Prompt prefix hit | 92.1% | 0% | 86.9% |
| Page-aligned reverse KV reuse | 52.3% | 0% | 53.8% |

## Scheduler and KV state

Queue values are time averages over the formal measurement window. Per-D values
are averages over all Decode engines. Queue categories are engine-local states
and should not be summed as unique requests because a request can be represented
at multiple pipeline stages while a transfer is in flight.

| Metric | 4x colocated | 1P:3D, no reverse KV | 1P:3D, native Mooncake |
|---|---:|---:|---:|
| Shared/P allocator-slot occupancy, mean / P95 / max | 68.6% / 90.1% / 95.9% per GPU | 7.2% / 12.2% / 15.2% | 2.8% / 11.6% / 20.8% |
| D allocator-slot occupancy, mean / P95 / max | 68.6% / 89.9% / 95.9% per GPU | 91.1% / 93.9% / 95.6% | 4.7% / 16.2% / 30.6% |
| D running/GPU, mean / P95 / max | 63.2 / 69 / 73 | 8.9 / 19 / 28 | 84.6 / 106 / 116 |
| D running total, mean / P95 | 252.7 / 255 | 26.8 / 49 | 253.9 / 257 |
| D prealloc/GPU, mean / P95 / max | 0 / 0 / 0 | 34.6 / 48 / 56 | 0 / 0 / 0 |
| D transfer/GPU, mean / P95 / max | 0 / 0 / 0 | 41.6 / 52 / 64 | 0.63 / 3 / 16 |
| P queue, mean / P95 / max | 0.06 / 0 / 10 per GPU | 121.2 / 145 / 155 | 0.29 / 2 / 5 |
| P inflight, mean / P95 / max | 0 / 0 / 0 | 1.31 / 4 / 8 | 1.35 / 3 / 9 |

Every engine in all three cases physically preallocates the same 376,064-token
KV tensor: 25.83 GiB K plus 25.83 GiB V, or 51.66 GiB per GPU. The model weights
use another 15.28 GiB. Decode engines report about 9.66 GiB available after
CUDA-graph capture, so the native Mooncake Decode GPUs are physically holding
roughly the same 68--70 GiB as the other cases. The 4.7% value only means that
an average of 17,752 out of 376,064 KV slots are logically non-free/non-evictable
according to SGLang's allocator after Decode offload releases its accounting.

The experiment did not record an NVML/DCGM SM-utilization time series. The
available compute-side measure is SGLang Forward busy time: 73.4%, 99.2%, and
99.9% per Decode GPU for colocated, no-reverse, and native Mooncake respectively.

## Mooncake state

Only the native Mooncake case uses shared storage. During its 1200-second
measurement window, Mooncake residency averaged 81.2%, reached the configured
85.0% high watermark, and ended at 83.5% (213.8/256 GiB). It recorded 46
successful eviction rounds, 1,005.55 GiB cumulatively evicted, and zero
allocation failures.

The expected final partial 64-token page may be recomputed on the next Prefill;
that separate boundary behavior is not the failure described above. The failure
is premature release of earlier full pages while the same request is actively
decoding.
