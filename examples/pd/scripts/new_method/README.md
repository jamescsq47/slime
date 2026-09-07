# Request-generation KV pipeline

These launchers use the modified `pd` environment.  Every launcher validates
the environment fingerprint before allocating GPUs.

Supported entry points:

- `run_1p3d_case.sh`: current validated formal configuration; Mixed 1:1,
  c256, seed 2026, 300-second warmup and 1200-second measurement.
- `run_1p5d_case.sh`: scale-out 1P:5D configuration.
- `run_2p6d_numa_case.sh`: two-NUMA-domain 2P:6D configuration.
- `run_4p4d_numa_case.sh`: four-P-domain 4P:4D configuration.  Each D uses
  the Host/slow-path arena of its same-NUMA paired P, while completed Prefill
  KV may still be late-bound to any feasible D.
- `run_qwen3_8b_tp1_browsecomp_4p4d.sh`: formal Qwen3-8B pure-BrowseComp
  source-order run, TP=1, 4P:4D, closed-loop c256, 300-second steady-state
  warmup and 1200-second measurement.
- `internal/`: lifecycle and model-service implementation used by the entry
  points.  KV payloads use Direct or Shared Host Arenas; lifecycle metadata is
  kept in the run-scoped `/dev/shm` directory.  Do not launch these directly.

Both D→P and P→D Shared Host payloads default to the `memfd` backend.  The
arenas are pageable ordinary CPU memory, are first-touched under each P
process' NUMA binding, and do not consume the `/dev/shm` mount quota.  Only
small ledgers and marker directories remain in `/dev/shm`.  The Qwen3-8B
4P:4D launcher reserves independent 256 GiB D→P and 64 GiB P→D arenas per P.
Set `SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND=tmpfs` and/or
`SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND=tmpfs` to use the file-backed data
plane.

Both directions lazily acquire the final request extent from a bounded
process-local CUDA-registered window cache on a transport worker.  The generic
defaults are 8-GiB windows/64-GiB cache; the Qwen3-8B 4P:4D launcher uses a
1280-GiB per-process upper bound. Global Decode/Prefill routing lets one
process visit all four P domains, so the bound covers `4 * (256+64) GiB` and
prevents already-warmed remote arenas from evicting one another.
D2H DMA lands directly in the durable arena and H2D DMA reads directly
from it. On CUDA 13, a background CPU index mirror coalesces adjacent allocator
tokens and `cuMemcpyBatchAsync` moves every run with copy engines only, removing
both the extra CPU memcpy and gather/scatter SM kernels from all four Slow legs.
The Qwen3-8B launcher uses two lanes and 4096-token batches. Registration or
batch-API unavailability before DMA safely falls back to the prior gather/
scatter two-bounce pipeline. Set
`SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA=0` to force that fallback.  D→P Slow
control uses a bounded, sticky round-robin active window instead of scanning
every retained snapshot.

The same launcher enables `SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA=1`.  The
first transport-worker access registers every backing-arena window during
warmup, then drops the temporary references while retaining the mappings in
the bounded cache.  This keeps sparse D→P requests from paying a one-time
page-registration cost inside the measurement window; a prewarm failure is
non-fatal and returns to request-level registration/fallback.

The current policy uses a 2-second fast-tool threshold, one 2-second
tool-return-relative Direct setup bound, exact-size Direct receive pages,
arrival/native Prefill admission independent of KV source, two-stage Slow
Host/HBM routing, late Decode binding and P-ready compute-ahead backpressure.

```bash
cd /homes/siqic/slime
bash examples/pd/scripts/new_method/run_1p3d_case.sh
```

Override `RUN_DIR`, GPU/port variables, `MAX_INFLIGHT`, `WARMUP_SECONDS`, or
`MEASURE_SECONDS` through the environment.  The retained reference result is
`runs-host/new-method/1p3d-c256-s2026-w300-m1200`.
