# Slow-path Host-to-P H2D pipeline optimization

## Scope

This note records a performance issue observed in the request-generation slow
recovery path.  It is an optimization task, not a request to change the
request-generation lifecycle, routing policy, eviction policy, or correctness
semantics.

The reference run is:

```text
runs-host/current/ablations/browsecomp-qwen3-8b-4p4d-c512/
  full-shadow-accounting-fix-20260903-r2
```

Measurement window: 1200 seconds, starting at `2026-09-03 01:02:27 UTC`.

## Observed data

Across the four Prefill workers during the measurement window:

| Metric | Value |
|---|---:|
| Slow Host-to-P recoveries | 964 snapshots |
| Average snapshot length | 14,643 KV tokens |
| Average snapshot size | about 2.01 GiB |
| H2D chunk size | 1,024 tokens, about 144 MiB |
| Average chunks per snapshot | 14.75 |
| Exact CUDA-event H2D/scatter time | 164.7 ms mean |
| CUDA-event H2D/scatter bandwidth | 12.20 GiB/s byte-weighted |
| Logged H2D start-to-complete time | 1.18 s mean, 1 s median, 2 s p90, 4 s p99 |
| Logged H2D complete-to-Radix-release time | 2.20 s mean |
| Logged H2D start-to-Radix-release time | 3.38 s mean |

The start/complete/release wall-clock timestamps have one-second resolution,
so their individual values are quantized.  Their aggregate means are useful,
but new profiling should use monotonic sub-millisecond timestamps.

For comparison, recomputing the average 14,643-token parent at the measured
active Prefill rate of about 9,730 token/s/card takes approximately 1.50 GPU
seconds.  The physical recovery copy/scatter takes only about 0.165 GPU
seconds.  Recovery is therefore still substantially cheaper than recomputing
the parent; the problem is control latency, not raw PCIe bandwidth.

## What currently overlaps with Prefill

Slow H2D already runs on separate high-priority CUDA streams and is progressed
by an asynchronous control worker.  Logs show Prefill batches being submitted
between `shared_host_h2d_start` and `shared_host_h2d_complete`, so the 1.18
seconds is **not** a period during which the P GPU is completely unable to
Prefill.

Overlap is not free:

- Host-to-device DMA uses the PCIe copy engine.
- KV scatter uses GPU kernels and HBM bandwidth.
- Prefill uses the same HBM and SM resources.
- CPU bounce preparation and CUDA submission share the P control worker.

The exact H2D/scatter work occupied only about 3.3% of the aggregate four-P
wall-clock capacity in the reference window.  It cannot by itself explain a
large Prefill Forward gap.

## Why 165 ms becomes about 1.18 seconds

One pageable Shared Arena snapshot is not copied as one operation.  Each
1,024-token chunk goes through:

```text
Shared Arena (pageable DRAM)
  -> synchronous CPU memmove into a pinned bounce buffer
  -> Host-to-GPU copy into a GPU staging buffer
  -> GPU scatter into the request's final KV pages
  -> CUDA-event observation by the control worker
  -> submission of the next chunk
```

The approximately 1.02-second difference between wall latency and CUDA-event
time contains:

1. CPU copies from pageable Shared Arena memory into pinned bounce buffers.
   The implementation loops over every K/V layer for every chunk.
2. Roughly fifteen chunk transitions per average snapshot.
3. The configured 20 ms general P-control polling gate.  A completed chunk can
   wait for a later control pass before the next chunk is submitted.
4. Four H2D lanes sharing one Python control worker.  CPU bounce preparation
   and submission are performed sequentially by that worker even though the
   CUDA streams are separate.
5. Ledger, admission, eviction, spill, and other control work executed by the
   same polling loop.  Long control cycles create the latency tail.

The additional average 2.20 seconds after physical H2D completion is a
different problem: the scheduler must observe the completion, perform the
Radix/workset handoff, make the request runnable, and release the Host copy.

Relevant implementation areas in
`sglang-agentic/python/sglang/srt/disaggregation/agentic_host_staging.py` are:

- `start_load_range_to_device`: pageable-to-pinned CPU copy and H2D/scatter;
- `_start_h2d_chunk`: per-chunk submission;
- `_progress_h2d_loads`: CUDA-event polling and chunk chaining;
- `_poll_once` / `_control_loop`: shared control-loop work;
- scheduler event consumption and `gate_request`: Radix/workset handoff.

## Recommended implementation order

### 1. Add precise phase instrumentation first

Record monotonic timestamps and counters per snapshot for:

```text
Host recovery becomes eligible
workset/physical lane acquired
first CPU bounce copy starts
CPU bounce copy time per chunk
CUDA chunk submitted
CUDA chunk completed
next chunk submitted
all H2D/scatter completed
completion event consumed by scheduler
Radix bind completed
request entered runnable queue
Host snapshot released
```

Also record time spent in the other control-loop stages.  Without this,
optimizations can merely move latency between phases.

### 2. Isolate H2D progress from the general control poll

Do not make chunk chaining wait for the 20 ms ledger/admission polling gate.
Use a lightweight dedicated H2D progress loop that only:

```text
queries each active lane's CUDA event
  -> finalizes a completed chunk
  -> immediately submits the next chunk
  -> emits one completion-queue item after the final chunk
```

It must not scan manifests, route requests, evict snapshots, allocate ordinary
requests, or manipulate Radix ownership.  Avoid unbounded busy-spinning; use a
short adaptive wait or event-driven wakeup.

This change must preserve the existing maximum of four physical H2D leases:
no snapshot may reserve P HBM while waiting behind an unavailable physical
lane.

### 3. Reduce the number of chunk transitions

A/B test 1,024, 2,048, and 4,096 tokens per chunk.  At 4,096 tokens the
average snapshot would require about four chunks instead of fifteen.

Larger chunks consume more bounded staging memory and can create longer HBM
interference bursts.  With four lanes, 4,096-token buffers require about 2.25
GiB of pinned Host bounce memory and about 2.25 GiB of GPU staging memory in
total.  Select the smallest chunk size that removes most control bubbles
without reducing Prefill throughput.

### 4. Pipeline CPU preparation with GPU transfer

Give each lane two bounded pinned bounce buffers and, if necessary, two GPU
staging buffers:

```text
GPU transfers/scatters chunk N
  || CPU prepares chunk N+1 in the alternate bounce buffer
```

Never overwrite a bounce or staging buffer before its CUDA fence completes.
All exceptional exits must retain a fence or quarantine the resources; do not
weaken the existing physical ownership rules.

### 5. Make the post-H2D handoff O(1)

After the final H2D event, publish one item into a bounded completion queue.
The request already owns its complete workset lease, so the scheduler-side
consumer should only:

```text
validate request-generation and lease identity
  -> bind the already allocated pages to Radix/request state
  -> place the request in the runnable Prefill queue
  -> release the Host snapshot
```

It must not repeat normal KV-capacity admission, rescan all Host snapshots, or
wait behind unrelated new-request bookkeeping.  Process a small bounded batch
of completions per scheduler iteration so this does not create a new long
scheduler pause.

## Correctness constraints

Any implementation must preserve all of the following:

- Lifecycle identity is `(request_id, generation)` / snapshot ID, never only
  request ID.
- A Host snapshot cannot be evicted while its recovery is pinned or H2D is in
  flight.
- A physical H2D lane, pinned bounce, GPU staging buffer, workset lease, and
  Host snapshot have one explicit owner until a CUDA fence proves completion.
- Direct and Slow recovery remain separate I/O queues.
- D-to-P/P-to-D transport progress remains independent of model Forward.
- Parent KV is released only after the next owner has safely accepted it.
- TP rank 0 owns lifecycle decisions; all TP shards perform the same command
  and commit or fail as one group.
- TP=1 behavior must not regress.
- Parent-KV reuse must remain effectively 100%, apart from the documented
  page-aligned tail and intentional correctness recomputations.

## Validation plan

Change one optimization dimension at a time and compare against the reference
run.  At minimum report:

- exact CPU bounce, CUDA H2D/scatter, inter-chunk gap, and post-H2D bind time;
- mean/p50/p90/p99 Host-ready-to-runnable latency;
- physical and wall H2D bandwidth;
- P Forward share and active/wall Prefill throughput;
- Decode throughput and Forward share;
- Direct/Slow counts and recovery success;
- Host Arena occupancy trend;
- parent-KV reuse and actual Prefill tokens per agent;
- leaked lanes, workset leases, active loads, or Host snapshots.

Initial performance goal: reduce H2D start-to-complete mean and p90 by at
least 50%, and separately reduce complete-to-runnable/release latency by at
least 50%, with no statistically material regression in P/Decode throughput or
parent-KV reuse.  The final target should be set from the new fine-grained
instrumentation rather than the current one-second wall timestamps.
