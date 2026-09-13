# Independent audit: global Slow congestion c512

Verdict: **GO for the scoped c512 trial**, with 300 seconds steady-state warmup and 1200 seconds measurement. This is a code/lifecycle launch gate, not a performance acceptance result.

Auditor: `/root/audit_resume_controller`, 2026-09-09. Reviewed the current production diff and the complete design invariants. No business code was edited and no GPU experiment was launched by this auditor.

## Scope and counting contract

The explicitly disclosed Q is a worker-admission proxy: unique parent request-generations whose tool has returned, whose Host snapshot is durable, and whose H2D worker has not taken the generation. It is **not** the count waiting for actual CUDA DMA submission. `HOST_READY` and `H2D_LOADING` with pinned/leased claims count; any shard at `io_inflight` or `handed` excludes the logical generation. CPU transfer preparation can already be included in `io_inflight`.

The Router tracks active background dispatch references, not tool-call ACK files or raw client connections. Duplicate references count once; dispatch cancellation/end removes its reference; a detached singleflight producer still running after client disconnect remains real work. Cross-P routing does not duplicate the generation. Existing Host ledger snapshot traversal and existing pressure publication are reused. No allocator, transfer, Host ownership mutation, or new control RPC was introduced for Q.

Adaptive mode is default OFF. When ON, it overrides the fixed failure-exit flag: normal mode selects Slow; congestion permits recompute only with existing verified fast-tool evidence. One-second sampling, two high samples to enter, low threshold to exit; configured c512 thresholds are 32/8. Missing, malformed, future-dated, or stale advisory samples cannot enable recompute. The top-level JSON list/null exception identified during review is fixed.

## Eight acceptance criteria

1. **Unique physical owner — PASS.** Q is advisory only. Failure selection remains after physical-fence handling and the force-refreshed `DIRECT_READY` check. The existing claim-safe failure CAS and durable recompute route precede D-source retirement. P-owned `DIRECT_LOADING`/`P_RECEIVED` are not overwritten.
2. **P-to-D Direct source release — PASS, unchanged.** No P-to-D transfer or source-release code changed.
3. **P-to-D Host durable source release — PASS, unchanged.** No Host durability or P-release boundary changed.
4. **D-to-P Host durable source release — PASS, unchanged.** Normal-mode Slow uses the existing staging/durability protocol; Q does not retain or release Host/D pages.
5. **Progress decoupling — PASS at source/test level.** No scheduler/Forward or allocator edits. Router uses its existing background ledger traversal; D performs at most one advisory pressure-file read per second in existing failure progress. Read errors fail soft to Slow. Performance impact must still be measured.
6. **TP atomicity — PASS at source/test level.** `_check_agentic_direct_progress` diverts TP followers before the new reader/decision. Rank zero alone chooses the mode and uses existing group abort fences and release commands. Q counts one logical generation across shards/domains. This c512 TP1 trial is not new TP>1 performance validation.
7. **Parent KV correctness — PASS at source/test level.** Existing fast-arrival evidence, physical fences, exact-generation CAS, route-before-release, and shared-prefix release remain intact. Explicit recompute must be reported separately from successful reuse and true loss. Slow tools cannot become congestion-driven recomputes.
8. **Modification gate — PASS.** Independent source/state-machine review completed. Independent CPU regression rerun: **446 passed, 5 warnings, 21.23 seconds**. Diff whitespace checks passed. Warnings were dependency deprecations and a non-fatal pytest cache permission warning.

## Independent regression command

```text
PYTHONPATH=/homes/siqic/sglang-h100-integration/python /homes/siqic/anaconda3/envs/pd/bin/python -m pytest -q python/sglang/srt/disaggregation/test_agentic_tp.py python/sglang/srt/disaggregation/test_agentic_startup_prewarm.py python/sglang/srt/disaggregation/test_agentic_slow_congestion.py /homes/siqic/slime/examples/pd/tests/test_late_binding_router.py /homes/siqic/slime/examples/pd/tests/test_slow_congestion_router.py
```

Run from `/homes/siqic/sglang-h100-integration`. Coverage includes malformed/missing/stale pressure, hysteresis, duplicate/cancel reference cleanup, global Q and shard phases, adaptive override, fast/slow classification, in-flight fences, lifecycle/TP release, and Router cancellation/singleflight behavior.

## Frozen reviewed SHA-256 hashes

| File | SHA-256 |
| --- | --- |
| `decode_kvcache_offload_manager.py` | `392bcb4a1aa82beaa2cf98545f7f1ea4d22555308a02d8866e2ec3cb5adf4cf3` |
| `agentic_slow_congestion.py` | `60dbe881d820e32fe888f0d80d36b7b71aa8a3fab205d94f7e8561e458bd2dec` |
| `environ.py` | `836ac8c91af0af9db01695281429484c6a958571f0284a4926ccb316ff6116c3` |
| `test_agentic_tp.py` | `70a89de8a05ffd63423b89fe7c09390ce3eb1111a062ccd7e49c66e5524a1127` |
| `test_agentic_slow_congestion.py` | `1213bbcd7400ba22e70bac0369014232f16e6751d4cc6c2ad2e793e9ab0c9797` |
| `late_binding_router.py` | `9ee841f155d3d82e3ae0443f8961c8b1f6c8aeab3a663efdebe13d4aaaba5031` |
| `test_slow_congestion_router.py` | `51f8426d3af50332d844fbb8452377ab6313552554cf29cef023c344f079b6ec` |

## Runtime acceptance still required

- Confirm adaptive flag, shared pressure-file path, thresholds, aligned GPU memory fractions, and unchanged workload before traffic. This source audit does not independently validate a new launcher.
- Check per-generation transfer/ownership conservation and classify every outstanding tail item. Explicit recomputes require durable routing before D release; no release is permitted merely because Q is high.
- Report fast Direct successes, fast Slow, explicit recomputes, unknown-arrival cases, Q/mode timeline, and Forward activity. Missing fast-arrival proof remains conservatively Slow.
- Quantify worker-admission-to-actual-DMA delay. CPU preparation backlog is outside this Q and may make the proxy underestimate actual DMA waiting; do not report the two queues as equivalent.
- The 32/8 thresholds and this Q definition are trial choices, not proven optimal settings or a promised throughput gain.
