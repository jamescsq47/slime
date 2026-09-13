# Native colocated c128 restart — 2026-09-12

Status: launched at 16:37:59 UTC; full 500-task evaluation,
no final performance/accuracy result yet. Supervisor PID2271781;
launcher PID2271782; offline report PID2272740.
All four TP2 replicas became ready; workload loaded all500 tasks at
16:42:13 UTC. Prefill and Decode batches are executing. No assertion observed
at startup; this is not yet a long-run correctness conclusion.

Environment: `pd_mamba_baseline`, SGLang 0.5.14 **plus minimal resumed-Mamba-chunk alignment fix**.
Not the unpatched package and not the custom agentic PD engine.

Configuration unchanged from c128 r1: Qwen3.5-27B; TP2; four replicas on
`0,4 / 1,5 / 2,6 / 3,7`; c128; static memory 0.80; Mamba/full memory ratio 0.9;
page64, track64, prefill chunk8192; context131072; per-turn8192; max64 turns;
response budget81920; temperature0.6, top-p0.95, top-k20, min-p0, thinking enabled.
Same 500 unique SWE-bench Verified tasks/order and external OpenEnv harness.
PD, HiCache, Mooncake remain off. This is finite500 evaluation, not a 300+1200
closed-loop serving benchmark. GPU7's pre-existing 588MiB process is untouched.

## Failure and minimal correction

r1 stopped at 21 completed tasks after native `cache_unfinished_req` found
`cache_len=21412`, page-aligned length21376. Resumed middle-chunk admission used
an arbitrary remaining-token budget, unlike first-chunk page alignment. A
36-token partial tail became the next chunk's prefix, shifting its Mamba
checkpoint off a KV page boundary.

Only two installed engine files changed:

- `managers/schedule_policy.py`: align non-final resumed SSM chunks to
  `lcm(page_size, mamba_cache_chunk_size)`; park if less than one chunk fits.
- `managers/scheduler.py`: only an actually admitted chunk is attached to the
  forward and counted as in flight / deducted from pending-token metrics.

Cache insertion assertions, state contents, final suffix tokens, harness,
sampling, transfer paths and attention-only admission are unchanged.
The separate upstream overlap/decode-boundary patches were not bundled.

Patch: `examples/pd/patches/sglang_0_5_14_mamba_resumed_chunk_alignment.patch`.
Run-local `engine-fix/` contains original files, patched files and patch.

Validation: 56 CPU regression/launcher/report tests passed (run-local
`engine-fix/tests.log`), including exact
old21412 reproducer, real tracking method, capacity edges, final tails,
attention-only invariance and parked-chunk cancellation accounting.
Independent audit (`audit_baseline_mamba_alignment`): **GO for GPU validation**;
the auditor independently reran all47 alignment tests. Full500 completion not yet proven.
Preflight dictionaries, dataset/workload hashes and launcher snapshot hash
were compared with c128 R1 and are identical.
Design checks: original HBM owner retained; no PD ownership transitions;
no added transfer/I/O waits; same-budget TP ranks make identical decisions;
KV and SSM checkpoint consistency preserved; independent test/audit gate met.

## Execution and results

Run: `/tmp/pd-persist/baseline-qwen35-27b-tp2-swe500-colocated-c128-20260912-r2-chunkalign`.
Launcher: `scripts/baseline/run_qwen35_27b_tp2_swe500_mamba_baseline.sh` (unchanged).
Supervisor: `scripts/baseline/monitor_qwen35_27b_swe500.py`, checks every30s,
stops only owned process groups/containers on fatal worker errors.
Metrics/report: `scripts/tools/summarize_native_swe_colocated.py`.
Records include trajectories, task/verifier/tool times, token lengths,
2-second engine metrics, Attention/Mamba pool use, running and Forward time.
No completed-result table is emitted for a cancelled/partial run.
