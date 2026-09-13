# BrowseComp / Qwen3-8B / native Mooncake / 4P:4D / c576

Completed on 2026-09-10 using the untouched `pd_baseline` engine.
The complete 300.54 s business warmup and 1200.00 s formal window were recorded.
This fills the previously missing native-Mooncake c576 row. Earlier failed runs
are retained and are not included in the reported numbers.

## Results

| Metric | Formal window |
|---|---:|
| Ended agents, including configured truncation | 1,192 |
| Agent/s | 0.9933 |
| Prefill compute throughput, all 4 P GPUs | 37,849.9 token/s |
| Decode throughput, all 4 D GPUs | 2,041.3 token/s |
| Decode throughput per D GPU | 510.3 token/s |
| Prefill Forward time per P GPU | 97.26% |
| Decode Forward time per D GPU | 99.03% |
| P KV utilization, mean per engine | 9.15% |
| D KV utilization, mean per engine | 91.94% |
| P queue / inflight, mean per engine | 17.52 / 1.02 |
| D running, mean per engine | 8.21 |
| D prealloc / transfer, mean per engine | 115.33 / 20.33 |
| Full page-aligned parent-prefix reuse, completion-weighted | 15.13% |
| Actual Prefill tokens / ended agent, window-counter accounting | 38,054 |
| Decode tokens / ended agent, window-counter accounting | 2,052 |
| Mooncake resident fraction, formal mean / maximum | 80.38% / 84.90% |

Throughput uses differences of the engine's realtime token counters, not HTTP
completion bursts. Forward percentages use GPU execution-time counters, not
`nvidia-smi` utilization. Gauge means follow the existing table's arithmetic
mean of per-engine samples. Sampling interval is 2 s.

Low Decode throughput coexists with high Forward duty cycle and a small active
batch. Large prealloc/transfer queues are observed, but this run did not perform
a per-page decomposition of D KV ownership; request counts are not KV bytes.

## Data validity and trajectory characteristics

Formal completions comprise 486 `completed` and 706 `truncated`:
422 hit the existing per-turn length limit, 284 reached the existing budget
condition. There were no recorded request errors, no `aborted` tasks, and no
`search_backend_error` termination. This is a valid throughput measurement
under the existing truncation settings, not evidence of a high task solve rate.

| Completion-weighted metric | Value |
|---|---:|
| Model calls | 3,834 |
| Mean model calls / agent | 3.216 |
| Mean first prompt | 870 tokens |
| Mean prompt / model call | 11,755 tokens |
| Mean Decode / model call | 634 tokens |
| Mean total model input / agent | 37,810 tokens |
| Mean total model output / agent | 2,039 tokens |
| Mean actually uncached input / model call | 10,370 tokens |
| Mean search / open-page calls per agent | 1.862 / 0.664 |
| Mean aggregate tool time per agent | 0.244 s |
| Mean ended-agent latency | 443.59 s |
| Maximum prompt / model call | 38,359 tokens |
| Maximum Decode / model call | 2,048 tokens |

Of 396 one-turn trajectories, 381 hit the per-turn output limit and 15 ended
normally. This is distinct from the previous failed run in which search errors
caused systematic one-turn termination. Across warmup plus measurement there
were 1,667 ended trajectories and 4,694 model calls, also with no aborted tasks.

The completion-weighted token averages above are not the same accounting as
window counters divided by ended agents: the latter also include work performed
within the window for agents still in flight at its end. Agent/s includes
configured truncation, matching the existing catalog, and is not accuracy.
No new bytewise KV comparison or answer-correctness evaluation was performed.

## Configuration and comparability

- Environment: `/homes/siqic/anaconda3/envs/pd_baseline`, SGLang 0.5.10.post1,
  Mooncake transfer engine 0.3.12.post1. No custom agentic modules loaded.
- Model: `/dataset/model/qwen3/Qwen3-8B`; TP1; P GPUs 0/2/4/6, D GPUs 1/3/5/7.
- Closed-loop 576 active agents at both measurement boundaries; pure BrowseComp,
  fixed n680 source-order pool cycled as needed; seed2026.
- Temperature0/top-p1/top-k-1; context40960, total response budget36864,
  max2048 generated tokens per call, max100 turns; inference logprobs disabled.
- GPU allocation: 0.80 on ordinary GPUs and 0.60 for D7 sharing search.
- P native HiCache128 GiB/P; D native Decode-offload Host56 GiB/D; page64,
  page_first layout, kernel I/O, P write_through. Mooncake shared segment256 GiB,
  TCP, native high watermark0.85 / eviction ratio0.10 / prefetch threshold64.
- Native NIXL P→D and native router `power_of_two`; no custom P-ready/cap,
  Shared Host Arena, congestion feedback, or custom reverse-transfer policy.

Startup-only retry r4 exceeded the stock 300 s watchdog while initializing
NIXL. All TP1 workers had automatically bound to NUMA0, including physical
NUMA1 GPUs. The run-local launcher in r5 adds explicit local `--numa-node`
(GPUs0–3→0, GPUs4–7→1) and `--watchdog-timeout 1200`. The latter also applies
during serving. The shared launcher, baseline engine and harness were unchanged.

**CPU/NUMA placement differs from the older c384/c512 runs**, although model,
cache capacity and workload settings align. Differences versus those rows must
not be attributed solely to concurrency. For a strictly controlled concurrency
sweep, the older rows would need the same explicit CPU placement.

Business origin: `2026-09-10 02:02:40.752808 UTC`.
Formal window: `02:07:41.292237` through `02:27:41.293378 UTC`.
Startup and post-window cleanup are excluded. At the final boundary all576
workers were still active; outstanding work was then cancelled by the existing
closed-loop evaluator rather than drained into the measured result.

The run recorded 46 successful Mooncake evictions and zero allocation failures
over its full lifetime. Formal resident-usage statistics are window-specific;
the cumulative eviction count must not be described as a formal-window delta.

## Reproduction and artifacts

```bash
cd /homes/siqic/slime
# Use a new RUN_DIR in the wrapper for a new repetition; do not overwrite this run.
bash examples/pd/runs-host/current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260910-r5/run.sh
```

- [Run wrapper](run.sh) and [isolated baseline launcher](baseline_case.sh).
- [Throughput summary](offload_analysis_summary.json).
- [Trace validity and resource summary](data_validity_and_resources.json).
- [Formal boundaries](closed_loop_boundaries.json), [resolved workload](resolved_workload.json),
  [environment verification](environment.json), [effective configuration](config.json).
- [Throughput visualization](pd_throughput.png), [offload visualization](steady_offload_analysis.png).
- Raw `requests.jsonl`, `engine_metrics.jsonl`, `engine_throughput_2s.jsonl`,
  `closed_loop_events.jsonl`, and `logs/` retained in this directory.
- `monitor.jsonl` contains supplementary low-frequency live checks, not the
  formal2-second source. Its first entry predates colon normalization and is
  invalid; use only entries marked `parser_version=2` for supplementary review.

The existing cleanup traps sent TERM, then bounded KILL to run-owned service
groups. Some driver teardown continued after the group leaders exited. Subsequent
checks confirmed those schedulers gone and no GPU compute processes. The launcher
exited0. No GPU reset, unrelated-process kill, shared-source edit or git change
was used to complete this run; the catalog and run-local artifacts were updated.
