# Qwen3.5-9B SWE-bench Verified: concurrency comparison
Same 500 tasks, eight TP=1 colocated replicas, pd_mamba_baseline, mem_fraction_static=0.8; no PD/HiCache/Mooncake. Same harness, source order and sampling: temperature=0.6, top_p=0.95, top_k=20, min_p=0, thinking on, 8192 tokens/turn, 64 turns. Docker 2 CPU/4 GiB limits; shell timeout 600 s, verifier timeout 2400 s. Finite full-dataset evaluations, not closed-loop steady-state acceptance runs. External GPU co-tenants differed across runs; results are single runs, not statistically isolated repetitions.

## Outcome and timing

| Metric | c128 | c256 | c500 |
|---|---:|---:|---:|
| Resolved | 162/500 (32.4%) | 158/500 (31.6%) | 157/500 (31.4%) |
| Environment failures | 5 | 4 | 4 |
| All 500 ended, minutes | 89.59 | 54.37 | 65.04 |
| 450 ended / 90%, minutes | 45.40 | 32.50 | 42.41 |
| 250 ended / 50%, minutes | 26.45 | 19.90 | 31.99 |
| Episode execution mean, seconds | 696.05 | 849.90 | 1,805.42 |
| Episode execution P50, seconds | 649.04 | 786.59 | 1,918.70 |
| Episode execution P90, seconds | 1,224.41 | 1,472.32 | 2,541.40 |
| Arrival-to-finish P50 including queue, seconds | 1,589.70 | 1,191.15 | 1,918.70 |
| Arrival-to-finish P90 including queue, seconds | 2,719.48 | 1,946.30 | 2,541.40 |

Completion includes failed episodes, not only correct solutions. Execution is from agent-pool admission through tools/verifier/cleanup; excludes admission queue. c500 also has one verifier timeout (not included in the four environment errors), django__django-14539, 2401.04 s. Its final result determines the 65-minute makespan.

## Whole-run wall-clock throughput and time-weighted GPU averages

| Metric | c128 | c256 | c500 |
|---|---:|---:|---:|
| Uncached Prefill token/s, total | 2,839.78 | 5,333.18 | 9,850.39 |
| Model output token/s, total | 1,244.85 | 2,016.79 | 1,674.80 |
| Model output token/s/GPU | 155.61 | 252.10 | 209.35 |
| Running/GPU | 7.09 | 13.91 | 15.93 |
| Active/non-evictable KV fraction | 14.33% | 26.74% | 29.45% |
| Evictable prefix KV fraction | 23.39% | 24.53% | 44.73% |
| Total resident KV fraction | 37.72% | 51.27% | 74.19% |
| Mamba active-state fraction | 6.57% | 12.84% | 15.00% |
| Prefill forward fraction/GPU | 6.55% | 11.33% | 18.00% |
| Decode forward fraction/GPU | 59.61% | 64.28% | 51.32% |
| Token-weighted prefix hit rate | 94.60% | 93.72% | 85.77% |
| Retractions in raw completions | 0 | 0 | 0 |

## Per-episode lengths

| Metric | c128 | c256 | c500 |
|---|---:|---:|---:|
| Model calls: mean | 44.26 | 44.19 | 43.08 |
| Model calls: p50 | 48.00 | 49.00 | 45.00 |
| Model calls: p90 | 64.00 | 64.00 | 64.00 |
| Model calls: total | 22,129.00 | 22,094.00 | 21,540.00 |
| Decode tokens: mean | 13,382.52 | 13,158.66 | 13,071.52 |
| Decode tokens: p50 | 11,430.00 | 11,516.50 | 11,542.00 |
| Decode tokens: p90 | 24,806.20 | 24,478.60 | 24,352.00 |
| Decode tokens: total | 6,691,261.00 | 6,579,330.00 | 6,535,760.00 |
| Actual uncached Prefill tokens: mean | 30,528.53 | 34,796.63 | 76,880.71 |
| Actual uncached Prefill tokens: p50 | 27,342.50 | 28,829.00 | 64,238.50 |
| Actual uncached Prefill tokens: p90 | 55,479.50 | 67,246.90 | 154,025.10 |
| Actual uncached Prefill tokens: total | 15,264,266.00 | 17,398,317.00 | 38,440,355.00 |
| No-reuse theoretical Prefill tokens: mean | 565,611.80 | 554,462.94 | 540,451.78 |
| No-reuse theoretical Prefill tokens: p50 | 494,933.00 | 495,758.00 | 456,693.50 |
| No-reuse theoretical Prefill tokens: p90 | 1,114,013.90 | 1,095,377.40 | 1,088,724.40 |
| No-reuse theoretical Prefill tokens: total | 282,805,898.00 | 277,231,469.00 | 270,225,891.00 |

## Post-hoc common elapsed 300-1500 s interval

| Metric | c128 | c256 | c500 |
|---|---:|---:|---:|
| Scheduler Decode token/s, total | 2,305.32 | 3,660.72 | 2,593.43 |
| Scheduler page-rounded Prefill token/s, total | 6,154.56 | 9,953.42 | 18,739.44 |
| Running/GPU | 13.75 | 26.66 | 27.91 |
| Active KV fraction | 28.55% | 51.91% | 38.06% |
| Resident KV fraction | 40.67% | 63.66% | 78.17% |

This common interval is diagnostic, not necessarily identical steady-state workload mix: c500 starts all tasks early and drains. Whole-run exact Prefill uses raw prompt_tokens-cached_tokens; theoretical Prefill sums full prompts across calls, not unique history; Decode includes reasoning. All raw completions matched trajectories. Scheduler Prefill rounds to page64 and is not exact unpadded token count. c500 has partial/missing metric scrapes: counters must be differenced per endpoint, not from partial aggregate sums. Gauges are held across gaps and are estimates; see scrape_diagnostics in JSON. Original summary.json throughput values are not authoritative for c500.

Conclusion: c256 had the shortest full makespan and T450. c500 lowered the prefix hit rate and increased actual Prefill despite similar model output lengths; zero retractions does not exclude prefix eviction/recomputation. Exact attribution among cache eviction, routing and changed trajectories is not established here. Accuracy differs by only 4–5 solved tasks across single sampled runs; no causal accuracy claim is justified.
