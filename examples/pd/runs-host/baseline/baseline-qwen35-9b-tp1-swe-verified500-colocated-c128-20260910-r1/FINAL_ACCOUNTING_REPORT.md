# Qwen3.5-9B SWE-bench Verified 500: final accounting

Native pd_mamba_baseline; eight TP=1 colocated replicas; total agent concurrency 128; mem_fraction_static=0.80 on every GPU. PD, HiCache and Mooncake disabled. External Miles PR51-compatible Docker harness unchanged: 8192 output tokens/turn, 64 turns, temperature 0.6, top_p 0.95, top_k 20, min_p 0, thinking enabled. Each container limited to 2 CPUs and 4 GiB. GPU 7 shared with an existing 588 MiB process; not an isolated benchmark.

500 episodes terminated; 495 graded, 5 environment failures; 162/500 resolved (32.4%). Completion milestones include failed episodes, not only resolved episodes. All experiment GPU processes have exited.

## Timing

| Metric | Seconds | Minutes |
|---|---:|---:|
| All 500 terminated, excluding model startup | 5375.16 | 89.59 |
| First 450 terminated (90%) | 2723.96 | 45.40 |
| First 250 terminated (50%) | 1586.84 | 26.45 |
| Individual episode mean, after admission | 696.05 | 11.60 |
| Individual episode P50, after admission | 649.04 | 10.82 |
| Individual episode P90, after admission | 1224.41 | 20.41 |
| Individual arrival-to-finish P50, including concurrency queue | 1589.70 | 26.50 |
| Individual arrival-to-finish P90, including concurrency queue | 2719.48 | 45.32 |

## Whole-run throughput and device occupancy

| Metric | Value |
|---|---:|
| Actual uncached prompt tokens / total wall time, eight GPUs | 2839.78 token/s |
| Model output tokens / total wall time, eight GPUs | 1244.85 token/s |
| Model output tokens / total wall time / GPU | 155.61 token/s |
| Ended episodes / wall time | 0.0930 episode/s |
| KV active/non-evictable pool occupancy, time-weighted GPU mean | 14.33% |
| KV evictable prefix occupancy | 23.39% |
| KV total resident occupancy | 37.72% |
| Mamba active-state occupancy | 6.57% |
| Running requests / GPU | 7.09 |
| Prefill extend CUDA-forward time / GPU / wall time | 6.55% |
| Decode CUDA-forward time / GPU / wall time | 59.61% |
| Prefix hit fraction, token-weighted | 94.60% |
| Raw request retractions | 0 |

Whole-run rates include the long finite-dataset drain. In the 300–2211 s interval before the final new episode was admitted, scheduler Decode was 2366.22 token/s total (295.78/GPU), running/GPU 13.85, active KV 27.47%, resident KV 38.85%, Prefill forward 13.12%, Decode forward 81.26%. This is a post-hoc diagnostic interval, not a separately run 300+1200 steady-state acceptance test.

## Per-episode token accounting

| Metric, tokens except turns | Mean | P50 | P90 | Total |
|---|---:|---:|---:|---:|
| Decode, including reasoning | 13382.52 | 11430 | 24806.2 | 6691261 |
| Actual uncached Prefill | 30528.53 | 27342.5 | 55479.5 | 15264266 |
| Theoretical Prefill without any prefix reuse | 565611.80 | 494933 | 1114013.9 | 282805898 |
| Reused prompt tokens | 535083.26 | 467456 | 1056825.6 | 267541632 |
| Model calls/turns | 44.26 | 48 | 64 | 22129 |

Theoretical Prefill is the sum of the full prompt length at each model call, including repeatedly presented history; it is not unique trajectory length or a single context window. Actual uncached Prefill is the sum of engine prompt_tokens minus engine cached_tokens. Every one of 22129 trajectory calls matched one raw engine completion by prompt length, output length and normalized full output hash, with zero unmatched or ambiguous calls. CSV includes all 500 episodes individually.

The original harness summary reports cached_input_tokens=0 because its Chat usage cache field was absent; this is not evidence of zero cache reuse. This report uses raw engine meta_info.cached_tokens instead and does not alter the harness or original artifacts.

The scheduler's prefill_compute counter reports 15973376 tokens (2971.70 token/s), not the exact unpadded 15264266 tokens above. schedule_policy._update_prefill_budget rounds extend lengths up to page64 before adding log_input_tokens. Rounding each recorded request accounts for 695222 of the 709110-token difference; a remaining 13888 tokens are not attributed by this analysis. Do not call that residual retraction: recorded retractions are zero. Scheduler Decode counter is 6690628 versus 6691261 model output tokens; keep these distinct counter/output accounting definitions.

Artifacts: per_episode_token_accounting.csv, token_accounting_summary.json and analyze_token_accounting.py. Original trajectories, raw logs and original summaries are preserved unchanged.
