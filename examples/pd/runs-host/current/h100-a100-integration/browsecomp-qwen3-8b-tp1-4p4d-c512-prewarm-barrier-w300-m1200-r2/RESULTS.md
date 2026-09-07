# BrowseComp Qwen3-8B c512: explicit Host prewarm barrier

## Status

Completed successfully with 301.20 seconds of business warmup followed by a
1,200.0006-second measurement window. At the measurement boundary there were
512 active agents; 2,784 agents completed during the window with no terminal
request failures.

One generation encountered a dynamic-P-route timeout and received one HTTP
500. The closed-loop client retried it successfully; the event did not spread
or terminate a worker.

## Startup barrier

The launcher did not start search, Router or workload traffic until every P/D
CUDA context explicitly published `prewarm_complete`.

| Role | Contexts | Arenas/context | Registered/context | Time/context |
|---|---:|---:|---:|---:|
| Prefill | 4 | 5 | 544 GiB | 296.6–296.7 s |
| Decode | 4 | 8 | 640 GiB | 339.6–339.8 s |

The sum across CUDA mappings is 4.625 TiB, but these are mappings of the same
640 GiB of physical memfd-backed Host Arena payload, not 4.625 TiB of distinct
DRAM allocations.

## Formal-window performance

| Metric | Result |
|---|---:|
| Decode throughput | 4,638.2 token/s |
| Decode throughput per D | 1,159.5 token/s |
| Prefill compute throughput | 32,889.0 token/s |
| Completion throughput | 2.320 Agent/s |
| D Forward per card | 97.90% |
| P Forward per card | 86.81% |
| D active-time throughput per card | 1,184.4 token/s |
| Actual Prefill per completed Agent | 14,154 tokens |
| Decode per completed Agent | 1,996 tokens |
| Page-aligned reverse Decode-KV reuse | 99.76% |
| Page-aligned parent-prefix reuse | 99.26% |

Average D state by worker was 40.3–58.2 running requests and 83.7%–85.7% KV
usage. GPU7 has the smaller 0.60 KV pool because it also runs search; the other
D workers use 0.80.

## Host paths

Within the formal window, 4,753 unique D→P Direct transfers completed and
2,340 unique snapshots became D→P Host-ready. Of those slow snapshots, 2,336
completed Host→P recovery within the same window; lifecycle work crossing the
window boundary accounts for the small difference.

| Arena/path | Mean occupancy | Peak occupancy | Transfer bandwidth mean |
|---|---:|---:|---:|
| D→P Shared Host, 512 GiB total | 251.0 GiB | 444.2 GiB | D2H 10.73, H2D 12.19 GiB/s |
| P→D Shared Host, 128 GiB total | 93.1 GiB | 126.1 GiB | D2H 12.08, H2D 9.80 GiB/s |

The run-local raw logs, request records, two-second samples, plot, boundary,
prewarm report and generated analysis JSON remain beside this report.
