# BrowseComp Qwen3-8B c384: explicit Host prewarm barrier

## Status

Completed successfully with 300.01 seconds of business warmup followed by a
1,200.0008-second measurement window. At both boundaries there were 384 active
agents. During the formal window, 2,822 agents completed and no request failed.
There were no Router HTTP 500 responses, CUDA OOMs, or worker tracebacks.

All eight P/D CUDA contexts completed Host-Arena registration before search,
Router, or workload traffic started. The four P contexts each registered
544 GiB in 295.6--295.7 seconds; the four D contexts each registered 640 GiB
in 336.9--337.0 seconds.

## Formal-window performance

| Metric | Result |
|---|---:|
| Decode throughput | 4,700.0 token/s |
| Decode throughput per D | 1,175.0 token/s |
| Prefill compute throughput | 32,669.1 token/s |
| Completion throughput | 2.352 Agent/s |
| D Forward per card | 97.62% |
| P Forward per card | 86.96% |
| D active-time throughput per card | 1,203.7 token/s |
| Actual Prefill per completed Agent | 13,856 tokens |
| Decode per completed Agent | 1,993 tokens |
| Page-aligned reverse Decode-KV reuse | 100.00% (99.995%) |
| Page-aligned parent-prefix reuse | 99.98% |

Average D state across workers was 52.8 running requests and 86.4% KV usage.
Per-worker running averages were 54.9, 55.9, 60.2 and 40.1; KV averages were
87.0%, 87.2%, 86.2% and 85.3%. GPU7 has the smaller 0.60 KV pool because it
also runs search. The other D workers use 0.80.

Average P state across workers was 90.6% KV usage, 8.7 queued requests and
12.4 P-to-D inflight requests. The 2,822 completed trajectories averaged 3.58
generation turns, 2.08 searches, 46,305 logical model-prompt tokens, 16,650
total response tokens and 2,014 model-completion tokens.

## Comparison

| Formal run | Decode | Agent/s | D Forward/card | D active token/s/card | D running/card | D KV |
|---|---:|---:|---:|---:|---:|---:|
| Current code, c384 | 4,700.0 | 2.352 | 97.62% | 1,203.7 | 52.8 | 86.4% |
| Current code, c512 | 4,638.2 | 2.320 | 97.90% | 1,184.4 | 50.6 | 85.0% |

The current c384 result is 1.33% faster than the current c512 result. These two
runs use the same code, D memory fractions (`0.80/0.80/0.80/0.60`), workload
sequence, startup barrier and measurement procedure.
