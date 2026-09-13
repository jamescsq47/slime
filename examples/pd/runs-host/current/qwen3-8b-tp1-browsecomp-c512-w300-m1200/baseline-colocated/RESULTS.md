# BrowseComp-only colocated baseline: 8 GPU, c512

The setting matches the c384 run except for closed-loop concurrency 512. The 680 original BrowseComp rows are admitted in source order and cycle, temperature is 0, the context limit is 40,960, the per-turn output cap is 2,048, Radix page size is 64, and `return_logprob=false`.

## Measurement-window results

| Metric | c384 | c512 | c512 change |
|---|---:|---:|---:|
| Agent throughput | 2.3917/s | 2.1808/s | -8.8% |
| Prefill compute throughput | 38,148.7 token/s | 46,523.3 token/s | +22.0% |
| Prefix-cache hit throughput | 70,708.1 token/s | 53,000.3 token/s | -25.0% |
| Decode throughput | 4,834.2 token/s | 4,434.2 token/s | -8.3% |
| Decode throughput/GPU | 604.3 token/s | 554.3 token/s | -8.3% |
| Prefill Forward/GPU | 51.12% | 60.40% | +9.28 pp |
| Decode Forward/GPU | 48.81% | 39.56% | -9.25 pp |
| Total Forward/GPU | 99.93% | 99.95% | +0.02 pp |
| Running/GPU | 46.35 | 59.90 | +29.2% |
| KV usage/GPU | 67.61% | 84.13% | +16.53 pp |
| Queue/GPU | 0.76 | 3.07 | +306% |

The workloads remain comparable: model calls/agent change from 3.537 to 3.564, logical prompt tokens/agent from 45,104 to 45,836, and Decode tokens/agent from 2,023 to 2,034.

## KV thrashing

| Metric | c384 | c512 |
|---|---:|---:|
| Parent-prefix reuse | 91.74% | 74.11% |
| Missing parent KV | 7,399,936 tokens | 21,716,288 tokens |
| Missing parent KV/agent | 2,578 tokens | 8,298 tokens |
| Extra Prefill from parent loss | 16.30% | 38.75% |
| Actual Prefill/agent | 15,819 tokens | 21,416 tokens |

c512 creates a larger Decode batch: active Decode efficiency increases from 1,238 to 1,401 token/s per Decode GPU-second. However, missing parent KV triples and actual Prefill/agent rises 35.4%. Prefill therefore occupies 60.4% instead of 51.1% of GPU time, leaving only 39.6% for Decode. The wall-clock Decode and Agent throughput both decrease.

The degradation grows over time. In the first and second 600-second halves, parent-prefix reuse changes from 77.39% to 70.47%, actual Prefill rises from 20,680 to 22,188 tokens/agent, and throughput drops from 2.233 to 2.128 agent/s.

As in the c384 report, missing-parent-KV tokens combine LRU eviction and any cache-aware-router placement miss because the clean baseline does not expose per-request worker selection or an eviction event counter.

## D→P 快慢路径比例

不适用：该实验为 colocated baseline，没有 D→P Direct/Shared-Arena 路径。
