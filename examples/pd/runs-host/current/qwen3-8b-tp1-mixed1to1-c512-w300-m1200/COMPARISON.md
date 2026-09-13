# Qwen3-8B TP=1 mixed 1:1 c512 comparison

> **历史对比，Agentic-PD需重跑：** colocated 行仍可作为原始 baseline；本轮
> Agentic-PD 使用旧“Direct失败→Slow”语义和未对齐显存比例，不代表当前
> “快慢路径+Direct失败重算”方法。

## Setting

- Model: Qwen3-8B
- GPUs: 8
- Agentic PD: 2P:6D, TP=1
- Colocated baseline: 8 colocated workers, TP=1
- Workload: Retool:BrowseComp = 1:1, fixed schedule
- Concurrency: 512
- Sampling: temperature=0, top_p=1, top_k=-1
- Warmup / measurement: 300 / 1200 seconds
- Schedule: `fixed_random_s2026_n8192.json`

The 8,192 dispatch positions have identical `position`, `task_type`, and
`experiment_sample_id` in both runs. All 3,144 sample IDs completed by both
runs also have identical source-question identity.

## Steady-state result

| Metric | Agentic PD 2P:6D | Colocated baseline | PD change |
|---|---:|---:|---:|
| Decode throughput, total | 8,781.7 token/s | 8,970.6 token/s | -2.1% |
| Decode throughput / D GPU | 1,463.6 token/s | 1,121.3 token/s | +30.5% |
| Completed agents | 2,600 | 2,717 | -4.3% |
| Completion rate | 2.167 agent/s | 2.264 agent/s | -4.3% |
| Actual Prefill throughput | 15,660.0 token/s | 23,983.2 token/s | -34.7% |
| Actual Prefill / completed agent | 7,215.9 tokens | 10,581.4 tokens | -31.8% |
| Decode / completed agent | 4,046.5 tokens | 3,957.8 tokens | +2.2% |
| Parent KV page-aligned reuse | 100.0% | 46.35% | +53.65 pp |
| D Forward share / D GPU | 99.63% | 69.25% | +30.38 pp |
| Average D running / GPU | 57.0 | 60.8 | -6.2% |
| Average D KV utilization / GPU | 66.2% | 75.4% | -9.2 pp |

For the colocated baseline, Forward shares are phase shares on each of the
same eight GPUs: Prefill 30.67% and Decode 69.25%. For Agentic PD, the two P
GPUs average 84.25% Prefill Forward and the six D GPUs average 99.63% Decode
Forward.

## Main observation

Agentic PD uses six dedicated D GPUs very efficiently and raises Decode output
per D GPU by 30.5%, while eliminating 31.8% of actual Prefill work per completed
agent through full page-aligned parent-KV reuse. However, total Decode
throughput is 2.1% below colocated in this run because work is strongly skewed
between the two NUMA/P domains:

| Domain | P Forward | D running/GPU | D KV/GPU | D throughput/GPU |
|---|---:|---:|---:|---:|
| P0 + D0-D2 | 98.8% | 34.7 | 47.4% | 1,014.8 token/s |
| P1 + D3-D5 | 69.8% | 79.4 | 84.9% | 1,911.6 token/s |

Thus the aggregate D Forward share is nearly 100%, but half of the D group is
running much smaller Decode batches. This lowers active Decode efficiency to
1,469.1 token/s per active D-GPU-second, versus 1,619.1 for colocated (-9.3%),
which offsets the extra dedicated Decode GPU time.

No Shared Arena exhaustion, Mooncake spill, OOM, or transfer deadlock occurred.
At shutdown, D-to-P fallback/release counts were 525/518 and P-to-D
stage/recover counts were 5,215/5,181; the small residual is in-flight work at
the measurement cutoff.

## D→P 快慢路径比例

本轮旧版快慢路径 Agentic-PD 在正式 1,200 秒窗口内按唯一
request-generation snapshot 统计为：
Direct 8,595，Slow 458，即 Direct 94.94% / Slow 5.06%。Colocated baseline
不使用这套 D→P Direct/Shared-Arena 状态机，因此该指标不适用。
