# SWE-bench Verified + Qwen3.5-27B TP=2

## 实验矩阵

当前可比配置使用 SWE-bench Verified 500 题、OpenEnv harness、Qwen3.5-27B、
TP=2、8 张 A100、并发 128。Agent 每轮最多生成 8,192 tokens，最多 64 轮，
总上下文上限 131,072 tokens；采样参数为 temperature 0.6、top-p 0.95、
top-k 20，并开启 thinking。

`4P:4D` 表示两个 TP=2 Prefill 组和两个 TP=2 Decode 组。PD 方法后续也应
使用相同数据顺序、harness、采样参数和并发，以便与 colocated 结果直接比较。

| 方法 | 配置 | 状态 | 正确率 | T450 |
|---|---|---|---:|---:|
| Colocated baseline | 4 个 TP=2 replica，c128，8K/64，run 1 | 完成 | 44.4% | 1:00:22 |
| Colocated baseline | 4 个 TP=2 replica，c128，8K/64，run 2 | 完成 | 46.2% | 0:54:56 |
| No-reverse PD baseline | 4P:4D，c128，8K/64 | 过慢；未形成完整可比结果 | — | — |
| 原生 HiCache + Mooncake PD baseline | 4P:4D，c128，page64，8K/64 | Qwen3.5 Mamba 状态回传有问题；结果无效 | — | — |
| 当前新方法 | 4P:4D，c128，page64，8K/64 | 待完成 | — | — |

## Colocated 已完成结果

| Run | 完成 / 失败 | 通过题数 | 正确率 | 平均轮数 | 平均模型输出 | T450 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 499 / 1 | 222 | 44.4% | 47.804 | 9,175.850 tokens | 1:00:22 |
| 2 | 500 / 0 | 231 | 46.2% | 48.916 | 9,140.526 tokens | 0:54:56 |
| 两次均值 | 499.5 / 0.5 | 226.5 | 45.3% | 48.360 | 9,158.188 tokens | — |

`T450` 从第一条 workload 到达到第 450 个状态为 completed 的任务完成为止，
不包含失败任务。两次 run 的准确率相差 1.8 个百分点；在 temperature 0.6 下，
应将它们视为同一配置的重复实验，而不是不同 serving setting。

## Colocated 稳态特征

| Run | 饱和窗口 Decode | Running / replica | KV / replica |
|---:|---:|---:|---:|
| 1 | 1,097 token/s | 27.3 | 88.8% |
| 2 | 1,199 token/s | 27.4 | 86.8% |

这里的 Decode 吞吐是每个 TP=2 colocated replica 的饱和窗口值。两次运行生成
工作量接近，完成时间差主要来自约 8.5% 的 Decode 速率差异。

## PD baseline 状态说明

- No-reverse PD 每一轮都需要重新 Prefill 累积历史，在 SWE-bench 的长轨迹、
  多轮工具调用下运行过慢，因此当前没有可与两次 colocated 全量结果并列的
  500 题正式结果。
- 原生 HiCache + Mooncake 的既有尝试没有正确保留 Qwen3.5 hybrid 模型所需的
  Attention KV 与 Mamba temporal/conv state 对应关系，出现不可信的异常高复用；
  这些运行不记录为有效吞吐或正确率结果。
- 当前新方法必须在 Mamba Cache V2、page64 和 TP=2 数据通路验证完成后，再填入
  正式的 300 秒预热、1,200 秒稳态性能以及全量正确率结果；本表暂不使用 smoke
  或失败运行回填。

## D→P 快慢路径比例

| 方法 | 配置 | Direct | Slow | Direct/Slow 比例 |
|---|---|---:|---:|---:|
| 当前新方法 | 4P:4D，c128 | — | — | 待正式实验 |
| Colocated / No-reverse / 原生 Mooncake | — | — | — | 不适用 |

正式结果将按 1,200 秒测量窗口和唯一 request-generation snapshot 统计；TP=2
的两个 rank 合并计数，不用 smoke 或失败运行回填。

## 结果来源

- 两次 colocated 复现实验及终止原因统计：
  [重复实验汇总](archive/docs/SWEBENCH_QWEN35_8K64_VS_16K128_REPEATS.md)
- 当前表只收录 8K/64-turn colocated 数据；16K/128-turn 结果仍保留在 archive，
  但不参与当前 PD 对齐矩阵。
