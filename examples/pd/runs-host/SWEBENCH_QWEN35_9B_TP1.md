# SWE-bench Verified + Qwen3.5-9B TP=1

更新日期：2026-09-11。本文整理三组完整 colocated 并发实验和四组完整
Agentic PD 实验；100 题调试、smoke 和中途终止的运行不作为全量性能结果。

新增完整R11：[H2D物理槽解耦对比](SWEBENCH_QWEN35_9B_H2D_DECOUPLING.md)。
恢复前排队91.24→62.58个，但中段D吞吐4059→3995 token/s、P Forward71.92%→67.74%，
未获得吞吐提升。该报告包含完整时间、长度、KV、工具分布及生命周期守恒，不将R10中止结果混入。

## 实验矩阵

共同设置：SWE-bench Verified **500 个不同任务各执行一次**，OpenEnv/Miles PR51
兼容的外部 Docker harness，Qwen3.5-9B、TP=1、a10 节点 8 张 A100。
每轮最多生成 8,192 tokens，最多 64 轮，上下文上限 131,072 tokens；
temperature=0.6、top-p=0.95、top-k=20、min-p=0、thinking 开启。
各轮使用相同数据顺序及 harness，不使用新增恢复 prompt。

所有 GPU 的 `mem_fraction_static=0.80`。Docker 每任务上限 2 CPU、4 GiB，
shell 调用超时 600 秒，verifier 超时 2,400 秒、最多 16 个并行 verifier。
镜像已提前下载；容器在任务进入执行池时启动，而不是每次 shell 调用重新启动。
`return_logprob=false`。c128/c256/c500 指整个系统的 agent 并发上限，不是每卡
Decode batch，也不是每类任务的固定并发。

| 方法 | 配置 | 状态 | 正确率 | T450（仅 completed） |
|---|---|---|---:|---:|
| Colocated baseline | 8 个 TP=1 replica，c128，8K/64 | 完成 | 32.4% | 0:45:59 |
| Colocated baseline | 8 个 TP=1 replica，c256，8K/64 | 完成 | 31.6% | 0:32:46 |
| Colocated baseline | 8 个 TP=1 replica，c500，8K/64 | 完成 | 31.4% | 0:42:37 |
| 早期新方法 R4 | 2P:6D，c256，Q=16/4 | 中止，保留 260 条记录；不纳入全量比较 | — | — |
| 新方法 R5 | 2P:6D，c256，Q=32/8，应用端确认终止 | 完成全量测评；性能/回收问题尚存 | 30.6% | 0:44:00 |
| 新方法 R6 | 2P:6D，c500，关闭主动重算，Q32/32 不生效 | 完成全量测评；Host 终止回收问题尚存 | 29.0% | 0:31:19 |
| 新方法 R9 | 2P:6D，c500，每P 4个H2D槽，Mamba准入/Host回收修复 | 完成全量测评；中段吞吐未提升 | 32.0% | 0:31:24 |
| 新方法 R11 | 2P:6D，c500，4个物理H2D槽/P，8个驻留恢复额度/P，槽位解耦 | 完成全量测评；中段吞吐未提升 | 29.6% | 0:31:20 |

`2P:6D` 表示 2 张独立 Prefill GPU（0、4）和 6 张独立 Decode GPU
（1、2、3、5、6、7），每个 worker 都是 TP=1。

这里沿用 [27B 报告](SWEBENCH_QWEN35_27B_TP2.md) 的 T450 定义：从第一条
workload 到达到第 450 个 `status=completed` 任务结束，不包含 failed 任务，
但包括未解题成功的 completed 任务。它不是第 450 个正确答案的时间。
下文另列“450 题收尾”，该指标包含 failed，避免与此前对话中的口径混淆。

**可比性限制：** colocated 使用 `pd_mamba_baseline`，SGLang 0.5.14，
Mamba:Attention 池比例为 0.9；R5/R6/R9 使用融合源码
`/homes/siqic/sglang-qwen35-integration`，base commit `7f22a376487998451087e23ce3a9b275275df62d`
加各轮保存的 `engine.patch`（R9另有 `engine-source/` 快照），Mamba 池比例为 0.5。R5/R6/R9 用 `pd` 的依赖及
独立 `PYTHONPATH` 加载融合源码，不修改安装环境。
虽然静态显存、模型和 harness 对齐，二者并非同一引擎、同一状态池划分的单因素消融。
各组只跑一次，GPU 上其他用户的小型进程也并非完全一致；不能把单次正确率差异
归因于某一种调度或缓存策略。

R6 的引擎 diff、Router、harness 和数据与 R5 一致，但同时将并发 256→500、
关闭拥塞触发重算；不能从这两组单独分离并发收益和关闭重算收益。

## Colocated 与新方法已完成结果

| Run | 完成 / 失败 | 通过题数 | 正确率 | 平均轮数 | 平均模型输出 | T450（仅 completed） |
|---|---:|---:|---:|---:|---:|---:|
| Colocated c128 | 495 / 5 | 162 | 32.4% | 44.258 | 13,382.522 tokens | 0:45:59 |
| Colocated c256 | 496 / 4 | 158 | 31.6% | 44.188 | 13,158.660 tokens | 0:32:46 |
| Colocated c500 | 496 / 4 | 157 | 31.4% | 43.080 | 13,071.520 tokens | 0:42:37 |
| 新方法 R5 c256 | 497 / 3 | 153 | 30.6% | 44.128 | 13,442.610 tokens | 0:44:00 |
| 新方法 R6 c500 | 498 / 2 | 145 | 29.0% | 43.996 | 13,731.738 tokens | 0:31:19 |
| 新方法 R9 c500，H2D=4/P | 498 / 2 | 160 | 32.0% | 44.342 | 13,451.970 tokens | 0:31:24 |
| 新方法 R11 c500，H2D解耦 | 496 / 4 | 148 | 29.6% | 43.212 | 13,567.036 tokens | 0:31:20 |

| 时间指标 | Colocated c128 | Colocated c256 | Colocated c500 | 新方法 R5 c256 | 新方法 R6 c500 | 新方法 R9 c500，4槽/P |
|---|---:|---:|---:|---:|---:|---:|
| 全部 500 题收尾 | 1:29:35 | 0:54:22 | 1:05:02 | 1:00:56 | 0:44:49 | 1:00:35 |
| 450 题收尾，包含失败 | 0:45:24 | 0:32:30 | 0:42:25 | 0:43:45 | 0:31:12 | 0:31:24 |
| 单题执行平均时间 | 696.05 s | 849.90 s | 1,805.42 s | 1,179.90 s | 1,290.46 s | 1,293.87 s |
| 单题执行 P50 | 649.04 s | 786.59 s | 1,918.70 s | 1,153.76 s | 1,359.18 s | 1,362.26 s |
| 单题执行 P90 | 1,224.41 s | 1,472.32 s | 2,541.40 s | 1,894.28 s | 1,868.18 s | 1,879.20 s |

全量收尾时间从首题开始执行到最后一题退出，包含工具、verifier 及末尾 drain，
不包含模型服务启动。单题执行时间从进入执行并发池计起，不包含此前的并发准入排队。
R5 实际区间为 **2026-09-11 02:38:37–03:39:33 UTC**。
R6 实际区间为 **2026-09-11 15:30:59–16:15:48 UTC**，精确墙钟 2,688.883 秒。
R9 实际区间为 **2026-09-11 18:07:13.906–19:07:49.154 UTC**，精确墙钟 3,635.248 秒。
Colocated c500 另有 1 个 verifier 超时（约 2,401 秒），它不是表内 4 个环境失败之一，
该尾部任务影响了全量收尾时间。

### 全程墙钟吞吐与数据长度

以下吞吐为精确原始 token 总数除以全量墙钟；不是只除以 GPU Forward 时间。

| 指标 | Colocated c128 | Colocated c256 | Colocated c500 | 新方法 R5 c256 | 新方法 R6 c500 | 新方法 R9 c500，4槽/P |
|---|---:|---:|---:|---:|---:|---:|
| 未命中 Prefill 吞吐，总计 | 2,839.78 token/s | 5,333.18 token/s | 9,850.39 token/s | 6,241.24 token/s | 3,914.49 token/s | 2,976.92 token/s |
| 模型输出吞吐，总计 | 1,244.85 token/s | 2,016.79 token/s | 1,674.80 token/s | 1,838.38 token/s | 2,553.43 token/s | 1,850.21 token/s |
| Token 加权 Prompt 命中率 | 94.60% | 93.72% | 85.77% | 91.83% | 96.13% | 96.16% |
| 平均实际未命中 Prefill / agent | 30,528.53 | 34,796.63 | 76,880.71 | 45,637.23 | 21,051.23 | 21,643.66 |
| 平均理想必要增量 / agent，不计 page 边界 | 20,141.51 | 20,053.25 | 19,860.61 | 20,194.81 | 19,698.55 | 20,281.21 |
| 平均全命中 Prefill / agent，含 64-token 边界 | 21,504.28 | 21,411.55 | 21,188.55 | 21,549.55 | 21,051.23 | 21,643.66 |
| 平均超出全命中值的 Prefill / agent | 9,024.26 | 13,385.09 | 55,692.16 | 24,087.68 | 0.00 | 0.00 |
| 平均 Decode / agent，含 reasoning | 13,382.52 | 13,158.66 | 13,071.52 | 13,442.61 | 13,731.74 | 13,451.97 |
| 总模型输出 tokens | 6,691,261 | 6,579,330 | 6,535,760 | 6,721,305 | 6,865,869 | 6,725,985 |
| 总实际未命中 Prefill tokens | 15,264,266 | 17,398,317 | 38,440,355 | 22,818,613 | 10,525,615 | 10,821,829 |
| 总全命中 Prefill tokens，含 64-token 边界 | 10,752,138 | 10,705,773 | 10,594,275 | 10,774,773 | 10,525,615 | 10,821,829 |
| 原始请求记录中的 retractions | 0 | 0 | 0 | 0 | 0 | 0 |

本表的 request/agent 指一条完整 SWE 任务（可能含四十多次模型调用），不是单轮 HTTP
请求。实际 Prefill 为引擎原始记录的 `sum(prompt_tokens - cached_tokens)`。
理想值改为**每轮可恢复历史全部命中时还必须计算的增量**，不再将各轮完整 prompt
累加称作“理论 Prefill”。旧的全历史累加量只保留在逐题 CSV 的
`cumulative_full_prompt_tokens` 字段中，用于审计命中率分母。

理想必要增量包括初始 prompt、后续工具结果、重新序列化的可见回复/工具命令及
chat-template 增量；不是仅对原始 tool 字符串单独 tokenize。当前 harness 删除旧
reasoning，所以命令在原 reasoning 之后生成的 KV 也不能直接接续复用。
因此，“全命中”不等于只计算 tool 文本，更不等于把 Decode 总长度直接当作 Prefill。

本模型的 `<think>\n` 开头实测为两个 tokens。当前融合代码的稳定 checkpoint
长度为 `floor((上一轮 prompt_tokens - 2) / 64) * 64`。在这六组未发生历史压缩的
固定轨迹上，首轮计完整 prompt；之后每轮计“当前 prompt 减去上一轮稳定 checkpoint”。
不做 page 对齐的理想必要增量使用 `上一轮 prompt_tokens - 2` 作为可复用长度。
例如 R6 每题理想必要增量 19,698.55，加上边界重算 1,352.68，等于实际 21,051.23。
Colocated 的理想值也统一按这个 **64-token 稳定前缀反事实**计算，不意味着它的
原生 Mamba 缓存实际采用同样的 checkpoint 保留策略；其超额量不能全部归因为驱逐。
该反事实按每个 agent 独立保留前缀，不假设跨任务共享。少量 colocated 单题的
actual-minus-ideal64 可以为负（本次最小 -128 tokens），不能裁成零；它不是硬下界。

六组各 500 条逐题明细及总数校验见
[增量 Prefill 统计](new-method/qwen35-incremental-prefill-accounting-20260911/)，
每组 `per_agent_incremental_prefill.csv` 给出 actual/ideal/page64/excess/decode tokens。
R5 的超额总数为 12,043,840，与显式重算旧前缀日志一致；R6/R9 的超额总数均为 0，
R9已进一步逐题核对500行，所有题的 actual-minus-ideal64 都为0，而非正负相抵。
原始 harness 汇总中的 `cached_input_tokens=0` 是 Chat usage 缺少该字段，
**不是没有缓存命中**，本文不使用该字段判断复用率。

SWE harness 会删除旧 reasoning，因此新方法回传的是可恢复的稳定 Prompt 前缀
Attention KV 和对应 Mamba checkpoint，回复及工具结果构成的新增 suffix 仍需计算。
“成功回传”不等于整条轨迹可以完全免 Prefill。

## Colocated 与新方法固定中段特征

单独汇总的 [中段吞吐、并行和缓存池对比表](SWEBENCH_QWEN35_9B_TP1_MIDWINDOW.md)
补充了窗口内活跃 agent 数，以及活跃缓存池占比与整卡 HBM 占用的区别。

取每组首题开始后的 **300–1500 秒**，按每个 endpoint 的 GPU token 计数器差分，
以及时间加权 running/KV 采样统计；不把多个 endpoint 的缺失采样直接相加差分。
这是全量测评的事后共同窗口，**不是另做一轮 300 秒预热 + 1200 秒持续补充任务的
稳态验收实验**，也不能保证各组窗口内恰好是相同任务集合。

| Run | P compute 总吞吐 | Decode 总吞吐 | Decode / 计算 D 的 GPU | Running / 计算 D 的 GPU | P Forward / P GPU | D Forward / D GPU |
|---|---:|---:|---:|---:|---:|---:|
| Colocated c128 | 6,155 token/s | 2,305 token/s | 288 token/s | 13.75 | 14.13% | 79.92% |
| Colocated c256 | 9,953 token/s | 3,661 token/s | 458 token/s | 26.66 | 21.04% | 73.01% |
| Colocated c500 | 18,739 token/s | 2,593 token/s | 324 token/s | 27.91 | 31.27% | 62.13% |
| 新方法 R5 c256 | 10,893 token/s | 2,351 token/s | 392 token/s | 16.45 | 94.18% | 97.20% |
| 新方法 R6 c500 | 5,982 token/s | 4,146 token/s | 691 token/s | 31.79 | 73.52% | 97.31% |
| 新方法 R9 c500，H2D=4/P | 6,173 token/s | 4,059 token/s | 677 token/s | 30.85 | 71.92% | 97.23% |
| 新方法 R11 c500，H2D解耦 | 5,893 token/s | 3,995 token/s | 666 token/s | 30.89 | 67.74% | 95.02% |

Colocated 的分母是 8 张同时执行 P/D 的 GPU；R5/R6/R9 的 P 分母为 2 张、D 分母为 6 张。
不能只看 D 单卡吞吐而忽略整机用了多少 P GPU。
GPU `prefill_compute` 包含 page64 计数对齐，不能替代上表精确未命中 token 数。

| KV 指标 | Colocated c128 | Colocated c256 | Colocated c500 | 新方法 R5 c256 | 新方法 R6 c500 | 新方法 R9 c500，4槽/P |
|---|---:|---:|---:|---:|---:|---:|
| 原生 Attention 活跃/不可驱逐占比 | 28.55% | 51.91% | 38.06% | — | — | — |
| 原生 Attention 总驻留占比（含可驱逐 prefix） | 40.67% | 63.66% | 78.17% | — | — | — |
| 新方法 P / D `full_token_usage` 平均值 | — | — | — | 19.14% / 29.15% | 12.52% / 50.76% | 18.68% / 51.02% |
| Mamba 活跃状态占比 | 12.78% | 24.58% | 26.36% | P 17.47% / D 18.03% | P 11.35% / D 33.17% | P 19.08% / D 32.46% |

KV 指标保留各版本的采样口径，不把原生总驻留、活跃 KV 和新方法 worker 的
`full_token_usage` 混为一个“显存总利用率”；这些都是各自池的比例，不是整卡 HBM。
已核对融合源码：`full_token_usage` 也扣除了 Radix 可驱逐页，属于活跃/不可驱逐
口径，而非全部物理驻留；具体定义及单独对比见上述中段表。
Mamba 活跃占比也不等于包含全部可驱逐快照的状态池总驻留占比。
c500 存在部分缺失采样，其时间加权 gauge 是跨缺口估计，详见原始分析的
`scrape_diagnostics`。

R5 全程 GPU counter 口径为 P compute **6,417.69 token/s**、D **1,838.17 token/s**；
P/D Forward 平均 **57.47% / 86.70%**，D running **12.80/卡**，
P/D `full_token_usage` **11.04% / 22.22%**。全程包含逐渐排空的尾部，
不能把这些值与早期单分钟峰值当作同一口径比较。

R6 全程 GPU counter 口径为 P compute **4,162.37 token/s**、D **2,551.92 token/s**；
P/D Forward 平均 **46.14% / 90.42%**，D running **19.18/卡**，
P/D `full_token_usage` **7.94% / 31.18%**，Mamba 活跃 **9.09% / 20.25%**。
中段 D prealloc/transfer 平均 **0.120 / 0.109 个/卡**；P 原生计算等待队列
平均 **0.020 个/卡**，不包含尚未通过 agentic 恢复准入的元数据等待队列。

R9 全程 GPU counter 为 P compute **3,156.51 token/s**、D **1,849.50 token/s**；
P/D Forward **33.58% / 68.40%**，D running **13.87/卡**，P/D Attention 活跃
**9.05% / 24.05%**、Mamba 活跃 **11.03% / 15.07%**。全程长尾显著拉低平均值。
中段 D prealloc/transfer **0.105 / 0.108 个/卡**，P 原生队列 **0.067 个/卡**。

## PD 新方法状态说明

本文件不沿用 27B 历史报告中的固定失败重算定义。R5 的实际配置是：

- 原生 HiCache、Mooncake 关闭；使用自定义 NIXL Direct + Shared Host Arena。
- 工具阈值 1 秒；快工具 Direct admission/建链共用 1 秒 deadline。
- 快工具 Direct 失败后，非拥堵时走 Host Slow；拥堵时显式完整重算。
  Q 只统计工具已返回、Host durable、尚未进入 H2D worker 的唯一 parent snapshot。
  连续两次 Q≥32 进入拥堵，Q≤8 退出；没有关闭重算出口。
- 每 P 的 D→P Host Arena 128 GiB、P→D Host Arena 32 GiB；H2D 并发 2。
  D transfer cap=8；P→D late binding；未修改已有物理 ownership/fence 协议。
- 请求拥有稳定的 Mamba checkpoint；D 仍保留原生 Radix 副本，单请求通常占
  active、private checkpoint、Radix checkpoint 共 3 个状态槽。
- `SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION=true`：引擎不再凭输出中任意位置的
  `TASK_COMPLETE` 提前丢弃父 KV，由现有 harness 的 tool/final ACK 决定。
  这是显式启用的引擎处理，不是修改 harness 或增加模型对话。

R4 的 260 条完成记录（其中 76 题 resolved）只用于诊断，不能把 76/260 当作
与完整 500 题可比的正确率。R4 发现合法 `echo "TASK_COMPLETE"` 等 shell 命令
被引擎误判终止，导致下一轮等待父路由 600 秒。更早的 R2 因 resident admission
死锁停止，R3 在服务启动阶段端口冲突；这些也不填入正式结果。

R5 通过 542 项 CPU 测试及独立审核后运行；全量日志未发现该父路由超时或旧
resident admission 死锁。20,563 次成功复用的 P 调用，其 `cached_tokens`
均与父 snapshot 的可恢复长度一致，额外历史前缀 miss 为 0。
这是复用长度核对，不等同于对全量输出做逐 token 的确定性参考验证。

**尚存问题：** R5 仍有 3 条环境失败（2 条 TimeoutError、1 条 ProcessLookupError）；
90 题因单轮输出达到上限结束，159 题达到 64 轮。结束前还有 38 个已经结束的
snapshot 留在 Host，约 **29.27 GiB**，对应 final ACK 晚于 offload 的既有回收窗口。
不能宣称 Host 生命周期已完全做到终止后立即释放。

### R6：c500，关闭主动重算

R6 保持 R5 的物理传输/状态机和配置，仅将并发改成 500，关闭
`SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE`，并显式保持
`SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=false`。失败 Direct 走 Host。
HIGH/LOW 记录为 32/32，但策略关闭后不参与判定；它不是队列容量，也不是
“相等阈值就禁止重算”。启用拥塞策略时，原验证器要求 low < high。

本轮修改仅为启动参数化和记录，538 项 agentic CPU 测试通过，独立启动审计 GO。
引擎 diff SHA256 与 R5 相同：`33d761fa5155a15240d1d8396d06eaea458dd09de00a27eacfd14ce6e187e653`。

最终 498 completed、2 failed、145 resolved。两条失败为环境 TimeoutError
（`django__django-13109`、`django__django-13590`），无 verifier 基础设施错误。
终止原因：174 max_turns、100 task_complete、88 max_tokens_per_turn、
102 repeated_command_outcome、30 no_command、4 command_timeout、2 environment_error。
HTTP 重试共 17 次；P 原始日志另有 **355 条 waiting-queue abort**，均无 prompt
token 计数且 completion_tokens=0，不计为额外模型调用/重算；22,353 条 P finished
记录中有 21,998 条实际调用，与 D 和轨迹轮次对应。不能宣称本轮零取消/零错误。

最终 Host 残留 **89 个 snapshot、69.879 GiB**，全部已有 final 标记，仍是迟到
final ACK 后未及时清理的已知问题。全程未观察到主动重算、Host 驱逐或 retraction。
平均工具时间/agent **50.08 秒**；shell 共 **21,778 次**；Docker 启动平均
**17.96 秒**（P90 31.62 秒），verifier 平均 **15.47 秒**。500 个镜像均为预先缓存，
但并发容器创建依然有启动开销，不能将“镜像已下载”等同于“启动耗时为零”。

### R6 之后的 Host final 回收修复（不追改上述实验数据）

修改隔离融合源码 `agentic_host_staging.py`：P 在既有后台完整 ledger resync 中
有界检查 owned HOST_READY 的 final 标记。final 可以早于排队后的 Host offer，
因此以 generation 的 `tool_started_at` 校验时间，而非 Host offer 的创建时间。
只在无 recovery assignment/claim/loading/workset 时，通过已有 EVICTING CAS
接管回收；各 TP rank 物理释放并 ACK 后发布 CONSUMED，不计驱逐或重算。
传输中和已认领的 snapshot 保留原来的完成/取消路径；释放或 ACK 失败安全重试。
不在 Forward 上增加扫描，不更改 pd 安装环境或其他 agent 的 h100 源码。

独立状态机审计 GO，包含 22 个新 final 回归用例的 agentic CPU 全套 **560 passed**。
首次修复时这里只完成代码/CPU验证；现已在下述R9完成500题GPU实验，最终回收结果见R9节。
历史 89 个/69.879 GiB 保留为 R6 的观测结果；实验进程退出后物理 arena 已释放，
本修复防止今后运行中长期保留，并非删除这次留存的结果文件。

### R7：每 P 4 个 Slow H2D 槽（启动后失败，不计性能）

R7 保持 R6 的 2P:6D、c500、Q32/32（主动重算关闭）、模型、数据、harness、
采样和显存配置，仅将 `SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT` 从每 P 2 改为 4，
全系统共 8 个槽；Host-copy workers 仍为每 P 2 个，不同步增加。
同时包含上节已授权的 late-final Host 回收修复，因此相对 R6 并非严格单一代码差异。
560 项 CPU 回归、参数检查和独立审核 GO 后启动，supervisor PID `2597896`。
完整执行同一批 500 题一次，另外比较 elapsed 300–1500 秒窗口，不冒称恒负载稳态。

结果目录：`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-q32-32-20260911-r7`。
2026-09-11 17:27:01 UTC，P1 在组建 20 请求 Prefill batch 时耗尽 315 个 Mamba
状态槽（remaining=0），触发 `HybridReqToTokenPool.alloc` 断言。17:27:02 P0
访问已失败 P1 持有的 memfd 时又触发 PermissionError。最终记录 319 条基础设施
失败，其余任务未完成；本轮无可用性能或模型正确率结论。已按自有进程组和 Docker
label 清理，日志、轨迹和 control-final 保留，其他用户 GPU7 进程未触碰。

修复：只对 custom request-owned Mamba 的 P worker，在原生 COW/入 batch 前
预留完整缺失的 active/ping-pong 状态及输出 checkpoint；不足时留队，已拥有完整
状态的请求仍可推进。回传 broker 同时预留输出 checkpoint，不进入传输状态布局。
New 分块请求的首次 checkpoint 替换也预留空间，后续 chunk 从退还的旧状态复用。
取消、拒绝入批和正常完成归还未使用状态。池比例0.5和容量不变；原生、D、Qwen3
不启用此 P 准入分支。最终582项 agentic 回归通过（含22项新增测试）。

### R8 / R9 重跑记录

R8 在派发前因补丁角色判断读取尚未建立的 `Scheduler.disaggregation_mode` 退出。
修正为读取 `server_args.disaggregation_mode`，并新增 P/D/null、混合/非混合模型及
开关关闭的初始化测试。R9 经582项回归及独立审核 GO 后启动，配置同R7。
R9 supervisor PID `2895274`；独立故障监测 PID `2897950`，每30秒检查致命
Scheduler异常，经 run/启动身份校验后用 pidfd 通知对应 supervisor 有序清理。
不操作别人的 GPU 进程或容器，不修改 serving 参数。

R9目录：`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-q32-32-20260911-r9`。
新 helper 与涉及的引擎文件额外保存在该目录 `engine-source/`，配置、外部harness
和数据快照由原启动脚本保存。R9现已完成500题，supervisor及监测进程均已退出。

### R9 最终结果：4槽相对2槽的对比

业务区间为 **2026-09-11 18:07:13.906–19:07:49.154 UTC**，不含模型启动。
仍为500个不同任务各执行一次；不是持续补充请求的恒并发测试。
配置与R6相同的部分包括模型、TP、2P:6D、c500、数据/harness/采样、静态显存0.8、
状态池比例0.5、Q32/32且主动重算关闭。变化为H2D槽2→4/P，以及Host final回收、
Mamba完整准入修复。后者将incoming workset临时预算从4补齐为5个状态槽，
**总Mamba池仍315槽**；Host-copy threads仍2/P。因此不是纯槽位数单因素消融。

| 全程指标 | R6：2槽/P | R9：4槽/P＋修复 |
|---|---:|---:|
| completed / failed | 498 / 2 | 498 / 2 |
| resolved / 全500题正确率 | 145 / 29.0% | 160 / 32.0% |
| 全500题收尾 | 2,688.883 s（44:49） | 3,635.248 s（60:35） |
| T450，仅completed | 1,879.204 s（31:19） | 1,883.575 s（31:24） |
| 450题收尾，含failed | 1,872.417 s | 1,883.575 s |
| 单题执行均值 / P50 / P90 | 1,290.46 / 1,359.18 / 1,868.18 s | 1,293.87 / 1,362.26 / 1,879.20 s |
| 平均轮数 | 43.996 | 44.342 |
| 平均Decode/题，含reasoning | 13,731.74 tokens | 13,451.97 tokens |
| 平均实际Prefill/题 | 21,051.23 tokens | 21,643.66 tokens |
| 平均理想必要增量/题，不含page边界 | 19,698.55 tokens | 20,281.21 tokens |
| 平均page64边界重算/题 | 1,352.68 tokens | 1,362.44 tokens |
| 平均超出全命中page64值的Prefill/题 | 0 | 0 |
| Token加权Prompt命中率 | 96.13% | 96.16% |
| 总实际Prefill / 总Decode | 10,525,615 / 6,865,869 tokens | 10,821,829 / 6,725,985 tokens |
| 精确实际Prefill总数 / 全程墙钟 | 3,914.49 token/s | 2,976.92 token/s |
| 精确Decode总数 / 全程墙钟 | 2,553.43 token/s | 1,850.21 token/s |
| 全程GPU counter P / D总吞吐 | 4,162.37 / 2,551.92 token/s | 3,156.51 / 1,849.50 token/s |
| 全程P / D Forward平均每卡 | 46.14% / 90.42% | 33.58% / 68.40% |
| 全程D running平均每卡 | 19.18 | 13.87 |
| 全程P / D Attention活跃池占比 | 7.94% / 31.18% | 9.05% / 24.05% |
| 全程P / D Mamba活跃池占比 | 9.09% / 20.25% | 11.03% / 15.07% |
| 工具累计耗时/题均值 | 50.08 s | 58.27 s |
| Docker容器启动均值 / P90 | 17.96 / 31.62 s | 15.40 / 28.19 s |
| Verifier均值 | 15.47 s | 14.38 s |
| HTTP重试日志条数 | 17 | 20 |
| 原始retraction计数 | 0 | 0 |

精确Prefill从原始P请求日志计算，GPU counter另有page计数对齐，不能混为同一口径。
R9有按小时轮转的`.log.YYYY-MM-DD_HH`文件：离线统计工具已修复为读取`.log*`
并按rid去重；只读当前`.log`会漏掉几乎全程数据。逐题结果见
[R9增量统计CSV](new-method/qwen35-incremental-prefill-accounting-20260911/pd-r9-c500-h2d4/per_agent_incremental_prefill.csv)。

| 同一中段300–1500秒指标 | R6：2槽/P | R9：4槽/P＋修复 |
|---|---:|---:|
| 活跃agent均值（窗口首→尾） | 370.23（491→188） | 371.69（491→182） |
| P compute总吞吐 | 5,982.37 token/s | 6,173.23 token/s |
| D总吞吐 / 每D吞吐 | 4,146.31 / 691.05 token/s | 4,059.16 / 676.53 token/s |
| D running / 卡 | 31.79 | 30.85 |
| P / D Forward / 卡 | 73.52% / 97.31% | 71.92% / 97.23% |
| P / D Attention活跃池占比 | 12.52% / 50.76% | 18.68% / 51.02% |
| P / D Mamba活跃池占比 | 11.35% / 33.17% | 19.08% / 32.46% |
| P原生计算队列 / 卡 | 0.020 | 0.067 |
| D prealloc / transfer / 卡 | 0.120 / 0.109 | 0.105 / 0.108 |
| Host→P恢复完成数 | 9,722 | 11,917 |
| Host→P单次CUDA event均值 | 40.73 ms | 46.45 ms |
| Host→P单次I/O wall均值 | 110.64 ms | 153.79 ms |
| Host→P聚合已完成字节 / 窗口墙钟 | 3.831 GiB/s | 4.780 GiB/s |
| 可恢复后至首次H2D提交日志，匹配子集均值 | 约15.48 s | 约8.73 s |
| 同上P90 | 约28.28 s | 约19.79 s |
| D→P Direct完成数 | 5,515 | 3,449 |
| Direct + Host恢复完成数 | 15,237 | 15,366 |
| P0 / P1 Forward时间，窗口各1200s | 884.05 / 880.53 s | 867.64 / 858.35 s |
| P0 / P1 非Forward时间 | 315.95 / 319.47 s | 332.36 / 341.65 s |
| 全系统实际shell执行中，平均并发范围¹ | 14.40–14.51 | 16.28–16.38 |
| Host已durable、尚未收到下一轮请求，平均snapshot数² | 18.85 | 53.11 |
| 下一轮已到达、D→Host尚未durable，平均snapshot数 | 0.52 | 0.58 |
| 下一轮已到达＋Host已durable、尚未进入恢复I/O，平均snapshot数³ | 约125.81 | 约91.24 |
| 上行平均数的秒级日志时间精度范围 | 121.82–129.86 | 86.58–96.16 |
| 上行等待队列非空的窗口占比，近似 | 98.82% | 94.11% |
| Host→P恢复I/O等效平均在途数，总计⁴ | 约0.90 | 约1.53 |
| P已经Prefill完、处于P→D inflight，平均每P | 4.41 | 5.88 |
| P0 / P1 原生计算等待队列均值 | 0.040 / 0.000 | 0.050 / 0.083 |

新增snapshot数量均为**两张P相关路径的全系统总数**，除非明确写“每P”；
不是累计经过多少条，也不是有多少个snapshot位于P HBM。计算方法为每个snapshot
在该状态内与300–1500秒窗口重叠的秒数求和，再除1200；包括跨越窗口边界的等待。

¹实际shell执行并发与Host物理状态是不同维度，不能把各行直接相加。轨迹记录了每次
`command_seconds`，但没有完整的绝对shell起止时间；用D完成到下一轮Router到达/应用
final之间的时间括住工具执行，并对窗口交集求上下界。R9有21,774次完整时间括号，
其余160次的全部工具时长也保守计入上界；无工具时长超出括号的矛盾。范围不是统计置信区间。

²“尚未收到下一轮”**不等于都在执行工具**：还包括D结果返回、应用端解析/序列化、HTTP
提交等轮间间隙。R9约53.11个等待下一轮的Host snapshot不能写成“53个工具正在执行”。
真正shell执行的全系统平均并发只有约16.3。Host等待行只含之后实际恢复的snapshot，
另有等待final标记0.20个、final后待清理0.18个，未混入续轮等待。

³由Router日志中完整的**首次**`PD_EARLY_CLAIM_ARRIVAL`关联D durable与P I/O日志，
R9覆盖全部14,428次Host恢复，无缺失；R6覆盖全部11,686次恢复，不包含其89个final残留。
这比前述仅用最终保留arrival文件的“完成子集延迟”统计覆盖更完整，且计量对象不同。
此状态包含路由、等待恢复槽、完整Attention/Mamba workset准入、映射与CPU准备，
不能单凭该数继续细分为“纯带宽排队”或“Mamba容量排队”。

⁴I/O等效在途数使用实测`wall_ms`与窗口交集，不是DMA忙碌百分比。日志绝对时间只有
秒级精度，不能可靠拆出每个亚秒阶段的瞬时并发/P90，也不能把日志重建的瞬时峰值
当作真实lane数。P→D inflight来自原生计量：Prefill完成后到传输/释放队列退出。

#### 本轮为什么P Forward仍只有71.92%？

**不是简单的“工具还没回来，所以没活干”。**R9平均已有约91个工具返回、Host KV就绪
的snapshot等恢复，而真正进入原生P计算等待队列的只有0.067/卡。P0/P1分别仍有
332/342秒不在Forward。主要可见问题是：**有可继续的任务，但恢复前置流程未能持续
把它们变成可立即Forward的完整workset**。数据也显示约5.88个/P已算好处于P→D
交付，不能把所有P侧存量都当成尚待Prefill的工作。

恢复槽从选中请求开始，跨越workset准备、H2D和Radix交接才归还；“4槽”不意味着
4条DMA全天工作。当前P Attention/Mamba活跃约18.68%/19.08%，不支持“池持续满了”
这种解释；但总平均容量空闲，也不能证明每一次准入和交接都无阻塞。
以约2秒GPU计数器间隔观察，83.57%的P端点窗口时间同时出现“全局已就绪Host队列
至少1个、该P Forward低于95%”。这只证明排队与未满Forward并存，**不是把28.08%
空档精确归因到恢复**；它还可能包含CPU调度、batch组装/launch和其他控制交接。
当前没有逐Forward空档的CPU/CUDA关联trace，不能编造“28%中各原因分别占多少”。

另一个需要区分的量：在可关联的续轮中，D结束到下一轮到达/终止标记的总轮间占用
平均约67.14个request，其中真正shell执行约16.28个，其余约50.85个是**非shell轮间
阶段**。这里包含结果返回和应用/HTTP处理，不能全部归到P scheduler，也不能全部称
工具慢。它与上面的Host物理状态有重叠，不应相加。

#### R9工具调用时间分布

下面是单次真实shell `command_seconds`，包含容器exec调用/执行等待，不包含容器创建、
模型生成或verifier。全程列覆盖21,934次；中段列按“前一轮D完成/创建snapshot在
300–1500秒”选择可关联的15,275次工具，统计其完整耗时，**不是只取中段结束的工具**。
110个终止回收snapshot的metadata已变成fallback标记，无法进入中段时间归属；
所以中段列标为可关联cohort，全程列不受该缺失影响。

| 单次工具指标 | R9同一中段，可关联cohort | R9全程 |
|---|---:|---:|
| 调用数 | 15,275 | 21,934 |
| 平均 | 1.318 s | 1.328 s |
| P50 | 0.943 s | 0.818 s |
| P90 | 1.843 s | 1.809 s |
| P95 | 2.278 s | 2.310 s |
| P99 | 5.024 s | 5.172 s |
| 最大 | 600.269 s | 600.290 s |

| 单次工具时长 | R9中段次数 / 比例 | R9全程次数 / 比例 |
|---|---:|---:|
| ≤0.5s | 307 / 2.01% | 975 / 4.45% |
| (0.5, 1]s | 8,009 / 52.43% | 12,715 / 57.97% |
| (1, 2]s | 5,789 / 37.90% | 6,574 / 29.97% |
| (2, 5]s | 1,014 / 6.64% | 1,437 / 6.55% |
| (5, 10]s | 102 / 0.67% | 162 / 0.74% |
| (10, 60]s | 39 / 0.26% | 44 / 0.20% |
| (60, 300]s | 14 / 0.09% | 21 / 0.10% |
| >300s | 1 / 0.007% | 6 / 0.027% |

中段约54.44%的工具≤1秒、92.34%≤2秒、98.98%≤5秒。可见大多数工具不长，
不能用少量600秒工具解释所有P空档。约600.3秒是600秒超时加收尾开销，不是取消了超时。
完整统计与可复现方法见
[P阶段/工具统计JSON](new-method/qwen35-incremental-prefill-accounting-20260911/R6_R9_P_STAGE_WINDOW.json)、
[只读统计脚本](../scripts/tools/summarize_swe_p_stage_window.py)。

#### 2槽与4槽横向比较（与上述P空档问题分开）

结论：**4槽改善了Slow恢复，但没有观察到D吞吐提升**。匹配子集的等待均值下降约44%，
Host→P聚合流量提高24.76%；但Slow多恢复2,195次、Direct少2,066次，两条路径合计
只多129次（0.85%）。全程shell耗时P50为0.682→0.818秒，在1秒内完成的比例
75.44%→62.41%；阈值不变时，工具时间变化会增加Slow需求。不能将路径改变全解释为
P的空间或传输槽竞争，也不能将增加的Slow处理量全视作新增有效供给。
详细关联口径、样本覆盖、量化误差和因果限制见
[中段诊断与原因分析](SWEBENCH_QWEN35_9B_TP1_MIDWINDOW.md#r94槽重跑结果)。
中段D总吞吐低2.10%，running略降，
Forward基本相同。更多Slow恢复完成不等于更多模型产出：路径比例和任务轨迹也变了，
单次H2D wall反而增加；不能据此宣称PCIe打满或槽位增加本身必然有害。
T450只相差4.37秒，单题均值/P50/P90也很接近。全程收尾多35.20%受尾部影响，
最后的`pydata__xarray-6992`累计工具执行 **2,319.27秒**（58轮），任务耗时
3,630.53秒；倒数第二题在3,114.12秒结束。不能把全程吞吐下降全归因于传输槽。
正确率多15题，但单轮并非统计显著性证据，也不能归因于缓存调度。

R9仍有2条环境TimeoutError：`django__django-13590`、`django__django-15731`；
无verifier基础设施错误。终止原因：172 max_turns、121 task_complete、87
max_tokens_per_turn、89 repeated_command_outcome、27 no_command、2 command_timeout、
2 environment_error。原始P finished共有22,557条，其中386条缺prompt计数的队列abort
不计额外调用；实际22,171次调用与D及轨迹一致。不能宣称零重试/零取消。
正式运行没有再出现Mamba分配断言或Scheduler致命异常；停止阶段SIGTERM是正常清理。

回传/释放守恒（唯一generation，TP1）：
`7,243 Direct + 14,538 Host durable + 303 app-final + 87 length = 22,171`。
Host durable与D source-release同为14,538；
`14,428 H2D恢复并释放 + 110 final主动回收 = 14,538`。
`7,243 Direct + 14,428 Host恢复 = 21,671 = 22,171 - 500首轮`。
Direct/Slow比例为 **33.25% / 66.75%**（分母21,781，含随后final回收的110个）。
未观察到主动重算、Host驱逐或retraction，逐题实际Prefill等于全命中page64理论量。
两P最终control日志均为host_ready=0、h2d_loads=0、h2d_lanes=0/4；迟到final回收
已在真实500题运行中得到验证，而非只通过CPU测试。

原始数据：[R9结果目录](/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-q32-32-20260911-r9/)，
[吞吐图](/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-q32-32-20260911-r9/pd_throughput.png)。
所有轨迹、日志、control和代码快照保留；GPU服务及监测进程已退出，GPU7其他用户进程未触碰。

## D→P 快慢路径比例

按全量运行中唯一 request-generation 去重，TP=1 不涉及多 rank 重复计数。

| 方法 | 配置 | Direct 完成 | Slow Host durable | 显式重算 | Direct / Slow / 重算 |
|---|---|---:|---:|---:|---:|
| 新方法 R5 | 2P:6D，c256，Q32/8 | 14,348 | 6,253 | 1,001 | 66.42% / 28.95% / 4.63% |
| 新方法 R6 | 2P:6D，c500，关闭主动重算 | 9,812 | 11,775 | 0 | 45.45% / 54.55% / 0% |
| 新方法 R9 | 2P:6D，c500，H2D=4/P＋修复 | 7,243 | 14,538 | 0 | 33.25% / 66.75% / 0% |
| 新方法 R11 | 2P:6D，c500，H2D解耦 | 6,196 | 15,039 | 0 | 29.18% / 70.82% / 0% |
| Colocated 三组 | c128/c256/c500 | — | — | — | 不适用 |

R5 的比例分母为三个出口的 21,602 个 snapshot；不含 372 次应用确认后直接终止释放
和 90 次 length 终止。Slow 中包含 38 个随后结束但未及时清理的 Host snapshot，
因此不能把它全称为“成功恢复到 P”的请求。

全量出口守恒：`14,348 + 6,253 + 1,001 + 372 + 90 = 22,064`，
与 D `request_seen` 唯一 generation 数一致，五组互不重叠。
Slow 中 6,215 个成功恢复、38 个结束前残留；因此
`14,348 Direct + 6,215 Host 恢复 = 20,563 次成功父前缀复用`。

R6 路径比例分母为 21,587；出口守恒为
`9,812 Direct + 11,775 Host + 323 app-final + 88 length = 21,998`。
D→Host durable 与 D source-release 都为 11,775；Host 恢复 11,686、结束时残留 89。
`9,812 Direct + 11,686 Host 恢复 = 21,498`，等于 21,998 次调用减去 500 次首轮。

### 重算成本与结论

全体 22,064 次调用中，1,001 次显式重算占 **4.54%**（与上面的路径出口分母不同），
却消耗 12,534,568 个 Prefill tokens，占实际未命中计算量的 **54.93%**。
其中 12,043,840 tokens 是本可复用的历史前缀，占全部实际 Prefill 的 **52.78%**。
即使整体 Prompt 命中率达 91.83%，少量长前缀重算仍能主导剩余计算开销。

相近早期 5 分钟窗口，R4→R5 重算调用占比从 4.87% 降到 3.81%，但 Prefill
完成调用数 4,207→4,168，未看到明确早期吞吐提升。R4 没跑完全量，不能推算
Q32/8 相对于 Q16/4 的全程加速比；R5 又同时修了终止判断，不是单独的 Q 消融。

在 R6 之前的完整结果中，colocated c256 全量时间最短。Colocated c500 输出长度相近但命中率下降、
实际 Prefill 增多；没有 retraction 不代表不存在可驱逐 prefix 的失效与重算。
R5 相对 colocated c256 总耗时增加 **12.07%**、正确率少 5 题；中段 P Forward
约 94%，D batch 仅约 16.5/卡。当前结果说明重算开销和回收问题仍需处理，
不能宣称新方法已取得端到端收益，也不能据此断言 PD 分离本身一定较差。

R6 相对 R5 全量收尾时间缩短 **26.45%**、中段 D 吞吐提高 **76.38%**，
平均实际 Prefill 降至 **21.05k tokens/agent**；但正确率从 30.6% 降至 29.0%，
不能忽略少通过 8 题。相对 colocated c256，R6 全量收尾时间缩短 **17.58%**，
正确率少 13 题。R6 同时提高并发并关闭重算，单轮结果不支持单因素因果或显著性结论。
全量更快但单题执行均值/P50高于 R5 不矛盾：c500 更早同时接纳所有任务，
更多内部等待计入单题执行时间，而不是留在 c256 的任务池准入等待中。

## 结果来源

- Colocated 三组汇总及精确 token/gauge 统计：
  [并发对比报告](baseline/swe-qwen35-9b-c128-c256-c500-comparison-20260910/COMPARISON.md)，
  同目录 `c128/c256/c500/token_accounting_summary.json` 和逐题 CSV。
- Colocated 原始结果：
  [c128](baseline/baseline-qwen35-9b-tp1-swe-verified500-colocated-c128-20260910-r1/)、
  [c256](baseline/baseline-qwen35-9b-tp1-swe-verified500-colocated-c256-20260910-r2/)、
  [c500](baseline/baseline-qwen35-9b-tp1-swe-verified500-colocated-c500-20260910-r1/)。
- 新方法 R5 原始结果（a10 本地，不是 Git 中的实验数据）：
  [结果目录](/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-appfinal-q32-20260911-r5/)，
  包含 `requests.completed.jsonl`、`raw/`、`engine_metrics.jsonl`、
  `control-final/`、`preflight.json`、`engine.patch` 和完整来源快照。
  [吞吐图](/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-appfinal-q32-20260911-r5/pd_throughput.png)。
- R4 与 R5 修复/早期窗口记录：
  [终止判断与 Q32/8](new-method/qwen35-request-owned-mamba-20260910/APP_TERMINATION_Q32_FIX.md)。
  该文早期“尚在运行”记录是历史状态，以本文件的全量结果为准。
- 新方法 R6 原始结果（a10 本地）：
  [结果目录](/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-no-policy-recompute-q32-32-20260911-r6/)，
  [吞吐图](/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-no-policy-recompute-q32-32-20260911-r6/pd_throughput.png)。
  复现时向同一新方法 launcher 传入 `MAX_INFLIGHT=500`、
  `PD_FUSED_CONGESTION_RECOMPUTE=false`、`PD_FUSED_CONGESTION_HIGH=32`、
  `PD_FUSED_CONGESTION_LOW=32` 和新的独立 `RUN_DIR`。
- 可复用入口：
  [colocated launcher](../scripts/baseline/run_qwen35_9b_tp1_swe_verified_500_colocated.sh)、
  [新方法 launcher](../scripts/new_method/run_qwen35_fused_swe500_2p6d.sh)、
  [固定 workload 配置](../configs/experiments/swe_bench_verified_miles_pr51_8k_t64.yaml)、
  [外部 harness](../data/swe_bench_openenv/harness.py)。
- 新增离线逐题增量统计：
  [summarize_swe_incremental_prefill.py](../scripts/tools/summarize_swe_incremental_prefill.py)。
- 新方法 R9 原始结果：
  [结果目录](/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-q32-32-20260911-r9/)，
  [逐题增量统计](new-method/qwen35-incremental-prefill-accounting-20260911/pd-r9-c500-h2d4/per_agent_incremental_prefill.csv)。
  同一 launcher 在R6参数基础上增加 `PD_FUSED_P_H2D_MAX_INFLIGHT=4`；
  必须同时使用R9保存的Mamba准入/Host回收修复，不能直接给R6代码改4槽复现。

原始自动汇总与本文可能有小的吞吐差别：自动汇总有采样区间均值，本文区分
原始完整 token 数/首末墙钟和按 endpoint 插值的 GPU counter；
`prefill_compute` 的 page 对齐开销也与未对齐的实际未命中 tokens 不同。
没有改写原始日志、harness 自动汇总或 27B 报告来消除这些口径差异。
