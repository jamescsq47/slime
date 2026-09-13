# SWE-bench Verified · Qwen3.5-27B · TP=2：新一轮对照实验

更新：2026-09-13。两组均已完整完成500题：Colocated c64通过308题（61.6%）；新方法 **4P:4D、c128、R6通过297题（59.4%），T500=6,428.21秒，监护进程退出码0**。R6业务窗口为01:39:57–03:27:05 UTC，不计模型加载与Host预注册。R1–R4定位并修复NIXL/UCX、跨P Host就绪、TP提交顺序及取消指令残留问题；R5发现融合版缺少Qwen3.5-27B的GDN split-view连续化修复，201题后停止。R6只补上与baseline相同的两行`.contiguous()`，此前完成523项回归、GPU数值正/负对照及独立审核。R1–R5仅作诊断，不混入结果。

## 对照原则与公共配置

先完成 baseline-colocated，再测试带 Attention KV + Mamba checkpoint 回传的新方法。公共配置必须一致，不使用旧轮次数字填表。SWE 为固定顺序的完整 500 题有限测评，不是重复补题的闭环压测；另截取业务开始后 300–1500 秒作为共同中段，不能自动称为稳态。

| 配置 | 两边共同要求 |
|---|---|
| 模型 / 数据 | 本地 Qwen3.5-27B；SWE-bench Verified 500 个不同 instance，记录数据 SHA256 |
| 硬件 / TP | a10，8 张 A100；TP=2；GPU 组 `0,4;1,5;2,6;3,7`，每组 NUMA `0,1` |
| Agent 总并发 | Colocated **64**；新方法按用户最新要求 **128**；完成一个补一个，500题耗尽后自然drain，不重复题目；不是等并发对照 |
| 静态显存比例 | 每张 GPU `mem_fraction_static=0.80` |
| Attention / Sampling backend | `triton` / `flashinfer` |
| TP 通信 / 确定性 | **允许 custom all-reduce；关闭 deterministic inference**；不强制 NCCL tree；实际是否启用须核对启动日志 |
| Page / Mamba checkpoint 间隔 | 64 / 64 tokens |
| Prefill chunk / 每批 Prefill 上限 | 8,192 / 8,192 tokens |
| 上下文 / 单轮输出 / 累计输出上限 | 131,072 / 8,192 / 81,920 tokens |
| 最大轮数 | 64 |
| 采样 | temperature=0.6，top_p=0.95，top_k=20，min_p=0，thinking=true，seed=2026；非确定性运行不承诺逐 token 一致 |
| 外部 harness | 现有 swe_bench_openenv；Chat Completions + openai_tools；reasoning=glm45，tool parser=qwen3_coder；不改工具解析或历史整理 |
| 历史 / 工具观察 | history_messages=12（仅触发上下文压缩时使用，并非每轮只留12条）；max_observation_chars=12000 |
| Docker / Verifier | 本地预下载镜像；每任务 2 CPU / 4 GiB 上限；工具与容器启动 600 秒；inline verifier 2,400 秒、最多 16 并行 |
| logprob / 指标 | return_logprob=false；每 2 秒采样，开启 Forward device timer |
| 运行依赖 | 两边均用pd_mamba_baseline解释器的Torch2.11.0+cu128/Triton3.6.0/FlashInfer0.6.7.post3；PD通过PYTHONPATH加载独立融合引擎，不改环境包；记录两边源码差异 |

## 有意保留的方法差异

| 配置 | Baseline colocated | 新方法 PD + KV/Mamba 回传 |
|---|---|---|
| GPU 分工 | 4 个 TP=2 colocated replica | 4P:4D，即 2 个 TP=2 P 组 + 2 个 TP=2 D 组 |
| Mamba/full memory ratio | **0.9**，原生缓存逻辑 | **0.5**，request-owned checkpoint 缓存逻辑；这是明确允许的方法差异 |
| 跨轮复用 | 本地原生 Radix cache | 回传与下一轮 Prompt 匹配的 Attention KV + Mamba checkpoint |
| 原生 HiCache / Mooncake | 关闭 / 关闭 | 关闭 / 关闭；慢路径用 Shared Host Arena |
| 快路径 | 不适用 | 工具阈值 1 秒，Direct admission/握手另限 1 秒 |
| Direct 失败 | 不适用 | 转 Shared Host；关闭 Q 拥堵主动重算及固定失败重算 |
| Host / I/O | 不适用 | 每 P rank D→P 128 GiB、P→D 32 GiB；H2D 4 槽；沿用 TP 原子 fence |
| 业务启动门槛 | 模型与 router ready | **内容哈希关闭，所有 Host 预注册完成后才放行业务**；身份/长度/checkpoint/fence 校验保留 |

0.5 是引擎的 Mamba/full 分配比例参数，不是整张 HBM 的 50%；池容量与实际占用分别记录。SWE harness 会整理历史，不能把已删除 reasoning 的末尾 Mamba 状态用于新 Prompt；理想增量须按真实下一轮 token 前缀计算。

## 执行状态与复现路径

| 实验 | 状态 | 原始结果目录 |
|---|---|---|
| Colocated c64 | 已完成500/500、通过308题（61.6%），退出码0；仅并发由128降为64 | [永久归档](baseline/qwen35-27b-tp2-c64-triton-customar-20260912/)；运行目录 `/tmp/pd-persist/baseline-qwen35-27b-tp2-swe500-colocated-c64-20260912-r4-triton-customar` |
| 新方法 4P:4D c128 | R6已完成500/500、通过297题（59.4%），退出码0；修复GDN数值错误，harness/调度配置不变 | [启动记录](new-method/qwen35-27b-tp2-c128-aligned-20260912/LAUNCH.md)；`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c128-20260913-r6-gdn-contiguous` |

- Baseline：`scripts/baseline/run_qwen35_27b_tp2_swe500_mamba_baseline.sh`；监护：`scripts/baseline/monitor_qwen35_27b_swe500.py`。
- 已停止 c128 诊断轮：178题结束、119题通过；原始数据保留，不作为完整500题测评。c64记录见 [LAUNCH.md](baseline/qwen35-27b-tp2-c64-triton-customar-20260912/LAUNCH.md)。
- c128归档与验收记录：[baseline/qwen35-27b-tp2-c128-triton-customar-20260912/LAUNCH.md](baseline/qwen35-27b-tp2-c128-triton-customar-20260912/LAUNCH.md)。
- Baseline 环境：`/homes/siqic/anaconda3/envs/pd_mamba_baseline`，SGLang 0.5.14 + 最小 resumed-chunk 对齐/容量修复，不是未经修改的发行包。补丁：`patches/sglang_0_5_14_mamba_resumed_chunk_{alignment,capacity}.patch`。
- Baseline 依赖：Torch 2.11.0+cu128、Triton 3.6.0、FlashInfer 0.6.7.post3；最终以本轮 packages.txt 为准。
- PD入口：`scripts/new_method/run_qwen35_fused_27b_tp2_swe500_4p4d.sh`；引擎 `/homes/siqic/sglang-qwen35-integration`。本轮Triton、关闭确定性、Mamba=0.5、Q=32/32，并使用与baseline相同的依赖解释器；数据/workload/harness逐字节核验一致。没有修改共享pd或pd_mamba_baseline环境包；R6已跑完完整500题。
- 公共 workload：`configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml`。
- 保存完整轨迹、每轮 tokens、工具/容器/verifier 时间、结束原因、启动参数及源码快照。本文取代旧诊断文档中的配置建议。

## 完整 500 题结果

T450/T500从各轮最早`started_ts`算起，结束包含verifier；单题时间从各自准入算起。模型加载不计入。全程包含500题耗尽后的drain和长工具/verifier等待，不能用全程Forward占比代表满载阶段。PD的“每组”指一个TP=2逻辑组，不是单张GPU。

| 指标 | Colocated c64 | 新方法 4P:4D c128 |
|---|---:|---:|
| 完成 / 通过 / 正确率 | 500 / 308 / **61.6%** | 500 / 297 / **59.4%** |
| T450 / T500（全程完成时间） | 4,152.87s（1h9m13s） / 7,762.35s（2h9m22s） | 3,747.76s（1h2m28s） / 6,428.21s（1h47m8s） |
| 单题完成时间 Mean / P50 / P90 | 560.58 / 521.79 / 891.07s | 944.26 / 948.99 / 1,368.38s |
| Agent/s（完整业务墙钟） | 0.06441 | 0.07778 |
| 实际 Prefill token/s 总计 | 2,056.11 | 2,537.61 |
| Decode token/s 总计 / 每 TP2 replica 或 D 组 | 652.99 / 163.25 | 769.57 / 384.79 |
| Prefill / Decode Forward 时间占比 | 15.20% / 43.91% | P 50.98% / D 70.69%（不同GPU，不能相加） |
| Decode 活跃 Forward 吞吐 / TP2 组 | 371.81 token/s | 544.36 token/s |
| 平均 running / TP2 组 | 7.21 | D 14.02 |
| Attention KV 容量 / 平均占用 / 峰值（P、D 分列） | P/D共用池；前三组635,392、末组627,136 tokens；总占用平均37.87%、单组采样峰值78.61%；不可驱逐占用平均23.75% | P每组804,416；D两组804,416 / 796,160 tokens；**不可驱逐**平均P 20.39% / D 44.38%，峰值74.05% / 92.97%；含可驱逐缓存的总占用未导出 |
| Mamba 池槽数 / 平均占用 / 峰值（P、D 分列） | P/D共用池；前三组243、末组240槽/rank；总占用平均98.90%、单组采样峰值100%；不可驱逐占用平均12.43% | P每rank171；D两组171 / 168槽/rank；**不可驱逐**平均P 21.61% / D 31.06%，峰值99.42% / 100%；含可驱逐缓存的总占用未导出 |
| Prefix hit / retraction 次数 | 96.529%（实时token计数器）；0次（25,904次模型调用逐条核对） | 96.345%；0次（25,262次成功模型调用的P/D日志均核对） |
| 超时 / 输出格式异常 / 截断 / 达轮数限制 | 工具超时结束3题；Verifier超时1题；no_command 23题、tool_format_error 1题；单轮8k截断结束17题；达64轮236题 | 工具超时结束1题；Verifier超时1题；no_command 22题；单轮8k截断结束13题；达64轮227题；重复命令结果3题 |

## 共同中段 300–1500 秒

两组均按各自业务开始后的300–1500秒统计，不按“窗口内完成题的总tokens”替代GPU计数器。TP=2的token吞吐按逻辑组去重；Forward占比对角色内物理rank取平均。Colocated的P/D共享GPU；PD的P、D分别占4张GPU，两种Forward百分比不能直接相加或视为同一种资源分配。

| 指标 | Colocated c64 | 新方法 4P:4D c128 |
|---|---:|---:|
| 完成题数 / Agent/s | 142 / 0.11833 | 147 / 0.12250 |
| Prefill compute token/s 总计 | 3,315.78 | 4,412.17 |
| Decode token/s 总计 / 每 TP2 组 | 1,217.54 / 304.38 | 1,294.39 / 647.20 |
| P / D Forward 占比 | 25.99% / 72.95%（合计98.94%） | P 88.30% / D 99.54% |
| D 活跃时 token/s / TP2 组 | 417.27 | 650.18 |
| Running / waiting / prealloc / transfer（每组） | 13.64 / 0.030 / 0 / 0 | D 24.91 / 0 / 0.209 / 0.364；另P waiting 0.042、prefill-inflight 7.43 |
| P / D Attention KV 平均占用 | 共用同一池：总占用54.73%；不可驱逐47.46%，可驱逐7.27% | 不可驱逐P 36.80% / D 80.15%；可驱逐/真正空闲/总占用未记录，不能与baseline总占用直接比较 |
| P / D Mamba state 平均占用 | 共用同一池：总占用99.49%；不可驱逐23.46%，可驱逐76.03% | 不可驱逐P 40.59% / D 54.54%；可驱逐/真正空闲/总占用未记录 |
| Prefix hit / 额外 Prefill / retraction | 97.192%；窗口实际计算3,978,939 tokens，但相对理想增量的额外量无法精确归因；0次retraction | 96.477%；窗口实际计算5,294,606 tokens；相对理想增量的额外量未作token级核算；0次retraction |
| Direct / Slow / 显式重算 数量与比例 | 不适用 | 成功跨轮恢复：5,175（65.14%） / 2,770（34.86%） / 0；按snapshot去重，不重复计算TP ranks |
| Host-ready 待恢复数量 / Host bytes / H2D in-flight | 不适用 | 全P合计：Host已就绪且下一轮已到达、未完成HBM恢复约38.37个，其中I/O开始前约36.39个；H2D操作起止包络约1.98个。Host bytes未保存可直接积分的完整时间序列；配置D→P容量512 GiB，不是实测占用 |

## 数据长度与工具时间

“累计 Prompt”是把每轮完整输入相加；“理想必要增量”是对相邻轮真实 token 序列与可恢复 checkpoint 分析后，成功复用时仍需计算的输入，不是把几十轮 Prompt 累加。去 reasoning、历史截断、模板变化和 page 尾部重算须单列；无法可靠计算时写明缺失，不用粗减法冒充。

| 指标（完整 500 题；tokens/题） | Colocated c64 | 新方法 4P:4D c128 |
|---|---:|---:|
| 平均轮数 / 工具次数 | 51.808 / 51.286 | 50.524 / 49.986 |
| 初始 Prompt / 工具新增文本 tokens | 920.97 / 18,553.44（工具观察经12k字符裁剪后的文本；未含chat模板，末轮未必再被模型消费） | 914.97 / 18,376.25（同左口径；初始Prompt实测比baseline少6 tokens，保留差异） |
| 总 Decode / 单轮 Decode Mean、P50、P90 | 10,139.41 / 195.71、87、471 | 9,893.95 / 195.83、87、473 |
| 单轮 Prompt Mean、P50、P90 | 17,719.00、16,444、32,720.7 | 17,636.24、16,468.5、32,207 |
| 累计理论完整 Prompt/题（无复用） | 917,985.84 | 891,053.28 |
| 实际 Prefill/题 | 31,920.51（GPU实时计算计数器） | **32,624.51**（GPU实时计算计数器） |
| 理想必要增量 Prefill/题（成功复用） | 无法精确核算：未记录真实Prompt token IDs与checkpoint插入事件，重新分词一致性检查未通过；不能用累计Prompt或“初始+工具”代替 | 暂无经全量真实token前缀/checkpoint核验的数值；不能把HTTP未命中31,041.12或初始+工具19,291.22当作精确理想值 |
| 额外 Prefill/题及可解释来源 | 相对理想值暂无可靠数值；可单独核算GPU计算比请求级未命中输入多1,655.25 tokens/题，但不能将其全部归为cache淘汰或page重算 | 相对理想值未精确归因；GPU计算比成功请求级未命中多**1,583.39 tokens/题**；不能全部归为KV淘汰或page重算 |
| 请求级未命中输入/题（完整Prompt − cached_tokens） | 30,265.26；这是HTTP请求口径，不是理想必要增量，也不包含全部GPU内部重复计算 | 31,041.12；25,262次调用的P、D记录总量一致，且与harness逐题调用数/Prompt总量一致 |
| 工具总耗时/题 | 66.12s | 52.21s |
| 单次工具耗时 Mean / P50 / P90 / P99 | 1.289 / 0.540 / 1.449 / 7.955s（25,643次） | 1.044 / 0.536 / 1.371 / 7.566s（24,993次） |
| 工具耗时 ≤1s / 1–2s / 2–10s / >10s 比例 | 82.20% / 11.74% / 5.31% / 0.75% | 82.90% / 11.28% / 5.22% / 0.60% |
| 容器启动 / Verifier 时间 Mean、P50、P90 | 启动0.358 / 0.249 / 0.572s；Verifier 18.741 / 9.584 / 20.385s | 启动0.718 / 0.268 / 2.183s；Verifier 18.618 / 9.559 / 20.364s |

### 缓存池占用拆分

以下百分比均除以**对应缓存池容量**，不是除以整张HBM；按时间积分后对该角色的逻辑组取平均（colocated四组、PD每角色两组）。Colocated的Prefill/Decode共用池，不能加两遍。`full_token_usage` / `mamba_usage`表示不可驱逐占用；已缓存但可驱逐的前缀另列。PD没有导出`kv/mamba_{used,evictable,available}_tokens`组成计数，故不能由usage补出总占用或真正空闲。峰值为2秒采样观察值，不是连续监测的绝对峰值。

| 池 / 窗口 | 不可驱逐 | 可驱逐缓存 | 真正空闲 | 总占用 | 总占用采样峰值 |
|---|---:|---:|---:|---:|---:|
| Colocated Attention，全程 | 23.75% | 14.12% | 62.13% | 37.87% | 78.61% |
| Colocated Attention，300–1500s | 47.46% | 7.27% | 45.27% | 54.73% | 78.61% |
| Colocated Mamba，全程 | 12.43% | 86.47% | 1.10% | 98.90% | 100% |
| Colocated Mamba，300–1500s | 23.46% | 76.03% | 0.51% | 99.49% | 100% |
| PD P Attention，全程 | 20.39% | 未记录 | 未记录 | 未记录 | 总占用未记录；不可驱逐峰值74.05% |
| PD D Attention，全程 | 44.38% | 未记录 | 未记录 | 未记录 | 总占用未记录；不可驱逐峰值92.97% |
| PD P Attention，300–1500s | 36.80% | 未记录 | 未记录 | 未记录 | 总占用未记录；不可驱逐峰值74.05% |
| PD D Attention，300–1500s | 80.15% | 未记录 | 未记录 | 未记录 | 总占用未记录；不可驱逐峰值92.97% |
| PD P Mamba，全程 | 21.61% | 未记录 | 未记录 | 未记录 | 总占用未记录；不可驱逐峰值99.42% |
| PD D Mamba，全程 | 31.06% | 未记录 | 未记录 | 未记录 | 总占用未记录；不可驱逐峰值100% |
| PD P Mamba，300–1500s | 40.59% | 未记录 | 未记录 | 未记录 | 总占用未记录；不可驱逐峰值83.04% |
| PD D Mamba，300–1500s | 54.54% | 未记录 | 未记录 | 未记录 | 总占用未记录；不可驱逐峰值78.57% |

Mamba 总占用接近100%不等于所有状态都被运行请求锁住；中段约76%是可驱逐缓存。Attention每个TP rank的token槽占32 KiB；可用池约19.39 GiB（前三组）/19.14 GiB（末组）。Mamba每槽每rank约73.41 MiB；243/240个可用槽约17.42/17.20 GiB，另有分配器dummy槽等小额开销。

上述GiB容量为colocated。PD Attention池每rank约24.55 GiB（P两组、D首组）/24.30 GiB（D末组）；Mamba每rank约12.26 / 12.04 GiB可用槽，另有dummy槽等开销。比例0.5改变池容量分配，不能把“PD Mamba usage较低”单独视为整张GPU节约同等比例显存。

### 工具时间分布

仅统计 Agent 的 shell 工具调用，不混入容器启动、准备环境和Verifier。各任务并发执行，工具耗时求和不是业务墙钟。

| 单次工具时间 | Colocated次数 | 占25,643次 | PD次数 | 占24,993次 |
|---|---:|---:|---:|---:|
| ≤1s | 21,078 | 82.20% | 20,719 | 82.90% |
| 1–2s | 3,011 | 11.74% | 2,818 | 11.28% |
| 2–10s | 1,361 | 5.31% | 1,305 | 5.22% |
| >10s | 193 | 0.75% | 151 | 0.60% |

### 统计口径与限制

| Agent结束原因 | Colocated题数 | 占500题 | 最终通过 | PD题数 | 占500题 | 最终通过 |
|---|---:|---:|---:|---:|---:|---:|
| TASK_COMPLETE | 219 | 43.8% | 169 | 234 | 46.8% | 185 |
| 达64轮 | 236 | 47.2% | 130 | 227 | 45.4% | 106 |
| 单轮8k输出限制 | 17 | 3.4% | 2 | 13 | 2.6% | 2 |
| no_command | 23 | 4.6% | 3 | 22 | 4.4% | 4 |
| command_timeout | 3 | 0.6% | 3 | 1 | 0.2% | 0 |
| tool_format_error | 1 | 0.2% | 0 | 0 | 0% | 0 |
| 普通final_answer | 1 | 0.2% | 1 | 0 | 0% | 0 |
| 重复命令结果 | 0 | 0% | 0 | 3 | 0.6% | 0 |
| 合计 | 500 | 100% | 308 | 500 | 100% | 297 |

两组Verifier各超时1题，与上述Agent结束原因不是互斥维度。`no_command`表示该轮没有可执行命令，不直接等价于工具解析器故障；baseline 17题、PD 13题单轮截断也不能被顶层`status=completed`/`truncated=0`掩盖。PD轨迹中有22轮`structured_action_error`非空，不能因为独立的`tool_format_error`终止计数为0就声称没有格式异常。

以下五条为colocated原始核验说明；PD的独立核验在其后列出。

- 总模型调用25,904次，完整Prompt累计458,992,918 tokens；原始引擎日志与每题轨迹按调用数和Prompt总量逐题交叉核验。原始日志含小时轮转文件，不能只读最后一个`.log`。Harness顶层`model_prompt_tokens`及部分cached字段为0，不能拿这些未填充字段计算命中率。
- GPU实时计数：Prefill compute 15,960,256 tokens，Prefill cache 443,860,288 tokens，Decode 5,068,707 tokens。Prefix hit = cache / (cache + compute)。轨迹输出总数5,069,705与GPU实时Decode计数相差998 tokens（0.020%）；长度表用轨迹、吞吐表用GPU计数，保留口径差异，不强行抹平。请求级未命中输入为15,132,630 tokens；与GPU compute的827,626-token差额单列，尚未逐chunk归因。
- 理想增量必须对真实下一轮输入与此前可恢复前缀做token级核算。本轮没有发生历史压缩，但日志文本离线重新分词出现例如26,523 vs引擎26,531 tokens的不一致，不能据此给出精确LCP/page64理想值。且198题曾跨replica调用，不能把“在一个worker第一次出现”当作“该题第一轮”。这些限制不影响实测GPU compute、轨迹长度和Verifier结果。
- Forward使用device timer：对八个物理rank取平均；活跃Decode吞吐 = 全部Decode tokens / 四组累计平均rank Decode秒数。每物理卡墙钟摊销吞吐为全程81.62、中段152.19 token/s，不能将TP2拆成独立TP1能力。
- T500包含工具与Verifier长尾。最后结束的`scikit-learn__scikit-learn-14710`单题4,968.17s，其中模型调用358.11s、工具累计2,204.59s、Verifier 2,400.23s（超时）。因此全程Forward低于中段不表示满载时同样空闲；中段P+D Forward合计98.94%。本轮500个镜像均已在本地，无镜像下载冷启动，但容器仍须逐题创建。
- `completed=500`表示测评记录全部结束，不表示没有任务级问题：17题因单轮8k限制结束，236题达到64轮；另有工具/格式/Verifier异常，均保留在500题正确率分母内，不剔除。

### PD R6核验与结论

- 全量500个不同instance全部结束，通过297题。P、D各25,262次成功调用，逐题调用数与Prompt累计量均与外部harness一致，匹配差异为0；两侧日志均未报告retraction。完整Prompt为445,526,640 tokens，GPU Prefill compute为16,312,256、cache为430,006,080、Decode为4,946,973 tokens，后者与轨迹输出总量一致。实时compute比成功HTTP调用的未命中15,520,560多791,696 tokens；尚未逐chunk归因，不称为已量化的cache thrashing。
- 此外P日志有**1,449条`Abort in waiting queue`**结束记录，不含完整token统计。这些不是额外完成的题或成功模型轮次，未纳入轨迹长度分母；GPU计数器统计范围不因它们被剔除。不能将“成功调用均无retraction”解读成整个控制路径完全没有取消/重试。
- 成功跨轮恢复合计**24,762=25,262−500次**：Direct 15,086次（60.92%），Shared Host 9,676次（39.08%），数量与非首轮调用数闭合，未出现未被这两条路径覆盖的成功后续调用。慢路径D2H完成9,752个snapshot，9,676个被P消费，另外76个有最终结束回收日志，数量闭合；这不是对所有allocator内存均已回收的独立证明。没有开启Mooncake/原生HiCache，也没有改harness来绕过工具调用。
- 工具≤1秒占82.90%，但成功Direct只占60.92%：工具很快不保证在admission/握手时限内完成Direct。中段Host已就绪且下一轮已到达、等HBM恢复约38.37个；其中开始恢复I/O之前约36.39个，H2D起止包络约1.98个，**不是38个都在同时传数据**。这是全P合计、按TP snapshot去重的日志时间积分近似值；秒级日志、TP rank完成时间不同会带来误差。Host等下一轮到达的已恢复snapshot平均约1.15个，不等于全部处于工具执行中的agent数量；最终未再恢复的snapshot不在这一队列估计内。
- 中段P Forward **88.30%**，仍有11.70%非Forward；D Forward **99.54%**。本轮不能把全部P非Forward时间精确归为一种原因。此前对本轮尾段P0的只读CPU栈采样发现rank0经常在`prepare_tp_control`，rank1主要等待TP广播；源码中历史`_superseded_owners`扫描存在随历史增长的开销。这是控制面阻塞证据，但**CPU栈占比不是GPU空闲占比，尾段采样也不能代表300–1500秒**。本次仅归档结果，未修改这段运行逻辑，不能宣称已经修复。
- 相比colocated c64，PD c128的T450缩短**9.75%**，T500缩短**17.19%**；全程Decode总吞吐提高**17.85%**，共同中段总吞吐提高**6.31%**。但单题平均时间由560.58增至944.26秒，正确率由61.6%降至59.4%。500题配对结果为共同通过263、仅baseline通过45、仅PD通过34、共同未通过158。单次非确定性运行不足以将2.2个百分点差异归因为传输正确性问题，也不足以证明两者正确率等价。
- 两组平均轮数50.524 vs 51.808、单轮Decode 195.83 vs 195.71，长度分布接近但轨迹不相同；总Decode/题相差约−2.42%。并发128 vs 64、Mamba比例0.5 vs 0.9、计算卡分工不同，故以上是这两套完整配置的实测对比，不是严格等并发、等输出工作量的纯调度收益。
- PD最后一题仍是`scikit-learn__scikit-learn-14710`：单题4,332.04秒，模型400.44秒、工具1,527.67秒、Verifier 2,400.20秒超时。第499题在4,828.75秒结束，最后一题又延长全程约1,599.46秒。**T500收益包含工具/verifier尾部差异**，不能单独据此判断GPU服务能力；共同中段吞吐和T450须一起看。500题全部为本地已有镜像，无下载冷启动。

### 完成进度里程碑

单位秒，从第一题准入到第N题完成（含Verifier），不是这N题各自延迟的均值。

| 完成题数 | Colocated c64 | PD R6 c128 |
|---|---:|---:|
| 10 | 228.55 | 365.01 |
| 20 | 373.04 | 521.21 |
| 50 | 567.92 | 851.93 |
| 90 | 966.59 | 1,048.45 |
| 100 | 1,054.97 | 1,099.49 |
| 250 | 2,309.04 | 2,143.40 |
| 450 | 4,152.87 | 3,747.76 |
| 490 | 4,527.01 | 4,017.74 |
| 500 | 7,762.35 | 6,428.21 |

### 数据与复现

- [启动配置](baseline/qwen35-27b-tp2-c64-triton-customar-20260912/LAUNCH.md)、[全程与中段基础统计](baseline/qwen35-27b-tp2-c64-triton-customar-20260912/RESULTS.md)。
- [各表补充统计与逐题未命中输入](baseline/qwen35-27b-tp2-c64-triton-customar-20260912/table_metrics.json)、[轨迹/工具/Verifier分布](baseline/qwen35-27b-tp2-c64-triton-customar-20260912/swe_bench_profile_summary.json)。
- 离线复算：`scripts/tools/summarize_native_swe_colocated.py`生成吞吐/Forward基础表；`scripts/tools/summarize_native_swe_tables.py RUN --output OUTPUT.json`生成缓存组成、工具分布和逐题HTTP输入统计。仅进行离线分析，不修改引擎、harness或实验配置。
- PD：[R6逐表统计、逐题未命中输入和跨轮路径核验](new-method/qwen35-27b-tp2-c128-aligned-20260912/r6_table_metrics.json)。该JSON含全程/中段、工具分布、终止汇总、每题HTTP累计输入、计数器采样间隔及统计限制。原始`requests.jsonl`、`raw/`、`logs/`、`engine_metrics.jsonl`、`control-final/`与源码快照保留在上述R6运行目录，未删除，未把多GiB原始日志复制到仓库。
- PD离线复算：`python scripts/tools/summarize_swe_tp_pd_tables.py /tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c128-20260913-r6-gdn-contiguous --tp 2 --output runs-host/new-method/qwen35-27b-tp2-c128-aligned-20260912/r6_table_metrics.json`（在`examples/pd`目录执行）。计数器差分检查reset，gauge按时间积分；首条已产生工作量的计数器出现前按0，不能用第一条正值替代业务起点而漏计首批tokens。全程最大采样间隔3.59秒，中段3.56秒。
- 未补造的缺失项：PD缓存池可驱逐/真正空闲/总占用拆分、完整Host byte占用时间积分、两组逐token/checkpoint核验后的精确理想Prefill增量。本轮导出的usage与实际GPU compute可以可靠复算；上述缺项不写成0，也不拿代理量冒充。
