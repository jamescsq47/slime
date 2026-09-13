# SciAgentGym 原题：Qwen3-8B 实际 Agent 验证

日期：2026-09-07。结论：真实模型→上游工具→继续推理链路可运行，但本轮并未产生秒级工具调用，不能用先前工具 demo 的耗时代替实际 workload。

## 配置与来源

| 项目 | 本轮 |
|---|---|
| 模型 | `/dataset/model/qwen3/Qwen3-8B`，单卡 GPU 0，TP=1 |
| 引擎 | `pd_baseline`，SGLang 0.5.10.post1，colocated |
| 显存 | mem_fraction_static=0.80，KV pool=344,000 tokens |
| 采样 | temperature=0，top_p=1，top_k=-1 |
| 并发/样本 | c4；10 道原题各一次，非持续闭环性能测试 |
| 原始顺序 | 1,16,20,25,26,28,31,32,35,36 |
| 长度 | 单轮最大生成8192；整条新增tokens含工具32768；上下文40960，预留512 |
| 轮次/工具 | 最多10轮，每次工具60秒超时，CPU-only，每Agent独立工作目录 |
| 上游 | `CMarsRover/SciAgentGYM`，commit `e9dbbea4369d67694e38bf8be67bedbcaf9e9300`，工作树干净 |
| 数据 | `dataset/refine_merged_single_questions.json`，原始question/schema，未传入answer/golden_answer |
| 配置文件 | `examples/pd/configs/experiments/sciagentgym_offline.yaml` |

复用上游 `load_tools_for_case` 和 `GenericFunctionTool`。模型自主选择函数与参数；没有强制工具、人工 sleep 或自写科学解题函数。上游 case35 的工具本身包含符号公式功能；适配层没有注入参考答案。图像仅作为文件产物，文本模型不可见。

## 结果

| 指标 | 本轮 |
|---|---:|
| 已执行原题 | 10 |
| 输出最终回答 | 9（不代表答对） |
| 长度截断 | 1，case28第一轮达到8192 |
| Agent异常失败 | 0 |
| 实际使用工具的题目 | 6/10 |
| 模型调用轮次 | 17，平均1.7/题 |
| 模型生成tokens | 49,427，平均4,942.7/题 |
| 工具及模板追加tokens | 3,371，平均337.1/题 |
| 工具调用 | 10次，6次成功返回、4次错误观察 |
| 工具执行平均耗时 | 0.023744秒（包含错误调用） |
| 工具执行最大耗时 | 0.206257秒 |
| 工具RPC平均/最大耗时 | 0.024146 / 0.206642秒 |
| 超过1秒/2秒的实际工具调用 | 0 / 0 |
| 业务总历时 | 207.029秒，从首Agent开始至最后Agent结束，不含模型启动 |
| 平均Agent执行时间 | 62.084秒，不含信号量等待 |

模型生成tokens按 `/generate` 原始 `output_ids` 精确统计；工具tokens按实际追加suffix统计。整轮生成token数/业务墙钟约238.7 token/s，但本轮只有c4且包含启动填充与排空，**不是满载/稳态吞吐结论**，不与既有300+1200秒实验对比。通用 `summary.json` 中 first-turn TTFT及由请求元信息推导的TPOT存在既有不适用字段（TTFT=0、TPOT异常大），本报告不采用它们；不将自动命名的 `steady_state` 子窗口视作正式验收。

| 原题ID | 状态 | 模型轮次 | 生成tokens | 工具调用 | 备注 |
|---:|---|---:|---:|---:|---|
| 1 | 最终回答 | 2 | 5,697 | 1 | 成对产生阈值计算成功 |
| 16 | 最终回答 | 1 | 3,280 | 0 | 模型直接推理 |
| 20 | 最终回答 | 3 | 6,675 | 4 | 4次工具错误后自行回答 |
| 25 | 最终回答 | 2 | 6,444 | 1 | 调用快速旋转参考系变换，未调用慢ODE |
| 26 | 最终回答 | 2 | 1,697 | 1 | 哈密顿量期望值计算成功 |
| 28 | length截断 | 1 | 8,192 | 0 | 无最终回答 |
| 31 | 最终回答 | 2 | 2,916 | 2 | 自旋态计数成功 |
| 32 | 最终回答 | 1 | 4,009 | 0 | 模型直接推理 |
| 35 | 最终回答 | 2 | 2,485 | 1 | 上游符号EIF推导，0.206秒 |
| 36 | 最终回答 | 1 | 8,032 | 0 | 模型直接推理 |

## 问题与解释

1. case20两次 `construct_state_vector` 返回了复数，上游 `GenericFunctionTool` 的JSON序列化报 `Object of type complex is not JSON serializable`；另两次 `comprehensive_entanglement_analysis` 由模型提供 `[real,imag]` 数组，原函数处理后reshape不匹配。错误原样作为工具观察返回。没有修改上游函数或伪造成功结果。
2. 之前CPU demo中约12.2秒的case25 `evolve_spin_state`、约4秒的case36 `analyze_bandwidth_scaling`，本轮模型都没有调用。case25选用更快的解析变换函数；case36直接进行数学证明。因此“上游有慢工具”不等于“真实模型轨迹自然会调用慢工具”。
3. 上游case36原始题是相关高斯检验下Holm FWER的证明，但其附带工具为流形/图拉普拉斯谱分析。这种题目与工具关联不紧密的现象来自原始数据，适配器没有改配工具。选慢计算benchmark不能仅依据该case的工具demo。
4. 未运行官方grader，9条final不是90%正确率。case25还自行选择了数值参数，答案是否满足原题的符号推导要求需要另行判分。

## 审核和清理

- 运行前独立审计GO；baseline相关测试51 passed，PD生命周期回归11 passed，Shell语法通过。
- 审计发现并修复了上游wrapper二次解包嵌套arguments可能绕过文件参数检查的适配错误；补正常、取消、超时、路径穿越、嵌套参数回归。
- 此新增harness只允许colocated，没有修改PD allocator、router、transport或TP状态机。
- 精确初始token、各轮模型token与工具suffix保存在 `requests.jsonl` 的 `metadata.science_token_trace`；原始题、最终回答、工具参数、观察与执行时间同样落盘。
- 模型、router与科学worker已退出；GPU0回到启动前599 MiB，没有遗留本轮GPU进程。
- Python audit hook是已审查工具的纵深防御，不是任意代码OS沙箱；父进程遭SIGKILL的孤儿边界尚未增加PDEATHSIG。

原始证据：`requests.jsonl`、`dispatch_sequence.json`、`config.json`、`resolved_workload.json`、`engine_metrics.jsonl`、`logs/model-0.log`。不要将CPU demo结果与本轮实际工具调用混合统计。
