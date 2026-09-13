# DABstep + Qwen3-8B：六道真实开发集原题

运行日期：2026-09-07至2026-09-08 UTC。结论：修正动作边界后原题工具执行可运行，但本轮没有秒级Python调用，Qwen3-8B在这个CodeAgent配置下的数据分析质量较低；不能将本轮作为成功的慢工具负载。

## 数据和实际配置

- 官方数据：https://huggingface.co/datasets/adyen/DABstep ，revision `353c0c43533e06443d14dd8b67add008567d072e`。
- 官方baseline：https://huggingface.co/spaces/adyen/DABstep/tree/main/baseline ，revision `d4431c2e4a695cbe43c33aab2adaa304a37ae64a`。
- 使用dev原始顺序前6题：5、49、70、1273、1305、1464，3 easy+3 hard。未依据工具耗时筛题。原题、guidelines均未改，答案不进入模型或执行容器。
- 输入为官方7个CSV/JSON/Markdown文件，约24.2MB；payments.csv为138,236行、21列。没有放大数据或插入sleep。
- Qwen3-8B，GPU0，TP1，colocated，mem_fraction_static0.80；context40960；temperature0、top_p1、top_k-1；每次生成最多8192 tokens。
- 同时2个Agent，每Agent独立持久Python容器，最多10个CodeAgent步骤，每次工具超时60秒。没有使用SciAgentGym的整条32768新增token限制；本轮没有触发工具超时或步骤上限。
- 复用smolagents1.24.0 `CodeAgent`和官方task prompt。执行器替换为只读输入、无网络、无主机目录的Docker，使用已有slimerl/slime image，无需下载每题Docker；image精确ID见config.json。每容器2CPU配额/4GiB内存，BLAS1线程，pandas3.0.1/numpy1.26.4。
- agent依赖只安装在 `/homes/siqic/.venvs/dabstep_smoke`；未更改`pd_baseline`或`pd`依赖及SGLang源码。

## 结果汇总

| 指标 | 本轮 |
|---|---:|
| 原题数量 | 6 |
| CodeAgent调用final_answer结束 | 6；其中1条空答案 |
| 官方question_scorer判对 | 1/6，16.7%（仅小样本dev，不是全榜结果） |
| 模型调用总轮次 | 23，平均3.83/题 |
| 模型生成tokens | 16,410，平均2,735/题 |
| 累计模型输入tokens | 64,570 |
| 实际Python执行 | 20次 |
| 正常返回/执行异常 | 16/4 |
| 含final_answer的调用 | 6；其中部分还包含其他代码 |
| 不含final_answer且成功的调用 | 10 |
| 全部20次Python平均/最长时间 | 0.120012 / 0.799441秒 |
| 成功非终止调用平均/最长时间 | 0.121569 / 0.737446秒 |
| 超过1秒/2秒的Python调用 | 0/0 |
| 工具超时、容器清理错误 | 0、0 |

执行耗时来自容器内perf_counter，涵盖Python代码、imports与文件读取，不包括容器创建或模型启动。父进程RPC wall另有记录。最长0.799秒的调用读取CSV后因不存在的account_type列报错；最长成功非终止调用0.737秒涉及pandas导入、CSV读取及统计。不能把错误调用、结束函数或初始化开销当作科学重计算。

| Task ID | 难度 | 模型调用轮次 | 生成tokens | Python次数 | 答案 | 官方评分 |
|---|---|---:|---:|---:|---|---|
| 5 | easy | 3 | 2610 | 3 | Not Applicable | 错 |
| 49 | easy | 6 | 3496 | 5 | A. NL | 错 |
| 70 | easy | 1 | 675 | 1 | Not Applicable | 对 |
| 1273 | hard | 3 | 1863 | 2 | Not Applicable | 错 |
| 1305 | hard | 4 | 3123 | 4 | Not Applicable | 错 |
| 1464 | hard | 6 | 4643 | 5 | 空字符串 | 错 |

评分在所有模型任务完成后离线运行上游原始 `dabstep_benchmark/evaluation/scorer.py::question_scorer`；未调用评分API或提交排行榜。唯一评分正确的case70也值得保留警惕：模型代码把`relevant_data=False`写死，并未充分检查数据，其Not Applicable恰好匹配参考答案。不能把它描述为完整正确的数据分析过程。

## 实际问题

1. Qwen会猜`transactions.csv`、`account_type`等文件名/字段，而不是先充分探索文件与schema；遇错后有时直接放弃。
2. 部分模型代码自行写了“simulated”假设和费用常数。这是模型输出的问题，**不是适配器生成了模拟数据或替代答案**。原始输入始终是真实官方benchmark文件。
3. 出现漏import、缩进错误、未知列，以及未产生合法code block的解析错误。错误会返回给Agent再推理，没有静默伪造成功结果。
4. 数据本身只有约24MB，实际成功计算主要是读取和简单聚合。当前小样本不能证明整个DABstep没有慢操作，但没有看到稳定的秒级自然工具执行，不宜直接扩大为慢路径serving负载。

## 接入修正与有效性边界

- r1未进入任务，启动器PATH缺少baseline/bin，ninja无法找到；添加和现有启动器相同的PATH处理后恢复。
- r2忽略了CodeAgent停止标记，允许同一生成中出现代码→模型编造的Observation→后续代码，故不采用该轮结果。
- r3先移除完整Qwen reasoning，再应用CodeAgent原始停止标记，只将第一个action交给Python。只有实际执行结果进入下一轮观察；未闭合reasoning不允许执行。新增回归测试覆盖该边界。
- 为避免停止标记截断reasoning，目前在生成后本地应用该边界。raw_content和token计数仍包含截断后未使用的生成文本，因此本报告不比较LLM吞吐、TPOT或KV复用。
- 这是官方数据和CodeAgent的本地适配，并非原始官方baseline逐项配置复现。不是300+1200秒正式性能测试，也尚未注册进PD workload harness。
- 9项真实Docker/故障/动作边界测试通过；每轮修改均经独立审计GO。PD的ownership、router、allocator和TP逻辑未改动。
- 本轮6题JSON保留原题、guidelines、原始模型回复、执行action、真实工具输出和双重计时，config保留输入hash。

本轮结束后模型和所有带本轮label的容器已退出，GPU已释放。
