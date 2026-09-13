# BFCL V4 Web Search：无密钥 Colocated 接入测试

日期：2026-09-07。状态：300 秒预热 + 1200 秒测量完成，模型及实验进程已退出。

## 配置及修改范围

| 项目 | 设置 |
|---|---|
| 模型 / 引擎 | Qwen3-8B / pd_baseline 环境 SGLang 0.5.10.post1 |
| 硬件 / 部署 | 单张 A100 80GB，GPU 0，TP=1，Colocated |
| mem_fraction_static | 0.80 |
| 并发 | 8 条端到端 Agent，结束后补充 |
| 数据 | BFCL V4 Web Search，100 道原始题，base 条件，source-order 循环 |
| 采样 | temperature=0，top_p=1，top_k=-1，seed=2026 |
| 长度 | 单轮最多 8192 输出 tokens；总 response budget 32768；上下文 40960；最多 20 轮 |
| 工具 | DDGS auto 公网搜索 + HTTP 网页读取；不需要 API key |
| 工具限制 | search 并发 4，fetch 并发 8；配置 timeout=20s；每 Agent 第三次累计工具错误终止 |
| 实际窗口 | 预热 300.261 秒；正式测量 1200.003 秒 |

实现位于 `examples/pd/data/bfcl_web_search/`，配置位于
`examples/pd/configs/experiments/bfcl_web_search.yaml`，启动脚本为
`examples/pd/scripts/baseline/run_bfcl_web_colocated.sh`。只读取问题和工具定义，不读取答案；
真实网络调用，无人工延迟、无答案缓存。保留模型生成的 token 序列拼接工具 observation。
增加 URL/DNS 公网地址校验、严格工具参数解析、取消和后台搜索线程容量管理。
本轮只支持 Colocated；自定义 PD 生命周期显式拒绝，未修改 SGLang 传输或调度逻辑。

## 正式窗口性能

| 指标 | 结果 |
|---|---:|
| 实际 Prefill / 墙钟 | 179.4 token/s |
| Decode / 墙钟 | 185.8 token/s |
| Prefill Forward 时间占比 | 1.65% |
| Decode Forward 时间占比 | 89.54% |
| 两类 Forward 合计 | 91.19% |
| 平均 running requests | 2.23 |
| 最大采样 running requests | 6 |
| 平均引擎 queue | 0 |
| KV token pool 容量 | 344,000 tokens |
| 平均 KV 使用量 / 利用率 | 5,904 tokens / 1.72% |
| 峰值采样 KV 使用量 / 利用率 | 33,280 tokens / 9.67% |
| 测量窗口结束的轨迹 | 127 |
| 正常 final / length 截断 / 工具错误失败 | 111 / 5 / 11 |
| 窗口结束时仍在运行 | 8，结束测量后取消 |

性能取 `engine_metrics.jsonl` 中同一个 Colocated endpoint 的一份计数，避免将 P/D 角色重复相加。
Token 与 GPU Forward 累计计数在测量边界线性插值后取差值除以总墙钟时间；
running/KV 等 gauge 使用时间加权均值。不是“请求完成后计数”的平滑吞吐。
图见 [pd_throughput.png](pd_throughput.png)。低并发且大量请求等待公网，不是饱和性能测试。

## 数据特性及工具耗时

以下为测量窗口内结束的全部 127 条轨迹（包含失败与截断），其整条轨迹可能跨越预热边界；
不包含窗口结束时尚未完成的 8 条。因此不能拿本表均值乘完成数去替代上述窗口引擎吞吐。

| 指标 | 均值 / 数量 |
|---|---:|
| 模型调用轮数 / Agent | 2.94 |
| 首轮 Prompt | 843 tokens |
| 每次模型调用 Prompt | 2,536 tokens |
| 累计 Prompt / Agent | 7,470 tokens |
| 累计实际 Prefill / Agent（Prompt − cached） | 1,585 tokens |
| 模型输出 / Agent | 1,726 tokens |
| 总 response / Agent，含工具 observation 和模板 | 3,190 tokens |
| Agent 总延迟 | 74.80 秒 |
| 累计工具耗时 / Agent | 54.28 秒 |
| 没有工具调用的轨迹 | 14 |

| 工具 | 调用数 | 成功 | 平均耗时 | P50 | P90 | 最大 | 超过 1 秒 | 超过 2 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| search_engine_query | 278 | 185 | 24.76s | 26.08s | 37.55s | 43.62s | 98.92% | 96.40% |
| fetch_url_content | 11 | 8 | 0.86s | 0.38s | 2.19s | 2.79s | 27.27% | 18.18% |

工具耗时从调用到返回，包含本地 semaphore 排队、DDGS 的 provider 尝试和网络等待，
不是单次 HTTP 纯传输时延；timeout=20s 不代表含排队和 provider 内部行为的总耗时上限。
289 次调用中 96 次失败：80 次 TimeoutException、13 次 DDGSException、3 次网页读取错误。
前两次工具错误作为 observation 交回模型，可继续执行；第三次累计工具错误才终止轨迹。
因此“轨迹正常结束”不代表所有工具调用成功，更不代表回答正确。

## 验证和限制

- BFCL 新增测试 21 项 + workload 回归测试 21 项通过；另有自定义 request 生命周期测试 11 项通过，共 53 项。
- GPU 运行前独立代码审核取得 GO；单轮模板、精确 token 历史、错误传播、取消、公网 DNS 等路径有测试。
- 首轮 r1 在预热阶段发现 `wt-wt` 地区映射问题后主动终止；修正为有效默认地区 `us-en` 后重跑本轮。r1 不作为性能结果。
- 本轮没有引擎崩溃或 OOM；服务端口 27600/27610 已关闭，GPU 0 回到运行前 599 MiB 占用。
- 工具后端是无密钥替代实现，不是官方 SerpAPI 后端；公网结果和限流不可完全复现。
- 原始 `turn_metrics` 的 `inference_time` 有显然错误的累计时间值，派生 TTFT/TPOT 和单请求 decode_throughput 不可信，本报告不使用这些字段。Token 计数和独立 GPU 累计计数分别统计。
- 未执行官方 grader，不报告 BFCL 正确率。111 条 final 仅指 harness 正常结束。
- 结论：真实多轮工具流程已接通，存在明显大于 1 秒的工具等待；但当前免费搜索错误率较高，尚不适合直接作为严格可复现的 PD/Colocated 性能对比基准。单卡 c8 结果也不能与历史多卡 c512 满载结果直接比较。

原始记录：`closed_loop_boundaries.json`、`requests.jsonl`、`engine_metrics.jsonl`、
`resolved_workload.json`、`config.json`、`environment.json` 和 `logs/`。
