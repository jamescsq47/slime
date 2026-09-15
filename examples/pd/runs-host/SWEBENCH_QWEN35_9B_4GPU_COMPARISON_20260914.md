# Qwen3.5-9B：4 卡 SWE-bench Verified 结果简记

记录日期：2026-09-14。仅记录结果，不作原因分析。

## 共同配置

Qwen3.5-9B，TP=1，GPU 0/1/6/7；静态显存比例 0.80，Mamba/full memory ratio 0.5，page size/track interval 64，context 131072，Prefill chunk/max tokens 8192。Triton attention，确定性推理，seed 2026。

相同 500 条 SWE-bench Verified，各执行一次、保持原始顺序；外部 OpenEnv/Miles PR51 harness，单轮 8K、最多 64 轮，temperature 0.6、top_p 0.95、top_k 20、min_p 0。Docker 镜像已预下载，每容器 2 CPU / 4 GiB 上限。

Colocated 使用 `pd_mamba_baseline` 原生引擎，关闭 PD、HiCache、Mooncake。两轮 PD 使用相同的 `sglang-qwen35-integration` 融合源码，2P:2D，Attention KV + Mamba checkpoint 回传；工具阈值 1 秒，每 P 4 个 H2D 槽，Host event progress 开启，昂贵内容哈希关闭，Host 预注册完成后开始任务，拥堵/固定失败重算关闭，原生 HiCache/Mooncake 关闭。

## 全量结果

| 指标 | Colocated c128（R16） | 2P:2D c256（R15） | 2P:2D c500（R17） |
|---|---:|---:|---:|
| 结束任务数 | 500 | 500 | 500 |
| 正确率 | 158/500，31.6% | 166/500，33.2% | 161/500，32.2% |
| 正常完成测评 / 环境异常 | 494 / 6 | 495 / 5 | 498 / 2 |
| T250 | 34分49秒 | 33分44秒 | 39分13秒 |
| T450 | 57分31秒 | 54分04秒 | 57分11秒 |
| T500 | 85分57秒 | 66分50秒 | 71分56秒 |
| 单题执行时间 P50 | 13.2 分钟 | 24.4 分钟 | 39.3 分钟 |
| 单题执行时间 P90 | 26.1 分钟 | 41.4 分钟 | 57.2 分钟 |
| 全程实际 Prefill，总计 token/s | 3,126 | 2,809 | 2,609 |
| 全程 Decode，总计 token/s | 1,333 | 1,686 | 1,568 |

口径：T250/T450/T500 从首个任务开始执行，到对应数量任务结束，包含异常结束，不代表成功解出对应数量；不含模型启动及 Host 预注册。单题执行时间从取得并发槽开始，不含此前等待。吞吐为运行时实际计算 token 计数的墙钟速率。有限 500 题完成后不补新题，因此后期并发下降。

## 中段采样结果

| 指标 | Colocated c128 | 2P:2D c256 | 2P:2D c500 |
|---|---:|---:|---:|
| 业务时间窗口 | 300–1500秒 | 约400–1500秒 | 300–1500秒 |
| 实际 Prefill，总计 token/s | 4,391 | 3,387 | 3,948 |
| Decode，总计 token/s | 1,938 | 2,103 | 2,625 |
| Prefill Forward/对应卡 | 18.3% | 67.9% | 68.0% |
| Decode Forward/对应卡 | 75.0% | 98.3% | 98.1% |
| running/Decode 卡 | 28.4 | 55.1 | 68.3 |
| P或Colocated Attention KV池使用率 | 46.3% | 49.9% | 81.7% |
| D Attention KV池使用率 | 同上 | 87.4% | 85.4% |
| P或Colocated Mamba池使用率 | 37.0% | 46.6% | 92.0% |
| D Mamba池使用率 | 同上 | 57.5% | 72.7% |
| Host已就绪且下一轮已到达、尚未启动恢复：平均 / 采样最大 | 不适用 | 0.57 / 3 | 32.7 / 69 |

状态池百分比不是整卡 HBM 百分比。Colocated 的 P/D 在同四张 GPU 上；PD 的 P/D 各两张。R16 来自 2 秒引擎采样；R15/R17 此处来自 30 秒观察记录，R15 有效记录从约400秒开始，窗口和估计方式并非完全相同。

## 原始记录

- R15：[2P:2D c256](/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p2d-c256-host-event-20260913-r15)
- R16：[Colocated c128](/tmp/pd-persist/baseline-qwen35-9b-tp1-swe500-4gpu-c128-host-event-control-20260913-r16)
- R17：[2P:2D c500](/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p2d-c500-host-event-20260914-r17)

各目录保留 `summary.json`、`requests.jsonl`、轨迹及性能记录；PD 另有 `host-progress.jsonl` 和控制账本。未删除旧结果，未修改运行代码。
