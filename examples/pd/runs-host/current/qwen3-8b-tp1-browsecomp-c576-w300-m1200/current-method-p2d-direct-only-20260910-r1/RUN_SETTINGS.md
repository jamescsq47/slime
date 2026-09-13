# c576：回退统一池，仅关闭 P→D Host

状态：已完整完成301.87秒业务预热＋1200.001秒正式测量，窗口内2874条Agent完成、0失败；详见RESULTS.md。实验进程退出完成，GPU显存恢复实验前状态。

- SGLang：`921fbd46ab6d3baf68be66f097661a918dec6707`，启动前源码工作区干净。
- 实际源码：`/homes/siqic/sglang-h100-integration/python/sglang`。
- 模型：Qwen3-8B；8卡、TP=1、4P:4D、c576。
- BrowseComp：旧 source-order n680 循环；temperature=0。
- P GPU=0/2/4/6，D GPU=1/3/5/7，搜索在GPU7。
- mem_fraction_static：P全部0.80；D=0.80/0.80/0.80/0.60。
- 所有P/D预注册完成后，300秒业务预热＋1200秒正式测量。
- 8/8预注册完成，最大耗时250.59秒；每CUDA进程注册512 GiB的D→P映射，注册cache上限640 GiB。旧独立Host对照的cache上限为1280 GiB；两组都足够覆盖全部映射且完成预注册，此上限不是业务Host容量。
- 仅关闭P→D Host staging；P→D late binding仍开启，不使用prebind。
- D→P Direct/Slow保留；工具和Direct建立阈值各1秒；拥堵反馈重算Q=32/8。
- 恢复旧的每P独立D→P Host Arena：128 GiB/P；不启用统一NUMA池。
- 原生HiCache/Mooncake关闭；不修改full method默认开关。

## 运行前验证

- 547 passed（22.18秒），包括Router、双向staging、TP、生命周期、拥堵反馈测试。
- Bash语法通过。
- 独立agent只读审核GO，确认false开关贯穿启动链和prewarm支持p2d_path=None。
- 已知旧版本budget-final Host副本清理缺陷保留，不在单变量消融中顺便修复；结果需单独记录。

## 对照

- 旧独立Host完整方法c576：4822.5 token/s。
- 已撤回统一NUMA池c576：4656.9 token/s（含一次Router超时，不视为零错误验收）。

本次将记录实际Decode batch/running、D/P Forward、KV占用、D→P路径比例及Host生命周期。
