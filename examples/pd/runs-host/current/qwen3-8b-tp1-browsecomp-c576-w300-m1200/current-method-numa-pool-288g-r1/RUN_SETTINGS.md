# c576统一NUMA Host池

状态：已完成约302秒业务预热＋1200秒正式测量。Decode 4656.9 token/s；
存在1次终止/续轮不一致导致的Router超时，因此仅作为诊断结果，不标为零错误验收。
完整对比见RESULTS.md；实验GPU进程与两份Host池已全部停止。

- 修改前SGLang备份：921fbd46ab6d3baf68be66f097661a918dec6707（远端pd/pd_node_a）。
- 实际源码：/homes/siqic/sglang-h100-integration/python/sglang；本轮有统一池未提交改动。
- 数据：BrowseComp source-order n680循环；Qwen3-8B；TP1；4P:4D；c576；temperature0。
- P显存比例四张均0.80；D为0.80/0.80/0.80/0.60，GPU7搜索服务port8750。
- Direct/tool各1秒；失败按原Q32/8全局Slow恢复拥堵反馈决定Slow或重算。
- 统一Host池：每NUMA288 GiB，两方向各96 GiB保底+共享96 GiB，共576 GiB。
- 启动日志旧D2P128/P2D32每P参数在本后端不决定物理容量，仅供旧后端保留。
- 保持现有registered-extent批量DMA、两个方向独立队列，不增加bounce memcpy。
- 全部P/D注册prewarm完成后：300秒业务预热+1200秒正式测量。
- 启动进程：timeout PID1930974（已退出）。CUDA清理超过launcher等待期限后，
  launcher安全保留Host池；确认全部CUDA子线程退出后人工关闭两个broker。
- 门禁：482回归passed；新增小池真实CUDA往返1passed；独立审核GO。

物理池容量/占用/lease统计见logs/host-pool-0.log与host-pool-1.log。
不要累加多个P视图显示的192 GiB逻辑单方向上限。
