# c512：关闭P→D Host消融

状态：已完成。8个P/D完成Host注册预热后，完成300秒业务预热和1200秒正式测量；
正式窗口完成2866个Agent、失败0个。

- 运行源码：`/homes/siqic/sglang-codex-c512/python/sglang`，独立干净worktree，SGLang `921fbd46ab6d3baf68be66f097661a918dec6707`。
- 不使用`/homes/siqic/sglang-h100-integration`中另一项工作的未提交改动。
- Qwen3-8B；8卡、TP=1、4P:4D、c512；BrowseComp source-order n680循环，temperature=0。
- P GPU=0/2/4/6，D GPU=1/3/5/7，搜索在GPU7；显存P均0.80、D=0.80/0.80/0.80/0.60。
- 8个P/D完成Host注册预热后，300秒业务稳态预热＋1200秒正式测量。
- 唯一消融：关闭P→D Host staging；P→D late binding保留，prebind关闭。
- D→P Direct/Slow保留；工具/Direct deadline均1秒；拥堵反馈重算Q=32/8。
- D→P Host=128 GiB/P；原生HiCache/Mooncake关闭；不改变完整方法默认开关。

## 运行前验证

- 547项Router、双向staging、TP、生命周期及拥堵反馈测试通过。
- 专用脚本bash语法通过。
- 独立agent审核GO；确认false开关贯穿启动链、prewarm允许p2d_path=None。
- 已知budget-final Host副本清理缺口保持不变，结果中单独核对。

完整结果见[RESULTS.md](RESULTS.md)。
