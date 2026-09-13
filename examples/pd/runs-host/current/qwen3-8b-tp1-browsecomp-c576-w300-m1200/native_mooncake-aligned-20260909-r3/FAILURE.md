# 原生 Mooncake c576：重试运行的数据无效

- 日期：2026-09-09
- 配置：Qwen3-8B，BrowseComp source-order n680，TP=1，4P:4D，c576，temperature=0。
- 时间：完整通过300秒业务预热和1200秒正式测量，服务进程正常结束。
- 结论：搜索后端在负载下系统性失败，完成集合不是有效的BrowseComp轨迹；不计吞吐结果。

## 数据正确性检查

- 正式窗口表面完成8281个Agent，但全部只有1次模型调用。
- 全程请求状态为：7038条`aborted`、2977条`truncated`、94条`completed`。
- 7038条aborted全部为`search_backend_error`；正常完成的94条也都只有1轮。
- 因大量请求在第一次搜索时提前退出，表面的9582.6 Decode token/s不可与其他
  BrowseComp实验比较，不能填写为正式结果。

## 基础设施观测

- 本次未重现第一次运行的NIXL断连；P、D、Router和Mooncake完整运行到窗口结束。
- Mooncake正式窗口平均使用80.76%，峰值84.90%；发生47次成功驱逐，分配失败0。
- 当前搜索worker会把批处理内部异常作为HTTP 500返回，但没有把异常正文写入
  `search.log`；Agent只记录`search_backend_error`。因此现有日志可以确定搜索执行
  系统性失败，但不能据此把具体根因断言为OOM。

## 后续要求

若要取得原生Mooncake c576的有效吞吐，需先让搜索worker持久化内部异常正文，
确认并修复搜索侧故障，再按同一数据顺序和配置重跑。不得使用本目录中的表面
吞吐作为性能结果。

