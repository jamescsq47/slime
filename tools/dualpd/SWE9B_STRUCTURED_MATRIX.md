# 9B SWE500 structured-tools 重测

2026-09-18。入口：`bash tools/dualpd/swe9b_matrix.sh start|status|stop`。
默认顺序：colocated c500、colocated c256、2P:6D c256、2P:6D c500。
结果按组写入 `examples/pd/runs-host/SWEBENCH_QWEN35_9B_TP1.md`，旧表独立归档。

## r3：只增加工具调用示例

用户授权在 `_TOOL_SYSTEM_PROMPT` 追加简短 `shell(command)` XML 示例，
版本 `openai_tools_shell_example_v1`。`fenced_shell` 提示、工具定义、解析、
重试和终止逻辑完全不变。reference 校验只允许移除这段精确字节后与27B源码
完全一致，其他差异仍拒绝启动；源文件SHA随运行计划保存。
使用新r3目录及systemd unit，不覆盖或续写r2。四组共享相同新增提示。
不涉及snapshot ownership或任何传输/调度状态转换；八条验收标准沿用下述检查。
本次重新验证：62项应用回归、527项引擎CPU回归通过，shell语法与启动plan校验通过；
`/root/swe9b_launch_audit` 独立只读审核给出GO，确认r2进程/容器已退场。

## 配置和运行边界

- 四组共同使用27B的openai_tools workload、同一source-order Verified500和inline verifier。
- 采样0.6/0.95/20，8192单轮、64轮、81920累计输出；每题一次，非持续补样本压测。
- 所有GPU静态比例0.8；baseline状态池0.9，PD0.5，沿用旧9B配置并明确列出。
- baseline只读pd_mamba_baseline；PD用pd_multi_node+dualpd/sglang，不修改原仓库/环境。
- 用户允许共享GPU7上PID1868643；控制器校验GPU、PID和starttime，其余compute进程阻止启动。
- systemd用户服务独立于启动命令/会话；不自动重启失败实验、不承诺重启机器后自动续跑。
- 每30秒保存状态和观测；每组500个唯一ID及终态核验通过后更新表格。
- Verifier timeout作为已收尾的失败评分；基础设施失败单列，不伪装成未执行或成功解题。
- 致命运行错误、源代码漂移、清理未完成会阻止下一组，不覆盖或重复使用旧run目录。
- 正常清理沿原launcher；兜底只对本case环境marker+PID启动身份执行pidfd信号，Docker按独立label清理。
- Host backing目录本轮显式指定，不在无法证明GPU退出时递归删除；故障现场保留。

## 修改范围和设计验收

只扩展launcher路径/workload/reference/parser参数、后台排队、完成验证和报告。
不修改snapshot数据面、调度、credit、期限、失败出口；本次engine修改只有validation脚本内部路径覆盖。
已有engine dirty改动单独保存diff/源码哈希，不声称与历史引擎字节相同。

设计准则1（唯一owner）、2（P→D释放）、3（P→Host释放）、4（D→Host释放）、
5（各IO与forward解耦）、6（TP原子性）、7（复用正确性）均沿用现有引擎实现，
本轮不改变状态转换；PD运行后仍须检查queued/durable/source_release和Host残留，
完成500题不自动等同协议验收。第8条门禁：527项engine生命周期/Mamba/Host/TP/remote
CPU回归通过；47项既有harness/supervisor/采样测试通过，14项新控制器/报告测试通过。
所有测试屏蔽GPU；独立审计 `/root/swe9b_launch_audit` 已给出GO，可启动；
GO仅限启动/执行，不替代PD最终ownership审核。

旧report将“非Forward时间”误匹配为Decode Forward，已修正；openai_tools不套用
旧fenced-shell的L−2理想前缀公式，缺精确token-LCP/checkpoint依据的数字留空。
