# TP8 D→P Direct-only 对照（2026-09-21）

## 上一轮 r4：用户要求停止

`qwen35-122b-swe500-tp8-c128-socket-full-r4` 未完成500题，不是正式性能结果。
停止前诊断：最近5分钟 D→Host完成292、Host→P交接189、Host驱逐重算处理111。
已恢复集合中，工具返回且Host就绪后到P选中平均约94秒；全程选中到准备约0.79秒，
准备到本rank复制完成约1.38秒，复制完成到组交接释放约2.82秒。
30秒抽样 P Forward93.6%、D83.5%，D整个TP8组约444 token/s，running23–25。
另有一个已驱逐父generation未见后续推进，未宣称根因已解决。
通过协调器SIGTERM执行原有owned cleanup；已确认a10/a11无该轮GPU进程。

## 对照配置

```bash
bash tools/dualpd/qwen35_multinode.sh run --concurrency 128 --d2p-direct-only \
  --run-dir /homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-swe500-tp8-c128-socket-direct-only-r5a
```

- 模型/数据/顺序不变：Qwen3.5-122B-A10B，a10=P、a11=D，各TP8；SWE Verified500各一次，含verifier，c128。
- 显存0.8、Mamba比例0.5，openai_tools；temperature0.6/top_p0.95/top_k20。
- 工具窗口仍2秒，Direct建链仍1秒。工具超时或Direct失败，在真实DMA fence收敛后显式重算。
- 只关闭D→P Host staging，不启用原生HiCache/Mooncake；P→D late binding/Direct/Host保持不变。
- 默认不开此消融开关。这里的重算不是SWE默认关闭的“拥堵反馈重算”，而是没有反向慢路径时的显式失败出口。

## 启动前发现并修复的消融兼容性问题

1. 多节点启动检查允许D→P Host关闭，但仍要求自定义生命周期、D-hostless和P→D Host。
2. D没有Host源时，预热报告零arena/零bytes；P仍预热P→D Arena。只允许此明确配置接受零字节。
3. D无Host失败出口复用既有`fail_direct_offer`事务，避免旧的无claim `mark_failed`覆盖并发P claim；保持DMA fence、路由重试及TP释放。
4. P无Host时复用现有`terminal_prepare → all-rank ready → terminal_admit → clear`，
   防止各rank在不同tick观察到FAILED后各自入队。仅TCP、TP>1、Host关闭分支启用；
   正常Host-enabled路径和TP1调度不变。取消也保留该组级收尾，不能遗留四个metadata名额。
5. smoke验证Direct复用，以及慢工具显式FAILED+零缓存完整重算，并与完整参考逐token比较。
   单P不强制发布route，检查权威生命周期终态而非要求无意义Router marker。

## 验收不变量

- 唯一所有权：Direct成功交接或claim-safe FAILED后释放源；竞争失败保留D源。
- P→D Direct及Host释放逻辑未改。
- D→P Host释放在此消融不适用，要求Host写入/恢复计数为零。
- 不增加网络轮询/文件扫描；终止协调复用已有TP控制批次及内存镜像。
- TP所有rank统一终止入队；取消和未退还workset必须等待原有fence。
- Direct成功复用与显式失败重算分别计数，不能把重算称为缓存命中。
- CPU回归与独立审计GO后才能启动GPU；smoke通过才启动500题评测。

CPU门禁：1195 passed、2 GPU skipped；78 launcher tests passed。独立审核GO，
额外复现确认真实store上的并发P claim不会被D失败出口覆盖。
`r5`只做过preflight、未启动GPU；`r5a`重新捕获最终代码后启动。
02:39:40 UTC所有rank预热完成：P每rank8GiB，D每rank零arena/零bytes。
Direct smoke全部8rank提交、复用8192父tokens；慢工具触发FAILED原因
`shared_host_staging_unavailable`，后续cached_tokens=0。两例均与完整参考逐token一致，
均无D→P Host entry。
02:40:39 UTC启动500题/c128业务；结果待完成后补充，不代表吞吐或长期稳定性达标。
