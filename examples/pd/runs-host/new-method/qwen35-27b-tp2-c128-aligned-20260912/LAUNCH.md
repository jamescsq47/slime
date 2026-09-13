# Qwen3.5-27B TP2 · 新方法4P:4D · c128 · SWE500

用户2026-09-12明确要求**总并发128**，不是64。完整500个不同SWE-bench Verified instance，固定顺序、完成一个补一个，数据耗尽后drain；不是300+1200秒截断测评，另统计300–1500秒共同窗口。

R3故障目录：`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c128-20260912-r3-foreign-ready`；下一轮R4保留相同配置。
R1失败现场完整保留：`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c128-20260912-r1-aligned`。

| 配置 | 本轮 |
|---|---|
| 模型 | `/homes/siqic/Qwen3.5-27B` |
| 物理GPU / TP | 8张A100；TP=2 |
| P组 / D组 | P=`0,4;1,5`；D=`2,6;3,7` |
| 并发 / 题数 | **128 / 500** |
| 静态显存 / Mamba比例 | 0.80 / **0.5**（baseline为0.9） |
| Attention / Sampling backend | Triton / FlashInfer |
| Page / Mamba track | 64 / 64 |
| Mamba strategy | extra_buffer + request-owned stable prompt checkpoint |
| Prefill chunk / batch token上限 | 8192 / 8192 |
| context / 单轮输出 / 总输出 / 轮数 | 131072 / 8192 / 81920 / 64 |
| 采样 | T=.6、top_p=.95、top_k=20、min_p=0、thinking=true、seed2026 |
| 确定性 / custom all-reduce | 关闭确定性；允许custom all-reduce |
| 外部harness | swe_bench_openenv，Chat Completions，openai_tools；不修改 |
| Docker / Verifier | 本地预下载500镜像；2CPU/4GiB上限；工具600s、verifier2400s、verifier并发16 |
| 回传 | Attention KV + 匹配的Mamba checkpoint；Direct与Shared Host慢路径 |
| 快工具 / Direct建链阈值 | 1s / 1s |
| 主动拥堵重算 | 关闭；Q32/32记录但不生效 |
| H2D槽 | 每P worker 4；TP保留原子fence，TP1-only lane解耦关闭 |
| Host / 注册 | 每P rank D→P128GiB、P→D32GiB；物理640GiB；8个CUDA上下文全部预注册后启动业务 |
| 哈希 / 原生HiCache / Mooncake | 内容哈希关闭；原生HiCache/Mooncake关闭 |
| 解释器与依赖 | pd_mamba_baseline；Torch2.11.0+cu128、Triton3.6.0、FlashInfer0.6.7.post3 |
| 实际引擎 | 独立`/homes/siqic/sglang-qwen35-integration/python`源码overlay；不是baseline引擎，无环境包修改 |

启动脚本：`scripts/new_method/run_qwen35_fused_27b_tp2_swe500_4p4d.sh`。
监护：`scripts/baseline/monitor_qwen35_27b_swe500.py RUN --launcher SCRIPT`，每30s落盘进展并扫描致命引擎错误；退出调用本轮launcher自己的有序清理，不清理其他用户进程。

## 启动前验收

- CPU回归473项通过：lifecycle、TP、integration safety、request-owned Mamba、Mamba Prefill/admission、startup prewarm、内容hash、final Host cleanup、CUDA worker。
- 外部OpenEnv harness回归9项通过。
- 模型服务核心模块在baseline依赖解释器+fused overlay下CPU导入通过；GPU依赖兼容仍需启动实证。
- 数据、workload及harness源码与已完成c64 baseline归档逐字节一致。依赖版本在preflight再次验证。
- 独立审计`audit_pd_c128_alignment`：GO，条件为上述测试、数据检查和资源检查通过。此次仅改启动配置与监护入口，八项生命周期验收不变量不变；正式过程继续核验KV/Mamba回传及终态释放。
- GPU0–6启动前空闲；GPU7保留其他用户PID1868643、588MiB显存，沿用此前允许的0.80静态比例，不终止该进程。
- 启动前Host available约1.9TiB，可覆盖640GiB物理Host Arena；根盘空余554GiB，/dev/shm空余999GiB。

注意：最近完整baseline为c64，本轮为用户指定c128，不能写成等并发性能比较。Fused非workset-backed的旧resumed-chunk分支与最新baseline容量修复不完全一致，监护重点关注长New请求续算；没有在本次启动中臆测或修改其所有权逻辑。最终以preflight、启动日志和实测错误记录为准。

## R1故障与R2最小传输层修复

R1在全部8个CUDA上下文完成Host预注册、业务开始约90秒后，两个D的TP1在
`bootstrap_thread → add_remote_agent → ucp_ep_rkey_unpack` 原生段错误；随后其他rank
报Gloo断连。只结束1题/通过0题，不能填入吞吐/正确率对照表。监护已停止本轮全部
GPU服务及本轮Docker，保留GPU7其他用户进程。

安装的NIXL1.3.2二进制仍在UCX错误回调中立即关闭endpoint，与上游
[issue1986](https://github.com/ai-dynamo/nixl/issues/1986)及
[PR1987](https://github.com/ai-dynamo/nixl/pull/1987)描述的危险路径一致。
本轮触发endpoint异常的最初原因尚未完全证明，不能只凭相同调用栈宣称全部根因已解决。

针对性回补：以当前安装版本的精确源码`75ead3d7`为基底，只在UCX插件3个文件中
加入PR1987的atomic失败状态、延后endpoint关闭及rkey导入前状态检查。其余Python
binding、NIXL core、UCX1.21/CUDA库及SGLang/harness/调度保持不变；没有修改任何
共享conda包。新版wheel1.4.1检查后仍含旧回调，因此未用于正式实验。

- 插件：`/tmp/pd-runtime/nixl-132-pr1987-plugin/libplugin_UCX.so`。
- SHA256：`c006bd99d6eb6e1024820b88d4ecce6aee82a4147953ce00b0581942f7d52e77`。
- 仅本实验通过`NIXL_PLUGIN_DIR`加载；启动前核验hash/1.3.2版本，拒绝无声回退。
- 构建源码：`/tmp/pd-runtime/nixl-132-pr1987-src`；构建命令：`/tmp/pd-runtime/build-nixl-132-pr1987.sh`。
- 本轮`nixl-pr1987.patch`、`preflight.json`记录补丁及与baseline的依赖例外。
- 回归：473项生命周期/取消/容量/TP/Mamba等通过；外部harness9项通过。
- 原生故障回调测试：worker1/8、strict/RW共4配置通过，含重复回调与析构。
- GPU4/6的64MiB直传：释放源HBM后目标全部字节正确；这是工程smoke，不是正式性能数据。
- 独立审计`audit_pd_c128_alignment`：最终GO。八项snapshot验收约束均不变；仍须正式过程核验。
- 监护补充原生段错误/后台CUDA worker失败检测，发生时调用原有有序清理。
- R2启动时间：2026-09-12 23:25:28 UTC；仍需验证全部8个rank实际加载插件、Host预注册与完整500题。

## R2跨P恢复发现阻塞与R3修复

R2全部8个rank均加载指定插件、Host预注册完成；23:36:46开始业务。实际有数百次
Direct及数十次Slow回传成功，没有复现R1段错误。但出现72个HOST_READY长时间不恢复：
全部为跨P（arena0→P1或arena1→P0），工具均已返回；P HBM空闲、H2D槽0/4。
23:43附近账本：138个snapshot双rank D2H完成并释放D源，67个双rank H2D及group
commit完成；余71个Host等待随后增至72。不是“工具尚未返回”，也不是PCIe带宽打满。

根因：`scheduler.py`的TP Host命令选择调用`host_staging.snapshot_ready(parent)`，
原实现只看本地`host_ready`；跨P的本地映射却要在后续`gate_request`中才建立。
持续的Direct TP命令使普通扫描无法及时发现这些foreign snapshot，造成恢复饥饿。
偶发普通扫描能绕过，因此不宣称所有情况下都会无条件死锁。

最小修复仅改`AgenticPHostStagingManager.snapshot_ready`：TP>1、本地尚无映射时，
从后台已有ledger cache检查HOST_READY、指定恢复P、完整TP grant集合及owner。
不做磁盘读取、mmap、HBM分配、claim或DMA；选中后仍由原gate/CAS及TP屏障重新校验、
取得完整workset并恢复。TP1、本地ready、路由选择、Direct阈值、Q、harness均保持不变。

- 新增16项回归：旧代码两个foreign-ready用例失败；修复后通过。
- 完整489项生命周期/TP/容量/取消/故障/Mamba回归通过；独立审计GO。
- R2唯一Direct release_pending异常已确认双rank回滚→跨P H2D成功→后续22/23/24轮继续，非这批积压根因。
- R2结束24/500、通过10题；23:45附近发出有序停止，23:49:58退出码130；GPU0–6已空闲、GPU7其他用户601MiB保留。结果只作诊断，不填完整测评表。
- R3启动：2026-09-12 23:52:09 UTC；完整500题/总并发128，其他配置与R2一致。
- R3必须验证跨P两个rank均完成H2D和Host释放，并检查已返回工具的Host积压是否收敛。

## R3 TP消费顺序分叉与R4最小修复

R3于2026-09-13 00:03:33开始业务，跨P映射、双rank H2D和group commit释放
均有实际记录。但00:04:16 P1的TP1首先报`TP workset handoff commit reached
a non-handed local lease`，snapshot=`d80fd69ba4744ee6b8a8c5c87ca899d5:1`。
后续P0访问P1的`/proc/PID/fd`失败发生在P1异常退出后，不能当作最初根因。

根因是`_agentic_tp_consume_admission_control`把有序Host commit命令转成set，
`_drain_agentic_kv_waiting_queue`再按set迭代顺序消费。在同样5个Direct之后，
admission batch=8只剩3个位置，两个rank选择了不同的3个Slow，破坏TP一致性。
不放宽原handoff断言：只把commit序列改为保留rank0广播顺序的list/tuple。
不改容量、优先级、路径、timeout、物理fence、释放或Mamba payload。

- 新回归模拟两个rank不同set迭代、相反HTTP到达顺序、5Direct+4Slow、cap8；
  原代码真实失败，修复后通过。
- 完整490项生命周期/故障/取消/容量/TP/Mamba测试通过，16.13秒；diff whitespace检查通过。
- 八项检查：唯一owner不变；P→D Direct释放不变；P→D Host释放不变；
  D→P Host释放不变；不增加等待/I/O；修复TP消费序列原子性；KV/Mamba复用判据不变；
  GPU启动仍以独立审核GO为门禁。
- R3只结束2题/通过0题，不能作为正确性或性能结果，全部日志保留。
  00:08:39有序停止完成，00:09:35确认GPU仅保留其他用户PID1868643。
- 独立审计最终GO，独立运行两个新增回归文件17/17通过。
- R4于2026-09-13 00:12:48 UTC启动：`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c128-20260913-r4-tp-order`。
  monitor PID3712727，launcher PID3712742。仍须先完成Host预注册，再开始500题。

## R4取消后孤立Host命令阻塞New，R5修复

R4于00:24:14开始业务；全部8个rank插件/源码核验通过，8/8 Host预注册完成。
前几分钟Direct、跨P双rank恢复、提交顺序和Host释放正常（00:28检查282个Slow均已
恢复释放）。随后New不再进入Prefill：HTTP已接收217个首轮请求，P完成首轮仅102个；
两个P的HBM使用降到0，D只剩少量running。90题结束/25通过后主动有序停止；
00:46:48退出130，GPU仅保留其他用户进程。不是完整测评结果。

明确证据：7个已经在新P恢复释放的snapshot，在旧P的TP Host mailbox仍为双rank状态0。
例如`56121af509bd47b2b5e3600ab88e4ba9:0`：00:24:38旧P0取消时记录
`stale Host abort ignored`，00:24:42新P1双rank完成H2D/group commit释放，
旧P0也确认remote recovery release；但旧P的scheduler active命令没有被取消。
之后`reduce_host_status`找不到原HTTP请求，持续报告0→prepare；forced-TP扫描只找
这个已不存在的parent，New无法被扫描。共用的`d2p-host` mailbox还会使旧P状态覆盖新P。

R5限定修复，不提高New优先级、不改变容量或timeout：

1. 已在本地存储的Host记录也必须尊重明确的recovery P分配，旧P不能继续选中它。
2. `d2p-host`控制mailbox按P domain隔离；它仅用于P组内部TP同步，不改变P→D收发回执。
3. 取消保留首个HTTP attempt身份，原Host abort后台继续执行。确认没有load、lane、
   resident reservation、pending abort、Slow/Direct lease、intent、冻结TP plan后，
   每rank报告控制已退出；全部rank一致后才clear旧调度命令。绝不在这里释放Host/HBM。
4. 同parent重试在旧取消收尾前保持metadata-only；重复取消不覆盖最初owner。
   已取消命令只允许abort/clear，不能因另一rank陈旧的bind状态重新commit。

- 旧HEAD取消/reduce方法运行新增两例，均确切失败；修复后完整512项通过（19.10秒）。
- 覆盖跨P namespace隔离、不同rank完成顺序、未到HTTP不等于取消、CAS失败、在途DMA、
  lane/resident/冻结计划、重复取消、同parent新HTTP重试及混合陈旧状态。
- 八项验收：唯一owner与三种释放fence不变；只对同snapshot取消执行原物理fence等待；
  消除跨P控制串扰并保持all-rank clear；KV/Mamba与重算策略不变；GPU启动需独立GO。

另发现对照文档须保留的限制：baseline Tool协议schema多出`defer_loading:null`，
首轮模板比融合版多6tokens；相同harness源码不等于完全相同渲染文本。R4内部3409次
Prompt长度与harness声明一致，30个多轮实际token序列与harness重新渲染完全一致。
双方auto+strict=false未启用结构化grammar，因此parser capability差异不能解释本轮
格式失败。未为追求正确率改harness/parser；短失败完成集合不能当作全500题正确率。

独立审核最终GO，独立执行新增及相关39项全部通过。
R5于2026-09-13 00:53:01 UTC启动：
`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c128-20260913-r5-host-cancel`。
monitor PID4172738，launcher PID4172755；仍为完整500题、总c128，配置不变。

R5于01:04:31.503开始业务；8/8 Host预注册完成，最慢195.82秒；8个scheduler的
`/proc/PID/maps`均确认加载隔离的NIXL PR1987插件。启动不计入业务时间。
01:13检查187个不同首轮请求已到达、178个已完成Prefill，补题已越过初始128题，
没有复现R4的New永久阻塞。01:14约69题结束；这只是运行中进度，不是最终测评。
01:14:45附近所有权抽查：D→P Host完整D2H 1312个，D源释放1312个；
P→D Host完整D2H 842个，P源释放842个。双rank Host提交顺序相同，尚无engine fatal。

正确性仍需审慎：早期结束集合包含较多真实模型格式错误，不能只按无传输异常验收。
独立审计前17题的170对相邻轮，实际下一轮token前缀均等于parent所提供前缀；
全部checkpoint边界符合去除末尾think opener后按64对齐。该检查证明token身份和
边界一致，不证明未记录的Attention/Mamba物理状态数值一定正确。

## R5正确率异常：27B GDN门控错读，R6两行修复

01:19同题核对：R5已结束157题通过51；baseline在完全相同157题通过102。
R5有65题重复命令结束、30题no_command，不能用完成集合偏差解释全部差距。
首轮结构抽查均正常，错误更多出现在多轮。01:22:54确认根因后有序停止；
最终201题结束/64通过，01:27:02退出130。GPU源进程均已清理，其他用户PID1868643未动。
最终同201题baseline通过130题；R5结束原因no_command34、重复命令83、单轮8k26、
TASK_COMPLETE16、普通总结2、64轮40。baseline同题分别为no_command11、重复命令0、
单轮8k8、TASK_COMPLETE79、64轮100，另工具超时2、tool_format_error1。

根因与官方[issue22311](https://github.com/sgl-project/sglang/issues/22311)相同：
27B的linear V/K heads为48/16，ratio=3进入非融合投影分支；TP2切分后的a/b
shape=[N,24]，真实stride=[48,1]。旧fused_gdn_gating kernel假设行stride=24，
因而错误读取后续tokens的门控输入，污染Prefill结果及checkpoint。
baseline的模型文件已有a/b连续化，本融合版遗漏。这个故障不是工具parser造成，
也不等同于传输丢页；即使传输逐字节正确，也可能传递已算错的状态。

最小生产修复只有`models/qwen3_5.py` fallback中的两行：
`b = b.contiguous()`；`a = a.contiguous()`。
不更改共享kernel、Qwen3模型、9B fused分支、harness、温度、配置、调度或生命周期。

- 实际模型forward/split AST回归，修复前6失败/5通过（batch>1均失败），修复后通过。
- 独立CPU审核11通过；完整生命周期/TP/取消/容量/Mamba等523通过、2GPU项跳过。
- 隔离GPU0数值检查13通过，4.36秒；float32/BF16均与连续参考逐元素一致，且与
  PyTorch门控公式一致。旧CUDA split-view负对照GDN gate最大误差10.4166/10.3419，
  beta最大误差0.1490/0.1484。证明修复真实错误，不是仅满足布局断言。
- 八项不变量：唯一owner、三种源释放、TP一致性、前缀边界与释放fence均不改；
  不引入新控制等待；修复模型算术；完成回归与独立GO后重启。

R6于2026-09-13 01:28:20 UTC启动，monitor PID900678：
`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c128-20260913-r6-gdn-contiguous`。
保留4P:4D/c128只验证这次正确性修复；先不混入2P:6D配比变化。
配比初步依据：有效baseline共同中段P/D Forward时间25.99%/72.95%，约1:2.81，
因此2P:6D值得后续验证，但不能据无效R5数据宣布最优配比。

R6业务于01:39:54.719开始；8/8 Host预注册完成（最慢201.796秒），8个rank已验证
使用隔离NIXL修复插件。正式推理前没有额外探针流量；GPU数值探针在启动前已退出。
