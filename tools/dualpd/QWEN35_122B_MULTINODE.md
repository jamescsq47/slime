# Qwen3.5-122B-A10B：a10 P / a11 D

2026-09-18 用户授权本机与 a11 直接进行多节点验证，替代此前仅开发远端脚本的工作流。
仅修改 dualpd 两个仓库。a10 的既有 ComfyUI 不终止；a11 启动前必须无其他 GPU 进程。

## 固定配置

- 2026-09-18用户明确：所有SWE-bench Verified任务默认关闭拥堵重算策略。
  本入口及通用多节点入口均默认 `slow_congestion_recompute=false`。
  正在运行的r10不重启、不修改其已保存配置；它仍为true，结果必须如实标注。

- 两节点各 8 张 A100 80GB，各自 TP=8、EP=1，不跨节点做模型 TP collective。
- 两端 BF16、静态显存 0.8；context 131072、page/checkpoint 64、chunk/prefill batch 8192。
- 新方法后续默认 `mamba_full_memory_ratio=0.5`；r10及以前实际为0.9，不修改历史结果。
  colocated配置不在本次修改范围；本轮运行中的参数不变。
- a10=P+Router+SWE Docker/verifier；a11=D。复用已下载的同一模型 revision。
- `openai_tools`、temperature=0.6、top_p=0.95、top_k=20；单轮8192、64轮、累计81920。
- c64、SWE-bench Verified500 source-order 各一次，inline verifier；不是300+1200吞吐验收。
- 快工具阈值2秒、Direct建链1秒；拥堵重算关闭（历史r10及以前曾启用Q32/8）。原生HiCache/Mooncake关闭。
- D→P Host仅在D源节点：默认32GiB/rank，共256GiB；P→D Host仅在P源节点：默认16GiB/rank，共128GiB。历史实验按各自保存的配置解释，不追溯改写。
- IB接口10.0.1.170/171，mlx5_1:1，报告200Gb/s；尚不能把链路速率当成实测带宽/GDR证明。
- 数据面为NIXL GPU→GPU / 本地GPU→DRAM→远端GPU。NFS只存控制元数据和结果，不存KV payload。

## r10 非 running KV 诊断（2026-09-18，阶段性）

- 17:30:18 D 的不可驱逐 Attention KV 为 1,500,416 / 1,539,136 tokens；
  running 仅3个请求、57,971 tokens。`num_physical_used_tokens` 的实际计算已经
  扣除 evictable Radix，不能把这个差额解释为普通可驱逐前缀缓存。
- 约17:32的日志审计发现56个已 offer、尚未 release 的 snapshot，父序列总长
  1,486,848 tokens。该数是逻辑长度，不是去重后的物理页数，且采样时刻不同；
  但足以定位主要占用在尚未完成 D→Host 的父 KV。
- 例 `8ded90f479f248b188084a68b1e90c29:37`（60,096 tokens）：17:27:02
  Slow offer，17:31:29才启动D2H，17:32:13所有rank完成并释放。排队约267秒，
  最慢rank D2H墙钟44.3秒，其CUDA事件累计仅0.871秒。
- follower进度函数在筛选Direct/Slow类别及4条Slow lane前，遍历全部candidate并
  查询全部8个rank的abort状态文件。本轮控制目录在共享NFS；非阻塞栈采样也命中
  `TPGroupMailbox.local_status/_read`。配合256-token分块、每轮有界推进，
  元数据扫描反复拖慢D2H进度；最慢rank又决定整个TP组何时能释放源KV。
- 已审计的3,082个release snapshot均有8个rank提交，未发现其中有rank漏释放。
  当前证据指向释放前的控制/卸载积压，不是已确认传输成功后仍普遍保留缓存。
- 本次仅修改后续启动默认值（ratio=0.5、拥堵重算关闭），未修改引擎推进逻辑、
  未重启r10。修复方向是有界/事件驱动TP状态检查，保留Direct abort的
  no-future-write安全屏障，避免每个DMA进度周期扫描全部历史候选。

### 后续代码：D TP Slow 控制热路径优化

- Slow follower直接消费已有sticky active window（本配置最多4条lane），不再
  先遍历全部candidate查询各rank的Direct abort文件。Direct线程跳过TP0已经
  判定为Slow的candidate，但仍检查wait/direct/direct_abort并在发送锁内复查。
- 多节点launcher仅为D TP>1设置 `SGLANG_AGENTIC_KV_D_TP_CONTROL_DIR`，路径为
  `/dev/shm/dualpd-tp/<control-run-hash>/<engine_id>`。8个D rank在同一节点、
  共享IPC namespace；跨节点TP (`nnodes>1`) 不允许这个覆盖。
  P/D间claim、receipt及Host生命周期仍在共享控制目录，不搬迁。
- 不修改route、workset、chunk、lane、DMA fence和所有权提交/释放条件；
  单卡不使用D TP邮箱，原有单节点启动入口也不自动改目录。
- 所有者仍为D，直到原Direct完成或全部Host shard durable；超时、取消及失败
  仍等待原物理fence，不新增提前释放。退出不删除可能仍被rank使用的邮箱。
- 独立审核GO；CPU回归783 passed / 2 skipped，启动脚本59 tests通过。
  新测试覆盖TP2/8下56个Slow与64个wait的隔离、有界4-lane推进、run/engine
  目录隔离；既有abort/命令换代/取消/fence测试继续通过。
  八项验收：1–4、6–7不改状态机并通过回归；5移除无关元数据I/O耦合；8独立审核
  和CPU门禁通过，真实GPU性能及运行时所有权数量守恒仍待下一轮验证。
- 未重启r10，运行中实验仍使用修改前代码；不能将本节优化当成本轮已测收益。

## 启动

在a10、`dualpd/slime`目录运行：

```bash
bash tools/dualpd/qwen35_multinode.sh preflight --run-dir /homes/siqic/dualpd/slime/runs/dualpd/UNIQUE_RUN
# 必须通过CPU测试与独立GO审核后才能运行；先smoke-only，不通过不启动评测。
bash tools/dualpd/qwen35_multinode.sh run --smoke-only --run-dir /homes/siqic/dualpd/slime/runs/dualpd/UNIQUE_RUN
# 全量使用新run目录，避免复用旧进程注册地址/ledger：去掉--smoke-only。
bash tools/dualpd/qwen35_multinode.sh status --run-dir /homes/siqic/dualpd/slime/runs/dualpd/UNIQUE_RUN
bash tools/dualpd/qwen35_multinode.sh stop --run-dir /homes/siqic/dualpd/slime/runs/dualpd/UNIQUE_RUN
```

结果在run目录；每个节点进程日志在本机`/tmp/dualpd-multinode/RUN/ENGINE/service.log`。
停止使用已有带身份验证的supervisor；不按进程名批量kill，不清理他人的Docker容器。

## 此次接入与所有权检查

新增严格混合布局启动白名单，而非移除Mamba检查：必须BF16 MHA+GDN、request-owned、
extra_buffer和page对齐checkpoint。Direct沿用已有TP shard协议；Host复用完整复合snapshot适配：
P→D active+checkpoint两份state，D→P稳定checkpoint一份，不把旧reasoning尾部当作必然可复用。

1. 唯一所有者：沿用ledger/CAS；新launcher不引入owner或旁路。
2. P→D Direct释放：沿用全TP确认和物理fence。
3. P→D Host释放：源本地D2H durable后释放P，不等D接收；远端接管后释放Host。
4. D→P Host释放：源本地D2H durable后释放D；恢复前申请完整workset。
5. 解耦：未改Forward、队列优先级或传输调度。控制面NFS延迟仍需实测，不保证无开销。
6. TP原子性：rank匹配、全部shard完成/提交；部分失败不能释放未fence的资源。
7. 正确性：测试覆盖完整Attention+state字节，实际Direct/Slow两轮输出和checkpoint复用待GPU验证。
8. 门禁：CPU生命周期/故障测试+独立审核后才启动。测试通过不等于多机性能验收。

CPU状态：r6前独立审核GO（工程smoke）；766项CPU回归通过、2项GPU用例跳过，
launcher unittest 56项通过；审核者另独立运行40项重试/注册测试通过。

首次r1在D权重加载完成后，因SSH未激活Conda导致FlashInfer JIT找不到`ninja`退出；
两节点supervisor已完整清理，仅保留他人的ComfyUI。已修复launcher显式prepend
Python环境bin并固定CUDA_HOME，预检覆盖ninja/nvcc。不修改引擎调度或传输语义。

## 实际验证（2026-09-18）

- r2：第一次跨节点P→D及D→P Direct已完成，8rank接收/源释放，复用832tokens。
  但任意长度prompt的增量与完整重算第5个输出token分歧，测试立即停止；
  不运行500题，不将该分歧直接归类为正常数值差异。单P不发布Direct route marker
  是预期协议；smoke改为检查同generation的全部8rank实际admit，不以发现marker替代完成。
- r3：仅诊断、开启既有字节哈希；第一段prompt固定8192，使完整重算与checkpoint分块相同。
  Direct与Slow均复用8192tokens，输出16token逐token等于完整重算。
  两例P→D的active+checkpoint哈希、D→P的conv/temporal哈希均8/8一致；
  Direct Attention接收前后8/8一致；Slow无独立接收Attention日志，使用增量Prefill后
  仍未改变的8192父前缀哈希与D源对比，8/8一致。此证据不证明任意分块bitwise一致。
- r3所有权：Direct为8admit/8D释放；Slow为8D2H完成/8D释放/8H2D完成/8组提交释放；
  两例各2轮P→D均8rank释放。独立审计复核一致。账本是否排空必须查询分片event，
  全局JSON的entries为空不能证明无待办。
  原始证据位于 `runs/dualpd/qwen35-122b-a10p-a11d-tp8-c64-r3/evidence/`。
- 仍待P→D Host真实容量背压验证，再进入c64 SWE500。没有正式吞吐或评测结果。
- r4的容量背压已触发Host offer，但诊断请求使用`return_logprob=true`，被已有Host
  `_prefill_metadata`安全拒绝，随后D空出容量走Direct。测试正确地判定未覆盖Host而停止。
  r5仅将该probe关闭logprob，与真实SWE一致；仍比较服务器原始output_ids，不重分词。
- r5：P→D Host成功，8D2H/8H2D/8P释放、两份Mamba状态8rank哈希匹配、16token输出一致。
  随后D→P Slow出现RDMA注册资源不足，已停止，未运行SWE500。
  根因一：仅导出CPU数据的Host source agent也注册了完整GPU池；双向叠加重复注册。
  已改为source只注册Host，receiver首次READ前注册GPU一次；部分失败回收描述，无法确认
  回收时保留agent和描述并禁止创建新agent重试。仅改变多节点Host注册职责。
  根因二：peer触发RETRY_PENDING时，本rank pre-H2D被拒绝，workset已回退active但未记录
  io_quiesced，清理重复cancel不成功，导致少1个rank确认。权威分片仍保留Host，并非数据丢失。
  已记录成功回退；回退失败则隔离保留。TP1/8使用实际broker/ledger故障注入覆盖。
- r6：P→D Host、D→P Direct、D→P Host三例全部通过，16个输出token与各自完整
  重算参考一致；Direct/Slow均复用8192tokens。P→D Host两份Mamba状态、两条
  D→P路径的checkpoint状态均8/8哈希一致；Host实际D2H/H2D和源释放均有8rank记录。
  权威`SharedHostStagingLedger.snapshot_entries(force_refresh=True)`检查：D→P零条，
  P→D为1条consumed（loader_acks 0..7）及8条rejected，无非终态待办。
  没有重现RDMA注册错误/恢复卡住；退出后a10仅原有ComfyUI，a11无GPU进程。
  证据保存在同名run的`evidence/`，此轮不是性能测试。
  另发现跨节点响应`inference_time`/`decode_throughput`计时异常，不能用这些字段
  报告吞吐；后续应使用协调端墙钟、完成token总数和各worker本地计数。
- 独立审核r6全部rank哈希、源释放及真实账本后给出GO。r7启动正常容量、c64、
  SWE500（含verifier）；关闭诊断哈希和12288-token容量限制，先正常容量smoke。
  启动不等于完成评测，结果以r7的completion/failure及逐题输出为准。

P→D Host诊断使用新run目录：

```bash
bash tools/dualpd/qwen35_multinode.sh run --smoke-only --diagnostic-digests \
  --p2d-host-probe --run-dir /homes/siqic/dualpd/slime/runs/dualpd/UNIQUE_HOST_PROBE
```

该选项只在诊断时把D的原生token池限制为12288，两个8192token请求制造真实容量背压；
不修改Router选择或伪造Host状态。该配置禁止用于SWE/性能测量，正式配置不限制token池。

## r7 停滞调查与增量控制修复（2026-09-18）

r7不是完成的评测/性能结果，已停止并保留同名run的`evidence/`。现场区分两阶段：

- 早期P有空间但Direct响应慢：约1783个保留arrival文件，每个TP rank每次重新
  glob/read整个目录。非阻塞py-spy的24个有效Direct线程样本全部位于
  `iter_arrivals/read_arrival_path`。Direct控制周期升至约0.63秒；此前1秒建链窗口
  的arrival→receiver start均值约682ms/P90约960ms。这不是已测得的网络饱和。
- Slow存在明显启动等待：596个配对snapshot从工具返回/源rank0D2H结束两者较晚者
  到P rank0H2D完成平均49.4秒，P90约141秒；单次H2D操作墙钟均值约0.505秒。
  前一个统计不是严格全TP Host durable时间，不能把差值全部称为allocator阻塞。
  查到TP Host选择默认深度1，而实际恢复器有4条lane；本轮修复该默认不一致。
- 最终停滞不是Host Arena装满：P最后Forward16:25:05，inflight64、KV约97%；
  D空闲。真实分片ledger中D→P为0条，P→D历史2585条全部rejected/native_won。
  最老未交付room已被Router选择D，但8个sender停在Bootstrapping，未出现receiver。
  Router代码先await P HTTP、后读取D HTTP异常，存在D提前失败后P永等接收、Router
  永等P的错误隐藏环。现场符合该缺陷，尚未取得该room原始D异常，不能声称完全定因。

本轮工程改动：

1. 仅多节点arrival通知改为固定长度append journal+每rank游标；启动一次恢复扫描，
   后续只读新增marker。短写修复、并发发布和读写锁保证通知边界；journal不是KV
   ownership/grant，原manifest/claim/物理fence仍是唯一权威。单节点inotify不变。
2. TP Host默认并行深度取manager真实lane数量，显式override保留并受物理lane上限约束；
   仍只有rank0选请求，所有rank按prepare/start/bind/commit/clear屏障同步。TP1不变。
   增加selected/prepared阶段日志，下一轮验证剩余等待是否仍发生在启动之前。
3. Router等待P响应时同时观察D失败，立即执行已有D/P abort与fence清理，不再把D失败
   隐藏在P响应后面；不增加超时重算策略，不改变正常响应体/stream消费方式。

不变量复核：通知/并行深度不转移KV ownership；Direct、Host源释放仍由原物理fence
和全TP提交触发；完整workset及pin保留；Host失败不隐式重算；CPU分配/Radix仍在
安全scheduler边界；新的通知只改跨机控制，不改变TP1数据通路。异常HTTP cleanup
不能用关闭socket替代DMA fence。成功、容量不足、取消/重试沿原生命周期处理。

CPU门禁已通过780项（2项GPU用例跳过）+56项launcher unittest；独立审计GO允许
r8有界三路径smoke。审核者独立运行Router109项测试通过，主agent复核HTTP取消路径。
当前尚未宣称停滞已被GPU验证消除，也没有新的SWE正确率/吞吐结论。

r8三路径smoke已通过并由独立审核复核：P→D Mamba源/目的72/72 rank记录hash
一致；Direct/Slow各8rank Attention前缀与Mamba checkpoint一致，均复用8192tokens，
16token输出等于各自完整重算参考。P2D Host完成8rank D2H/源P释放/H2D；D2P Slow
完成8rank D2H/源D释放/H2D/组提交释放。权威ledger D2P为空，P2D为8rejected+
1consumed，后者loader_acks含0..7。未出现错误/隔离；不是正式性能结果。
独立审核GO后启动r9，但在SWE开始前主动停止：发现多节点启动器遗漏单节点已有的
`SGLANG_TIMEOUT_KEEP_ALIVE=120`，服务端默认5秒短于Router连接池30秒。
补齐该设置，独立审核GO；不改变请求超时/传输fence/重试策略。它解释了一种潜在
提前断连风险，但r7原始D异常未捕获，不能据此宣称唯一根因已经证明。
r9不是性能结果。下一轮r10沿用正常容量c64 SWE500，检查持续运行时恢复队列、
P→D交付、HTTP错误与实际完成进度。

r10：57项launcher unittest通过后启动；正常容量Direct/Slow smoke通过，
16:59:37开始载入500题并运行c64。截至17:01:14的早期诊断（不是最终吞吐）：
TP0业务Slow选中76个、H2D完成75个、余1个在途；匹配75个从选中到H2D完成
平均约1.69秒/P90约2秒/最大3秒（日志秒级时间），选中深度确实达到4/4。
Direct start 197次，arrival→start平均571.5ms/P90 868.2ms；仍包含TP调度/预留，
不能把目录扫描优化等同于完全消除准入延迟。17:00:46 Direct控制周期平均7.45ms，
旧r7后期约630ms；两者负载/历史规模不同，不作为正式性能加速比。
当时P日志无ERROR/Traceback，Prefill和交付继续。r7停滞发生较晚，必须继续长测，
不能用该初始窗口宣称长期稳定性已验收。

## r11 重新验证（2026-09-18）

- 用户授权重跑；17:52停止r10的owned服务，确认a10仅原有ComfyUI、a11无GPU进程。
  r10遗留64个工具容器已停止并保留文件，不作为完成的500题结果。
- r11使用上述D follower有界Slow推进和本地D TP abort邮箱；ratio=0.5、拥堵重算
  关闭，其余沿用TP8/EP1、c64、BF16、静态显存0.8及同一500题source-order。
- 启动实际Attention容量：P每rank1,928,640 tokens，D每rank1,948,608 tokens；
  P每rankMamba616 slots。该容量变化必须与控制优化一起标注，不能将吞吐差值
  全部归因于单一代码优化。
- 正常容量Direct/Slow两轮smoke通过：各复用8192个page-aligned父tokens，后续16个
  输出token与各自完整重算参考一致；Slow ledger为consumed，writer/loader及source
  release覆盖全部8rank。本次未单独再次制造P→D Host容量背压。
- 17:56:34载入SWE-bench Verified500开始完整评测（含verifier），不是已完成结果。
  协调器会监督服务退出并清理本run。重点复核D→Host排队/释放延迟、非running KV、
  P/D Forward、快慢路径和完成进度。跨主机响应里的inference_time仍不作为吞吐依据。
- 结果目录：`runs/dualpd/qwen35-122b-a10p-a11d-tp8-c64-r11`；
  本机/远端服务日志：`/tmp/dualpd-multinode/qwen35-122b-a10p-a11d-tp8-c64-r11/`。

阶段监测18:07:43–18:09:48（约125秒，非最终性能结果）：

| 指标 | P | D |
|---|---:|---:|
| 每rank平均Forward/墙钟 | 42.3% | 88.3% |
| Attention KV平均使用率 | 8.2% | 49.2% |
| Attention KV采样范围 | 4.7%–18.5% | 36.7%–56.2% |
| Decode running（逻辑TP组） | 不适用 | 平均11.1，范围5–17 |
| Decode吞吐（整个TP8组，计数器差分） | 不适用 | 439.2 tokens/s |

18:07:44前300秒D2H每rank墙钟平均2.76秒/P90 4.51秒/最大9.47秒；
TP0 Host offer到D2H start排队均值9.28秒/P90 21秒/最大29秒。
当时尚未全TP释放的41个parent合计693,952逻辑tokens，其中Slow尚未启动28个、
已启动4个、Direct或工具等待9个。恢复在持续推进，但非running占用仍显著，
不能宣称完全消除控制/卸载排队。18:09时完成6/500，verifier成功5；样本很少且
短任务先完成，不用于估计最终正确率。该阶段P/D日志未发现ERROR/Traceback/隔离。

## r12 D2H本地推进与启动预注册（开发/验证中）

用户批准修复r11中Slow排队。此次不改路由、workset、拥堵重算、Host容量和模型参数。

- `AgenticDHostStagingClient.progress`统一rank0/follower/TP1：首次共享grant之后，
  活跃D2H只读本地event/future，不再每块读取NFS ledger；完整复制之后仍按原
  `complete_host_write`全TP CAS提交。无有效CUDA fence时沿原quarantine保留源端。
- 一次visit可回收完成块、移交CPU收尾并提交最多一个新块；保持每snapshot一个DMA、
  全局4条lane及原CPU future/bounce fence。多节点启动配置chunk从256改1024tokens，
  不改变单节点chunk默认值，不批量排入整条长请求。
- 复用registered-window预热与complete/failed记录。D每rank仅注册自己的16GiB
  D→P源arena，P每rank仅注册自己的8GiB P→D源arena。不打开远端memfd路径。
  16个rank全部成功才启动Router/smoke/业务；缺项、失败或超时禁止发请求。
  这里是CUDA Host注册预热，不是取消逐snapshot NIXL RDMA memory registration。
- 成功：event与CPU提交完整→全TP durable→D释放；取消/超时：允许当前有界写入
  安全收敛，最终CAS拒绝失效提交并沿原writer-drained流程回收；容量不足仍等待原grant。
  进程退出/无法证明fence时不复用仍在途页，沿原fail-closed清理。

八项检查：唯一owner不变；P→D Direct/Host释放点不变；D→P释放仍等全TP durable；
I/O在原后台worker推进、每visit提交量有界；TP共享提交不变；Attention+Mamba完整
提交与复用语义不变；需要CPU故障门禁及独立审核GO之后才能开始新GPU验证。

门禁完成：794 passed、2 skipped；launcher unittest 60 passed。独立审核相关测试
21 passed并给出GO。审核指出裸`cuda`在后台线程可能使用默认GPU，已在调用线程
捕获明确rank device，新增cuda:7线程参数回归。注册barrier从旧multinode禁用列表
移除，但仅使用显式本地源路径模式。r11于18:44左右主动停止，非完整500题结果；
旧日志保留。下一轮先完成16rank预热及Direct/Slow正确性smoke再开始c64业务。

r12首次启动在模型启动前被33900端口检查拒绝；无残留GPU/监听进程，端口释放后
不改源码重启为r12b。16/16预热完成，P每rank8GiB约4.3秒，D每rank16GiB约7.6秒。
18:49 Direct/Slow smoke均通过：8192父tokens复用、16输出token与完整重算严格一致，
Slow ledger consumed且writer/loader/source-release覆盖0..7。未制造P→D Host背压，
不等同于完整吞吐验收。普通容量SWE500/c64继续由协调器监督启动。

注意smoke D2H本地wall_ms约52–81ms，但TP3记录的最终日志晚于本地copy完成约12秒；
现有wall_ms在export/ledger提交之前取值，不包含NIXL导出/全TP收尾，不能当作D HBM
总释放延迟。本轮CUDA预注册没有消除逐snapshot NIXL注册，后续须看完整端到端释放。

## r13：组内 TP 控制去掉 NFS 依赖，下一轮 c128

r12b不是成功完成的500题评测。19:09左右P停止推进，19:14 watchdog退出。
抓到TP0后台线程持`TPGroupMailbox._cache_lock`进行NFS `stat`，处于
`nfs_lookup_revalidate`；主线程在`publish_local`等待同锁，其他rank等待广播。
此前313个完整Slow恢复，all-rank D2H完成到P选择平均约30.4秒，而选择到组提交
约2.73秒（日志秒级精度）。此处确实存在控制等待，不能归因为P算力不足。

r13只把组内Direct/Host/producer/cleanup/admission报告移入run+engine隔离的
`/dev/shm/dualpd-tp`。跨P/D的`p2d-receiver`完成确认、Host/lifecycle ledger和
arrival/abort消息仍共享；不修改所有权、workset、4条lane、DMA fence和TP决策。
TP1和没有启用新override的launcher保持原行为，跨主机TP组拒绝本地邮箱。
共享ledger准备/提交仍有开销，不宣称这版已消除全部Host恢复延迟。

验证：805 passed、2 skipped；62项launcher unittest通过；独立审核18项通过、
代码GO。完整说明见`../../../sglang/validation/TP_LOCAL_MAILBOX_R13.md`
（从工作区查看：`dualpd/sglang/validation/TP_LOCAL_MAILBOX_R13.md`）。

截至2026-09-18 19:23 UTC，**尚未启动c128**：a10 PID1754686的主线程已退出，
残留TID1759850卡在内核NFS等待，仍占GPU0约66948MiB；SIGKILL已pending但未退出。
a11已无模型GPU进程。没有终止GPU7的ComfyUI，也没有reset GPU/修改NFS挂载。
必须先恢复NFS/清理该残留并通过GPU ownership检查，不能绕过检查强行启动。

复测入口（其余参数与r12b一致，SWE拥堵重算仍关闭）：

```bash
bash tools/dualpd/qwen35_multinode.sh run --concurrency 128 \
  --run-dir /homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-a10p-a11d-tp8-c128-r13
```

入口保留默认c64；指定c128会写入该run的实际配置/命令。预热全部16rank完成后
先跑Direct/Slow正确性smoke，再启动SWE500和verifier。旧run不同并发不能覆盖使用。
