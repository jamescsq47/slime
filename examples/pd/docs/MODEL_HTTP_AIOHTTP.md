# 实验模型 HTTP 传输修复（2026-09-14）

`PD_MODEL_HTTP_TRANSPORT=aiohttp` 让实验进程的模型 HTTP 请求使用 aiohttp
连接池。HTTPX 仍负责请求构造、JSON 字节、headers/cookies、响应解码和状态异常，
不再使用 HTTPX/httpcore 网络连接池。共享 `slime/utils/http_utils.py` 和外部
harness 没有修改；原重试次数、间隔、payload 和 generation 身份保持不变。

入口为 `examples/pd/inference.py`，实现为 `model_http_transport.py`。
环境变量未指定时仍为 `httpx`，不切换其他 agent 的实验。9B baseline 启动脚本
`scripts/baseline/run_qwen35_9b_tp1_swe_verified_500_colocated.sh` 默认启用 aiohttp；
显式设为 `httpx` 可复现原传输层。`config.json.model_http_transport` 保存实际配置。

## 连接与错误处理

- 池容量沿用原公式；当前 c500 对应最多 1000 个连接，不改变 agent 并发上限500。
- 保活30秒；pool等待5秒，connect10秒，write10秒，read沿用CLI（默认3600秒）。
- 不跟随重定向；禁用环境代理和 aiohttp 内部重试，所有重试只在原共享循环发生。
- buffered response 完整读取后立即归还连接；异常和取消同样归还容量。
- 实验正常退出或异常结束时关闭客户端。异步取消不会被当作网络错误重试。
- 保持 HTTPX 的状态异常、错误正文和 JSON/普通文本处理；gzip仅解压一次。
- 当前已验证两个环境的 aiohttp 3.14.3。使用了其 `_retry_connection` 属性和
  `BytesPayload.write_with_length`；升级 aiohttp 后必须重跑传输测试。

注意：传输层没有隐式重发，但原共享循环仍会重试网络失败。
如果服务器已执行、响应却丢失，保持相同 request identity **不自动等于 exactly-once**。
本次不改变原重试策略；不能据此宣称消除了原系统所有模糊失败下的重复执行风险。

## 验证

`tests/test_model_http_transport.py` 共11项：wire payload/headers/cookies一致性，
gzip、分片/截断正文、状态/文本/重定向、显式重试身份，连接失败/超时，真实阻塞
socket写超时，pool超时、排队/在途取消，退出清理，2×500请求无隐式重复及连接复用。
`pd_mamba_baseline` 与 `pd` 两个环境均11/11通过。
连同 SWE harness、baseline Mamba容量/取消/TP和启动配置回归共127/127通过。
启动脚本 `bash -n` 通过。

独立审核 `/root/audit_model_http_aiohttp`：**GO**。审核者独立重跑11项测试，
另以20次截断POST验证恰好20次服务端接收、取消及真实session清理，无阻断项。

本次未改 snapshot ownership。八项门禁逐条检查：1唯一所有者、2P→D Direct释放、
3P→D Host释放、4D→P Host释放、6TP原子性、7父KV复用协议均未修改；
5仅去除客户端HTTP池瓶颈，不添加计算/传输等待；8测试通过后要求独立审核GO再启动GPU。
本轮是 colocated，不创建 PD snapshot，ownership统计不适用。

正式测评为 SWE Verified 500个不同任务各一次，不是循环补样本的300+1200实验。
沿用模型 Qwen3.5-9B、TP1、8卡、c500、静态显存0.8、原生Mamba比例，
原配置8k/64轮及temperature0.6/top_p0.95/top_k20/min_p0。关闭PD/HiCache/Mooncake。
保留所有旧HTTPX结果与本次aiohttp原始数据，实际推理收益等待完整测评，不使用客户端
微基准速度比代替正式收益。
