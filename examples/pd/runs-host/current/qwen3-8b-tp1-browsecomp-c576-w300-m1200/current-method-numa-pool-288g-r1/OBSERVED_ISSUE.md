# 本轮观察到的终止/续轮不一致

本轮不能标为零错误验收；完整运行捕获1次Router 500。未在运行期间改动代码。

请求generation：`a00d9a53c61245e5b0639ee1c4cd7afe:7`。

1. 2026-09-09 23:53:56，P3完成P→D交付并释放23168 tokens。
2. 23:54:11，D0记录`request_seen`，输出268 tokens，随即记录
   `final_skip finish_reason=stop output_kind=terminal`。该分支不创建D→P snapshot。
3. 同秒Agent却提交generation8，prompt29336 tokens，声明generation7为parent。
4. 2026-09-10 00:04:11，Router等待该parent路由600秒后TimeoutError，HTTP500。
5. Agent随后重试同generation。其他请求和两方向Host数据流继续推进。

这不是已分配Host extent丢失、池满或DMA失败；该parent没有进入Host分配流程。
该Agent未完成，最终requests.jsonl没有其完整输出，无法核对触发D终止判断的
268个输出tokens；不能仅凭日志认定是Agent还是D分类器错误。
`decode_kvcache_offload_manager.py`终止分支及Agent
harness均不属于本次Host修改范围，未擅自修改。也不声称旧实验一定不会触发。

证据：logs/decode-0.log中的snapshot记录、logs/prefill-3.log中的g7释放记录、
logs/router.log中的generation8 arrival与TimeoutError。
