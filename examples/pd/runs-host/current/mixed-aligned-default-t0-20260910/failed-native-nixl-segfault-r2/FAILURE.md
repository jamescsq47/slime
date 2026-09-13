# 无效运行：原生 NIXL 建链段错误

配置：原生 Mooncake，Qwen3-8B，Mixed 1:1，2P:6D，c512，temperature=0。
启动端口冲突已解决；本轮在业务预热阶段 P0 的 scheduler 崩溃。

`prefill-0.log`：`bootstrap_thread -> _add_remote_peer -> loadRemoteMD ->
ucp_ep_rkey_unpack`，SIGSEGV / exit -11。随后 D 报 prefill down，Router 500/503。
上游 https://github.com/ai-dynamo/nixl/issues/1986 有同栈报告，但其 peer restart
触发条件尚未在本轮证实，不能直接等同根因。

控制器检测错误后发送 TERM；旧 launcher 在前台等待 inference Python，导致
shell 延迟执行 cleanup trap，残存单 P 又产出了完整时长的 boundaries。
这些统计不是健康的 2P:6D，全部不能作为性能结果。最终GPU进程已清理。

下一次只修改实验控制：inference 独立进程组并可中断 wait，纳入原 cleanup；
监测原生 segfault/child crash；增加 UCX info 日志。暂不更改 SGLang/NIXL源码、
传输后端、显存比例、工作负载和推理参数。干净重跑确认是否复现。
