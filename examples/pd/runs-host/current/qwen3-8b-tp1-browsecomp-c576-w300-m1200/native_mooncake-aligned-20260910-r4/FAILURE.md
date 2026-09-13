# Startup-only failure — not a throughput result

The retry never reached business warmup. Decode worker 1 exceeded the stock
300-second scheduler watchdog while creating/registering NIXL resources.
At 01:41:47 UTC its log reports `Scheduler watchdog timeout`; the HTTP worker
briefly became ready at 01:41:50, then received SIGQUIT at 01:41:52. The launcher
reported `decode-1 exited before becoming healthy` and cleaned up its services.

The watchdog's native stack identifies `register_memory` in NIXL. An earlier
read-only stack capture of Prefill worker 3 identified UCX backend creation
inside libibverbs. This is an initialization delay, not a measured Decode stall.

All TP1 workers automatically selected NUMA 0, even GPU6 (physically NUMA 1).
GPU6's P worker had approximately 122 GiB resident on NUMA 0 while that node
had only about 8 GiB free; NUMA 1 retained hundreds of GiB. This is evidence of
misplaced Host allocation, but does not prove that it alone caused the driver
registration delay.

The cleanup trap terminated only run-owned process groups. Driver teardown
took additional time; subsequent checks found all owned schedulers gone and
no GPU compute processes before retry r5. No GPU reset or unrelated process
termination was used.

Retry r5 uses a run-local launcher copy with explicit GPU-local `--numa-node`
and `--watchdog-timeout 1200`. All engine source, model, data, concurrency,
cache capacities, sampling, and 300+1200-second measurement settings remain
unchanged. CPU/NUMA placement is a configuration difference from older rows
and must be disclosed in any comparison.
