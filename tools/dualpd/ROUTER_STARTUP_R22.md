# R21 startup failure / R22 launcher repair

R21 did not reach the Direct/Slow smoke or SWE500 workload. Both TP8 model
groups became ready and all 16 Host prewarm participants completed. The
coordinator then exhausted its fixed 120 Router health probes and stopped all
run-owned supervisors. The Router child existed for about 104 seconds; its
service log was empty. Both nodes were verified free of GPU processes after
cleanup. This is a startup failure, not a KV recovery performance result.

A CPU-only `python -u -X importtime launch_late_binding_router.py --help`
probe, with CUDA devices hidden, exceeded the old 120-second window while
in `rpc_wait_bit_killable`. The import log was still in the Transformers
dependency chain. Router imports run before application logging. This
reproduces a startup phase that the former deadline could interrupt; no stack
was captured from the already-stopped R21 Router, so its exact blocked import
cannot be proved retrospectively.

The probe subsequently recorded 148.429 s self-time in
`transformers.utils.import_utils` alone (importtime microseconds converted to
seconds), plus 22.387 s self-time importing `transformers`. This is elapsed
import cost including shared-filesystem waits, not GPU work. The 120-probe
startup window is therefore demonstrably too short for this environment.
The CPU probe exited successfully with help output; cumulative import time
for `late_binding_router` and its dependencies was 245.971 s.

## Minimal change

- Qwen multi-node launcher now records `router_startup_timeout_seconds=1800`.
  Readiness uses a monotonic elapsed-time deadline, not a fixed probe count.
- All P/D/Router supervisor exits are checked each iteration. HTTP 200 is
  still mandatory, all unsuccessful probes sleep, and HTTP protocol errors
  during startup are retried within the deadline.
- Report elapsed time, last probe error and log location every 30 seconds.
  A successful barrier records `router-ready.json`.
- Actual Router child uses unbuffered Python. Optional
  `router_profile_imports=true` adds `-X importtime` only to that child, not
  P/D or the workload. It records cold import cost, not request timing.
- No engine, admission, Host capacity, route, workset or ownership change.
  Existing run-owned shutdown remains intact. Snapshot acceptance criteria
  1–7 are unchanged; criterion 8 uses the preceding engine gate (893 passed,
  2 skipped plus 2 extra cancellation cases) and this launcher gate.

Validation: 71 launcher unittest cases passed, shell syntax and diff checks
passed. New tests cover >120-second successful startup, each service exiting,
deadline exhaustion, non-200 and transient HTTP protocol errors, invalid
configuration, and Router-only profiling isolation. Independent reviewer ran
all 13 Qwen launcher tests successfully and returned GO. R22 keeps R21
c128/SWE500/model/memory settings; launched under a fresh run ID after review.

## R22 startup port failure / R23

R22 completed both model startups and all 16 Host prewarm participants, but
Router's existing port-ownership check could not bind 33902. It safely stopped
both model groups; no smoke or workload ran. After cleanup neither node had GPU
processes, and no listener remained on that port. The transient holder was not
captured, so an ephemeral-client collision is a hypothesis, not a proved cause.
Both nodes use ephemeral TCP ports 32768–60999, which included the old 339xx
listeners. R23 moves only these listeners to checked-free 23900–23903, outside
that range, and derives workload/preflight ports from the same configuration.
Reverse bootstrap stays 61900. Occupied-port refusal is unchanged; no process
is killed to obtain a port and no engine logic or experiment parameter changes.

R23 validation: 60 selected launcher/workload/smoke tests passed; the 4
process-supervisor tests also passed. Independent review separately ran all
14 Qwen launcher tests and returned GO. The fresh c128 R23 run was launched
2026-09-19 16:30 UTC. Startup is not counted as a successful experiment.

R23 physical progress: both models and 16/16 Host prewarm participants became
ready; Router reached HTTP 200 after 185.070 s (beyond the former window).
At 16:48 UTC both TP8 Direct and Slow two-turn smoke cases passed: 8192/8192
expected parent tokens reused, ranks 0–7 committed, and generated token IDs
matched full recomputation. The c128 SWE500 workload supervisor then started.
This validates startup and small-path correctness, not sustained performance
or recovery from a later NIC failure. Raw evidence is in the R23 run directory
and `/tmp/dualpd-multinode/<R23 run ID>/smoke/smoke.json`.

At 16:51 UTC the real SWE500 workload began generating. Early load is NOT
healthy-performance acceptance: at 16:54 P reported 49–52 inflight requests
with ~3% attention KV, while D sometimes had only 1–2 running requests and
~3% attention KV. Snapshot `3672f82102ea428fa37b51659966cc5c:0` completed
Host recovery on all eight ranks, but wall time ranged 25.1–33.9 s against
0.206–0.728 s worker time (worker time is not pure network DMA).
Read-only process sampling at 16:54–16:55 found P scheduler threads in
`rpc_wait_bit_killable`, `open_last_lookups`, and `do_renameat2`, and multiple
control threads in NFS waits. This is evidence of remaining filesystem/control
stalling, not proof of network bandwidth exhaustion or full HBM. No new engine
patch was applied to a running experiment. The startup repair is validated;
sustained progress/performance and recovery after hardware errors are not.
