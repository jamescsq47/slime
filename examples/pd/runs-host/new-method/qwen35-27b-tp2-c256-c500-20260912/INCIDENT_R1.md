# R1 failure and diagnostic R2

R1 source: `/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c256-c500-20260912-r1/c256`.
Not a valid full500 performance/accuracy result. 35 tasks ended,13 resolved;
last task ended2026-09-12 03:38:39UTC. c500 never started.

## Evidence, not an inferred root cause

- All8 Host preregistration contexts completed before workload; no missing barrier.
- P0 TP1 log progress disappears about03:23:25. Its pending P2D transfer was
  already90.967s old when NIXL_ERR_REMOTE_DISCONNECT appeared03:24:54.
- Both P groups' TP1 workers fail at that timestamp. D workers remain alive,
  with Decode/Host activity. No heartbeat-reconnect/node-removal evidence.
- Later P2D transfer timeouts and RouterHTTP500 leave all4enginequeues empty.
  Monitoring wrongly checked only Scheduler fatal, so it kept retrying.
- User requested stop. Supervisor3299728 terminated; allownedGPUthreads and
  runlabelcontainers ultimately exited. NVIDIA UVM teardown temporarilyblocked
  nvidia-smi for severalminutes but recovered withoutreset/reboot. Unrelated
  GPU7 PID1868643 remained untouched. R1logs/traces preserved.

## Bounded repairs

1. Monitor fatal NIXL disconnect/quarantine/TPsubmit and workerinit failures;
   also detect>=300s withoutForward plus repeatedHTTP500 afterbusinessstarted.
   Slowtools/verifier alone do not trigger this check. Failure cancelssequence.
2. Log unreadable physicalhandle once per sender; keep polling and retain
   sourceownership until genuineDMAfence. Never convert disconnect toDONE.
3. Bind dedicated NIXLbootstrap, P2Dconsumer and PDirect worker to ranklocal
   CUDAdevice beforecallback. Context is threadlocal; UCX rkeyimport can select
   firstactivedevice without explicitcurrentcontext. This is contexthardening,
   not proof it caused the disconnect (UCX IPCcopy also uses pointercontext).
4. Diagnostic-only faulthandler threadstacks every120s, exit=False, tolocal
   workerlogs. Runtime defaultsOFF. Does not touch model/harness/fence logic.
5. Snapshot/hash untrackedenginePythonfiles as well as trackedgitdiff so the
   newhelper is reproducible; compare all beforec500. Uniqueparent+caseDockerlabel.

## Ownership mapping / eightcriteria

The incident is P_HBM_OWNED plus physicalhandoff with unreadable senderfence.
Nochange to targetallocation, Direct/Host routing, handoffCAS, TPgroupdecision,
pagealignment, Radixlocks, cancellation or source-release rules. Criteria1–4
preserved by unchangedhandoffs;5removeslogflood and binds workercontext without
addingqueues;6keeps allrankfences;7unchangedstableprefixprotocol;8requires
CPU lifecycle/fault/TPtests and independentGO before diagnosticGPUrun.
R2 must still demonstrate liveownershipconservation and full500completion.
The originaltransportrootcause remains unproven until reproduced or resolved
with evidence; do not report this as a confirmedperformancefix.

R2 rootplanned:
`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c256-c500-20260912-r2`.
Same27BTP2settings, sourceordered500, c256thenc500. DiagnosticstacksON120s;
hashOFF, mandatorypreregistration, noactivecongestionrecompute,2sfasttool.

Final combined CPUtest suite638passed18.43s; independent
`audit_hash_off_prewarm` returned GO for diagnosticR2. Approval is not evidence
that the originalREMOTE_DISCONNECT is resolved. Live status inR2root.

## R2 startup failure / R3 diagnosis

R2 ended beforeworkload: P0rank0 disappeared, P0rank1 reported Gloo
`Connection closed by peer` at04:38:39UTC; monitorstoppedthewholegroup04:39.
No task ran and c500wasnotlaunched. R2rootcause NOTidentified.
Read-only hostkernel/syslog inspection found noOOM/Xid/killentry around04:38;
the shell's laterKilled message is cleanup, notproof of the originalcause.
The rank0periodicfaulthandlerdumpwastruncatedmidframe; exit=False excludes an
intentionaltimerexit but notallnativeinterference. R3disablesthisnewtimer.

R3adds optionalservertracing (PD_TRACE_PROCESS_SIGNALS=1):
setsidstrace-ff--seccomp-bpf-ttt withprocess,signal,prctl syscalls, perprocesslogs.
NoGPU/harness/ownershipchange. CPUtestsverifyexit37,SIGTERM andgroupdescendant
cleanup; full641testspassed23.93s. Existingtestscoverall8designcriteria and
independentauditGOisrequired. Tracing capturesinternal killcalls andexitcodes;
it cannotguaranteeexternalsenderPIDforSIGKILL. Traceoverhead makesR3diagnostic,
notfinalperformanceacceptance. Parentregistration/hashoffsettings unchanged.

R3 started14:28:30UTC, supervisor919877. BothPgroups became ready and P0
remainedalive beyondR2's failureinterval; noinitialfailure reproduced. User
changedtoolthresholdto1s at14:35UTC, so R3wasstoppedbeforeanyworkloadtraffic.
This is intentionalconfigurationcancellation, notR3transportfailure.

R4 retainsR3diagnostics(trace1, stacktimer0), changesonlyfasttoolthreshold2→1s;
Directhandshake remains1s. Noactivecongestionrecompute, hashOFF,
allHostregistrationbeforebusiness unchanged. Launcher/supervisor/actualtracer
CPUtests24passed0.31s; engineunchangedfrom641passsuite andR3auditGO.
Independent audit_hash_off_prewarm returned GO forR4 afterR3cleanup.
Outputroot:
`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c256-c500-tool1s-20260912-r4`.
R4 launched14:38UTC, supervisor945597. Preflightconfirmsfasttool=1.0s,
handshake=1.0s; c256 then c500 only afteracceptance. R3cleanupfullycompleted
beforeR4launch; noownedGPUorcontainerresidue, cotenantGPU7untouched.

## R4 monitoring: Slow recovery stall, 15:09 UTC

Business started after all eight registration contexts completed, at14:49:58.
At15:09,44episodes had ended. No NIXL remote disconnect reproduced, but105
authoritative Host entries remained HOST_READY (31.64GiB); at15:07 the oldest
was985s old. P0/P1 manager host_ready counts47/58, H2D loads0 and lanes0/4.
D running fell into single digits. Both P physical KV pools were almost empty.
This is not a saturated-bandwidth result and cannot be accepted as a valid run.
Supervisor945597 was sent SIGTERM to stop its owned experiment; c500 must not run.

The initial read-only monitor incorrectly inspected only host.json entries.
Current SharedHostStagingLedger stores authoritative entries in events/*.json,
under the entry field. The main file is normally empty. This invalidates the
earlier "Host empty" observation and exposes the same omission in supervisor
validate_finished; both directions need event-aware, fail-closed validation.
Independent reviewer confirmed the omission. TP Slow admission/recovery cause
is still under investigation; arena_domain != recovery_domain alone is valid
two-stage routing and is not proof of an ownership bug.

### Independent read-only audit: cross-P stale TP admission

The reviewer identified seven persistent d2p-host mailbox records, each with
rank statuses [0,0], all cross-P snapshots. Example890975...:0 was assigned
Host1→recoveryP0 at14:50:39; oldP1 logged stale Host abort ignored on both ranks
at14:50:40. Examplecafc298...:4 redirected P0→P1 at14:51:41 and oldP0 logged
the same stale abort on both ranks. Old HTTP cancellation removes the Req from
agentic_kv_waiting_queue, but not agentic_tp_host_active_requests. Status
reduction defaults to0 for the now-missing Req, leaving a permanent prepare
command; the forced TP queue scan can then exclude other remote waiters before
they reach gate_request/remote_map. One group's four active slots are occupied.

Required fix: group-safe retirement of metadata-only obsolete admission,
recovery-domain-aware selection and remote-ready discovery. Physical claims,
leases and DMA must still follow existing cancellation/fence semantics. Host
TP mailbox keys also currently lack recovery-P isolation; old-group publish
or clear must not overwrite the new group's state. Do not globally alter
Direct/P2D cross-endpoint mailbox namespaces. Add redirect/cancel/mailbox-race
tests plus authoritative event-ledger validation tests before another GPU run.

46episodes ended before shutdown. At15:11 UTC the owning supervisor is still
waiting for worker teardown; several worker threads are in UVM destruction
or NVIDIA driver rwlock waits. No GPU reset or cotenant process termination
has been attempted. Cleanup must be verified before relaunch.
