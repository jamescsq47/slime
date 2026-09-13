# Request-owned Mamba: implementation and validation

**Current02:48UTC:** r4 stopped for user-authorized terminal classification
repair and Q32/8 rerun.260 completed records retained; all owned GPU contexts
cleaned, other-user GPU7 preserved. App-owned termination opt-in plus explicit
Q32/8 passed542 tests and independent audit. R5 launched, supervisor1673673:
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-appfinal-q32-20260911-r5`.
Same500/c256/2P6D evaluation, loaded02:38:33. At02:46,33 completed;
app-final releases verified, no route timeout/deadlock observed. Early5min
recompute calls4.87%→3.81% but no clear throughput gain yet. Latest61s native
counter D2991.5total/498.6percard. Still running, not completed results.
See `APP_TERMINATION_Q32_FIX.md`. Prior resident-admission repair remains;
it did not recur in r4. Earlier run statuses below are historical.

## Latest status: stopped and fixing resident-admission deadlock (2026-09-11)

Formal r2 below was stopped at user request;65 completed records retained.
All experiment GPU contexts and256 run-labelled containers were cleaned up;
other-user GPU7 process preserved. It is NOT a valid full500 result.
P0 bound78 requests but never admitted them to compute, holding312/315 Mamba
slots and89344 Attention tokens. The metadata scan repeatedly selected the
oldest16 unallocated waiters, starving later already-bound worksets. This is
an admission head-of-line deadlock, not evidence that the pool ratio is too low.
Fix scope: drain committed TP1 Mamba resident worksets to existing native
Prefill independently of metadata-only I/O admission caps. No new timeout,
pool size, Router policy, state copy protocol or harness changes. TP>1 and
non-Mamba paths keep their previous selection behavior.
Both78-behind16 regressions fail with pre-fix code;8 added tests pass with fix.
Engine agentic suite518 plus native Mamba cache unit tests4 passed (522 total).
Independent audit GO. Diagnostic GPU smoke6/6 raw-token references exact;
12 P2D and4 D2P Host Mamba hash pairs exact; all12 generations released,
both P lease pools empty and all8 get_load endpoints empty at idle. Two Direct
attempts safely rolled back; positive Direct must still be validated in c256.
Smoke: `/tmp/pd-persist/fused-qwen35-resident-admission-smoke-20260911-r1`.
Smoke workers fully exited before replacement formal launch.
Formal r3 launched01:45UTC, verified supervisor214177 (the earlier01:43 launch
did not survive shell startup and wrote no run data):
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-request-owned-ratio05-20260911-r3`.
Same c256/500 dataset/harness/ratio0.5/static0.8; restoredchunk8192/default4096
alignment/digestoff. It is monitored load verification, not yet a completed result.
R3 failed before evaluation: D0 randomly selected torch.distributed port35937,
then TCPStore bind raised EADDRINUSE. Owned GPU contexts cleaned. Launcher now
optionally supplies explicit NCCL ports; this isolated job uses23910..23917,
all checked and disjoint from service/Direct/bootstrap ports. No sysctl change.
Awaiting audit to retry r4; this startup failure is not a runtime queue regression.
Evidence: r2 `STOPPED.md`, raw logs, completed records and `control-final`.

## Configuration

Qwen3.5-9B, TP1, 2P:6D (P0/4; D1/2/3/5/6/7), target c256,
all500 distinct SWE-bench Verified tasks, same external harness and sampling as
the completed colocated c256 run. Static memory0.80, Mamba:Attention ratio0.5,
page64, thinking enabled,8192tokens/turn,64turns, context131072.
Native HiCache/Mooncake off; custom Direct + Shared Host Arenas on.

Engine base `7f22a376487998451087e23ce3a9b275275df62d` in
`/homes/siqic/sglang-qwen35-integration`, with the request-owned patch retained
as `engine.patch` in each run directory. Installed environments are untouched.
Launcher: `scripts/new_method/run_qwen35_fused_swe500_2p6d.sh`.

Important comparison limits: prior colocated uses ratio0.9 and another SGLang
version. This compares deployed systems, not a same-engine, same-pool-ratio
ablation. Finite500 evaluation is not300+1200 closed-loop steady state.

## Implemented

- Frozen stable Prompt checkpoint follows generation ownership. D uses active
  state plus one checkpoint instead of two tracking buffers; P keeps native
  scratch. Capacity estimate P5/D3: D's third slot is the native locked Radix
  copy of the frozen checkpoint (shareable across requests), not just transient
  headroom. Thus this version retains3 physical states for an isolated active
  D request, rather than a strict2-slot design; no additional old checkpoint is
  retained without an owner.
- After replacement checkpoint is locked, retire unreferenced historical Mamba
  checkpoints using native Radix reference rules. Shared Attention is preserved.
- Source release remains after Direct/Host fences. Restore matching Attention
  prefix and Mamba boundary; visible response/tool suffix must be recomputed
  after SWE history removes old reasoning.
- Native retraction copies both active and immutable states; producer fence
  precedes state reads, invalid states cannot resume tracking, and failed restore
  retains backup. This path has CPU tests; GPU retraction not yet exercised.

## Gates completed

Full CPU suite: **552 passed**,20.05s,2 dependency warnings.
Log: `/tmp/pd-persist/fused-qwen35-request-owned-validation-20260910/cpu-tests.log`.
Independent audit `/root/audit_tp_patch_merge`: GO for isolated GPU smoke after
fixing producer-fence ordering; independently28tests passed.
`git diff --check` and both launchers'`bash -n` passed.

## GPU status

First service-only smoke completed; NOT yet a full GPU correctness pass or formal result.
Run: `/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-request-owned-smoke-20260910-r1`.
All first-run GPU contexts exited by2026-09-11 00:15:28UTC; GPU7's other-user
context was preserved. The normal bounded shutdown needed an additional reap
window for one CUDA worker; no old GPU context was left before restarting.
First P allocation:315 Mamba slots (~15.16GiB),990976 Attention tokens (~30.24GiB).

First smoke observations (not performance):
- Synthetic long-context three-turn Direct/Host:6/6 full-recompute references
  match exact output token IDs.
- Unchanged SWE renderer:6/6 shell commands agree;4/6 full messages/lengths
  agree. Both generation1 continuations differ from cold recompute in reasoning
  wording (102vs103 output tokens). Direct and Host outputs match each other.
  Keep this failure; numerical chunk-partition differences are a hypothesis,
  not yet a verified explanation.
- The real public prefixes are307 and551 tokens; snapshots256 and512 are valid
  and do not include the removed reasoning. Observed Mamba transfer hashes match.
- Independent audit:24 request_seen =24 D releases =24 P source =24 P releases;
  all7 Host stages counted5; Direct send/receive/bind/admit counted3;
  two Direct aborts each have drop and completion. All8 live/get_load endpoints
  report zero outstanding requests and physical KV at idle; both P arena usage0.
  Idle Mamba metrics can be stale: do not infer exact slot release from a gauge
  that was last refreshed during Decode.

Second diagnostic (independent audit GO):
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-request-owned-chunk256-smoke-20260911-r2`.
Uses256-token Prefill chunks to align cold/recovered computation partitions.
This launcher override is prohibited outside service-only diagnostics; formal
500 uses8192. Second smoke was stopped: diagnostic chunk256 conflicted with
native deterministic Triton truncation alignment4096, leaving the first309-token
request queued (no Forward). This is a diagnostic configuration error, not a
reverse-state result. All its GPU contexts exited by00:31:41UTC.

Third diagnostic:
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-request-owned-chunk256-smoke-20260911-r3`.
Chunk256 now explicitly pairs with diagnostic alignment256. Formal8192 clears
that override and retains default4096. Opt-in parallel bootstrap retains all-P
readiness before D and all-D readiness before Router, with original PID cleanup.
Independent audit GO and bash syntax tests passed for these launcher-only
changes. Full500 has NOT been launched.

Third diagnostic exposed a real short-tail bug: after a256-token chunk and53
remaining tokens, native caching retains a locked checkpoint256 but clears
`mamba_last_track_seqlen`; final P2D validation incorrectly rejected None.
It also affects normal8192 chunks with short tails. Fixed only the no-new-track
case, requiring exact protected and Radix-path boundaries, unique state, and
positive Attention/Mamba locks; wrong nonempty tracked boundaries still fail.
Added10 regression cases, including real native skip-cache behavior and stale/
unlocked state rejection. Full suite now **562 passed**,18.45s; log
`/tmp/pd-persist/fused-qwen35-request-owned-validation-20260910/cpu-tests-short-tail-fix.log`.
Independent audit GO to repeat smoke. All r3 GPU contexts were released.

Fourth diagnostic launched:
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-request-owned-chunk256-smoke-20260911-r4`.
Same diagnostic256/256 alignment; formal500 still pending GPU acceptance.

Fourth diagnostic completed successfully:
- Real SWE rewritten history: **6/6 raw output-token arrays exactly equal** to
  cold full-recompute references under matched256-token partitioning.
- **16/16** observed Mamba source/destination hashes exact:12P2D,1D2P Direct,
  3D2P Host; no missing source and no engine-error marker.
- This matched-partition control supports computation partitioning as the
  cause of r1's reasoning divergence; it is not a claim of bitwise invariance
  across arbitrary chunk/batch layouts.
- Raw evidence: `raw-token-reference-comparison.json`,
  `mamba-digest-comparison.json`, `swe-stable-checkpoint-smoke.json` in r4.
- P2D Host and native GPU retraction were not exercised by this small smoke;
  their CPU/fault tests passed, but do not label those paths GPU-validated here.

Final independent ownership accounting passed:12 generations all P/D-released;
three Host transfers have balanced stage counts; one Direct success and one
fully rolled-back failure; all8 live engines empty. Audit GO to launch formal.
All r4 GPU contexts were gone by00:49:20UTC; other-user GPU7 job preserved.

## Formal500 run

Launched2026-09-11 00:49UTC:
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-request-owned-ratio05-20260911-r1`.
Full500 distinct tasks,c256,8192/default4096,debugoff,ratio0.5,2P:6D,TP1.
The launcher validates the SWE harness source and workload YAML against the
colocated baseline byte-for-byte and saves their hashes. It uses normal
process-group supervision and exact Docker run labels for cleanup.
First formal startup aborted before any task: P1 bind failed on33720. Verified
`ss -tan` showed an outbound443 connection's TIME-WAIT at local33720, inside
the host ephemeral range32768..60999. All owned GPU contexts were cleaned up.
Move this launcher's service/Bootstrap/Direct ports to23700..23905 (preflight
checked), below the ephemeral range; no system network setting was changed.
Replacement formal run:
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-request-owned-ratio05-20260911-r2`.
Launched00:55UTC; all8 workers and Router ready, dataset loaded00:59:22UTC.
Verified256 run-labelled Docker containers and effectivechunk8192,ratio0.5,
temperature0.6,top_p0.95,top_k20,digest0. At01:01:44UTC,4 tasks finished,
including two verifier passes; this tiny early subset is NOT an accuracy estimate.
Hundreds of live Direct/Host/P2D release events observed with no engine exception,
CUDA error or checkpoint-boundary error. Formal run is ongoing;500 not complete.
The evaluator keeps2-second engine metric samples in memory and writes
`engine_metrics.jsonl` on normal completion. Per-task records are incrementally
durable in`requests.completed.jsonl` and`episode_progress.jsonl`; raw model and
device-timer metrics remain in per-worker logs. No final accuracy/throughput yet.

CPU validation and smoke-r1/smoke-r4 evidence have also been copied beside this
file into`validation/`,`smoke-r1/`,`smoke-r4/` on the shared filesystem. Original
failed diagnostic/formal startup directories are retained on a10's local disk.

Before full500: actual SWE renderer three-turn Direct/Host smoke, full-recompute
output comparison, source/destination state digest and ownership accounting.
Full500 must run with digest debug off and a fresh run directory.
