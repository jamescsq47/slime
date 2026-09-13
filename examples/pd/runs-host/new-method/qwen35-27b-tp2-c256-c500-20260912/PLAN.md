# Qwen3.5-27B TP2 SWE500 c256/c500 launch record

User request: run the new method from SWEBENCH_QWEN35_27B_TP2.md, align
documented colocated baseline settings, sequential concurrency256 then500.

## Baseline alignment and explicit differences

Same documented500 source-order Verified tasks, structured OpenAI shell tools,
external OpenEnv/MilesPR51 harness, glm45 reasoning parser and qwen3_coder tool
parser, thinking=true, temperature0.6/top_p0.95/top_k20/min_p0,8192 tokens/turn,
64turns,131072context. Docker2CPU/4GiB, shell600s, verifier2400s/concurrency16.
Static memory0.80 per physicalGPU. Mamba memory ratio0.9 explicitly matches
the baseline launcher's unoverridden ServerArgs default (not9B experiments0.5).

4P:4D means P groups[0,4],[1,5], D groups[2,6],[3,7], allTP2, preserving the
documented baseline's cross-NUMA TP pairing. Baseline was4colocated replicas,
c128, older engine/page1; requestedc256/c500 and reverse-safe page64 are explicit
differences. Historical baseline raw/harness snapshot paths are absent on a10;
do not claim historical byte identity. Current harness/config/engine are saved
per run and not altered. Engine is isolated sglang-qwen35-integration using pd
dependencies; shared installed environments and other agent's engine stay untouched.

Current SWE method: fast-tool threshold1s (user update2026-09-12 14:35UTC;
R1/R2/R3 used2s), Direct setup1s, failures goSharedHost,
no congestion/fixed-failure recompute (Q32/32 inactive), nativeHiCache/MooncakeOFF.
This supersedes the27B document's obsolete fixed-recompute definition, following
the ongoing user-selected return-all-KV experiment. StablePrompt Attention and
request-owned Mamba checkpoint, page/track64. Ordinary H2D4lanes; the TP1-only
H2D_DECOUPLED optimization is explicitly OFF. No speculation/int8 checkpoints.

Host128/32GiB perP rank,640GiB totalphysical(2groups*2ranks*160GiB).
EachP CUDA context registers288GiB, eachD320GiB; fourP+fourD participants and
four(domain,rank) arena manifests. Content hashingOFF and all-registration
startup barrier mandatory before any workload. No model traffic during preregistration.

## Ownership and acceptance mapping

No transport/state-machine code changed. Topology switches existingTP2 all-rank
admission/claim/prepare/fence/bind/commit paths; no one-rank ownership decision.
Direct source remainsD-owned until group receive/commit, Host source until
durablefence; failure/cancellation retain existing abort/CAS/fence paths.
P2D Direct/Host release and complete-workset budgets unchanged.
Criteria1–6: inherited protocol, no newqueues/fences/evictions; criterion7:
runtime must audit stableprefix reuse separately from page64 boundary work;
criterion8:611lifecycle/fault/TP/Mamba tests passed22.90s;14launcher/sequence
tests passed0.09s, including cancellation/noadvance and missing/duplicate rank,
wrong registeredcapacity, incomplete500, leftoverHost, and fatal errors.
Four Host-registration barrier tests also passed. Independent audit
`audit_hash_off_prewarm` returned GO for this scoped sequence on2026-09-12;
runtime ownership conservation and prefix accounting remain required.

Each run is finite500 tasks, not recycling; report fullcompletion/accuracy and
posthoc300–1500s window separately. CountTP replicas separately fromphysicalcards.
Sequential supervisor will NOT launchc500 ifc256fails/cancels, registration
coverage is wrong, full500rows are missing or Hostfinalledgers remain nonempty.
It signals only its own launcher; existing supervisors perform labelledDocker
and ownedGPU cleanup. GPU7 unrelatedPID1868643 must remain untouched.

Launchers: scripts/new_method/run_qwen35_fused_27b_tp2_swe500_4p4d.sh and
run_qwen35_27b_tp2_swe500_sequence.py. Approved sequence output root:
`/tmp/pd-persist/fused-qwen35-27b-tp2-swe500-4p4d-c256-c500-20260912-r1`.
Live state is recorded in that directory's `sequence_status.json`.
