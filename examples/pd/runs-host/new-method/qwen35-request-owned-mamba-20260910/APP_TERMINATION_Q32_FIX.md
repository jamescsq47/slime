# SWE application termination and Q32/8 experiment (2026-09-11)

User authorized fixing the terminal-classification mismatch and increasing
Slow-congestion hysteresis to32/8; no harness/sampling/model/pool changes.

## R4 diagnosis and retention

Old run: `/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-request-owned-ratio05-20260911-r4`.
Stopped through verified supervisor239352 TERM;260 completed task records
(76 resolved, NOT final full500 accuracy) retained with raw logs/control-final.
All owned GPU contexts exited; other-user GPU7 PID1868643 untouched.

In02:10–02:15,1665 successful P calls computed2898200tokens.155 explicitly
recomputed calls consumed2120845tokens (73.2%);2055296tokens were existing
parent snapshot prefixes, not necessary suffix work.1476 reused calls matched
their advertised snapshot length exactly, with no extra historical-prefix loss.
P was busy but D batches small; prior resident-admission deadlock did not recur.

Three parent generations received next-turn requests after D final_skip;
two inspected outputs were valid fenced shell commands containing
`echo "TASK_COMPLETE"` or `print('TASK_COMPLETE')`. Engine substring matching
classified them as TERMINAL while unchanged Miles parsing executed the command.
Router waited600s for a parent route that was never published.

## Patch and ownership mapping

- Isolated fusion `agentic_kv_lifecycle.py`: opt-in
  `SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION` returns UNKNOWN before token/text
  marker matching. Default off preserves existing Qwen3/other workload behavior.
- This application already publishes tool/final ACKs. UNKNOWN retains a
  provisional snapshot; final ACK uses the existing DIRECT_READY CAS and
  release path. A continued tool call can use the existing Direct/Host path.
- D_HBM_OWNED no longer incorrectly transitions to TERMINAL on a quoted marker.
  Source remains owned until the existing physical fence/handoff or true final.
- Length/abort handling is unchanged. No parser prompt/recovery/harness edit.
- Launcher enables the flag and explicitly exports congestion HIGH32/LOW8
  instead of2P×2lanes default16/4. Same two-sample entry/low-water exit logic;
  no timeout, capacity, transport, eviction or I/O worker changes.
- Preflight records both settings. Installed pd and baseline env source unchanged.

## Eight acceptance checks

1. Unique owner unchanged; classification only chooses existing lifecycle entry.
2. P→D Direct source release unchanged, still after physical completion.
3. P→D Host source release unchanged, still after durable.
4. D→P Host source release unchanged, still after durable.
5. No added blocking wait/control work; classifier bypasses parsing in opt-in mode.
6. Existing rank0 final decision and TP group release/fences unchanged.
7. Parent Attention/Mamba boundary unchanged; explicit recompute remains accounted
   separately. Final marker no longer discards a still-needed parent.
8. Engine lifecycle/cancel/capacity/TP/fault tests and independent audit gate
   required before GPU launch. Final suite542 passed in22.56s, including20 new
   classifier/production-entry/final-fence tests. Independent audit GO for
   monitored formal launch (independently ran the first11 classifier cases).

Known limitation: if final ACK arrives after staging/sent, existing final handling
does not forcibly delete in-flight or Host-owned data. Late-final Host retention
must be measured, not claimed solved by this classification patch.

## New evaluation

Same full500 distinct SWE Verified tasks once, c256, Qwen3.5-9B TP1,2P:6D,
static.80, Mamba ratio.5, page64,8k/64turns, temperature.6, unchanged harness.
This is a finite full-dataset evaluation, not a300+1200 closed-loop benchmark.
Measure full completion/accuracy, live forward token counters, recompute tokens,
Direct/Host ownership releases, route timeouts and late-final Host residue.
Two changes are combined; do not attribute all performance change to either one.

Launched r5 under verified supervisor1673673:
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-appfinal-q32-20260911-r5`.
Preflight confirmed unchanged harness/workload hashes and records Q32/8 plus
app-owned termination. Startup/monitored run, not yet completed results.

## Early observations (02:46–02:48 UTC; not full-run conclusions)

All services ready02:38:07; dataset loaded02:38:33.200. At02:46,33 tasks
completed,17 resolved; neither value is final accuracy. No route timeout,
engine traceback or recurrence of bound-but-not-rematched deadlock observed.
Thirty app_final_release events at02:46, max0.562s. A later point-in-time check
found7 unremoved final markers and0 matching live Host extents; this is not a
proof that late-final Host retention cannot occur later.

Native realtime counter window02:45:05–02:46:06 (61.403s):
P compute10608.42token/s total; P forward90.65%/94.90%; D2991.52token/s total,
498.59/card; D running20.24/card; D forward99.14%; D Attention KV28.53%.
These are live compute counter deltas, not completed-response token smoothing.

Approximately phase-aligned early5min completion cohorts:

| Metric | R4 Q16/4,01:58–02:03 | R5 Q32/8,02:40–02:45 |
|---|---:|---:|
| Successful P calls |4207|4168|
| Actual Prefill tokens |3227951|3206302|
| Explicit recompute calls |205 (4.87%)|159 (3.81%)|
| Tokens in recompute calls |1192285 (36.94%)|1062900 (33.15%)|
| Extra old-prefix recompute tokens |1088384|979904|
| D completed output tokens |858435|842284|

The last row is completion-window accounting, NOT GPU realtime throughput.
Windows start~83s/~87s after dataset load, not precisely identical cohorts;
only one run each. Recompute frequency fell but early throughput has no clear
improvement. Longer-context phase and complete500 remain outstanding.
All4001 R5 successfully reused P calls in this window matched their advertised
snapshot lengths (extra historical-prefix miss0); frozen-boundary suffix still
requires normal computation. Q32/8 still enters congestion under the initial
burst and remains blocked until Q≤8; it does not disable recomputation.
