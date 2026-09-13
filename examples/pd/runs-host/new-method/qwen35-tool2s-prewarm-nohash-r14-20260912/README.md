# R14 results: Qwen3.5-9B SWE500, 2-second fast-tool threshold

Full500 finite evaluation; not a replenished 300+1200 closed-loop benchmark.
Business interval: 2026-09-12 01:29:01.847824–02:19:45.940741 UTC.

- `comparison.json`: full-run timing and per-endpoint full/middle counter/gauge statistics.
- `incremental_prefill_summary.json` and `per_agent_incremental_prefill.csv`:
  500 unique tasks, actual Prefill / necessary incremental Prefill / page64 boundary / Decode.
- `h2d_window.json`: copy-completion cohort bandwidth, timings, physical-slot samples.
- `p_stage_window.json`: system-wide lifetime-integrated state occupancy and shell distributions.
- `admission_stages.json`: all12841 recovered Host snapshots joined to Router/P HTTP/API times.
- `extras.json`: unique-snapshot conservation, final ledgers, H2D handoff timings, termination results.
- `preflight.json`, `host_register_prewarm.json`: actual settings and all-eight-context registration gate.
- `swe_bench_profile_summary.json`: unchanged automatic full-dataset profile; its `truncated=0`
  does not mean zero length-limited terminations (100 max_tokens_per_turn,155 max_turns).

Reproduce using `scripts/tools/summarize_swe_pd_comparison.py`,
`summarize_swe_incremental_prefill.py`, `summarize_swe_h2d_window.py`,
`summarize_swe_p_stage_window.py` on the raw run. Additional run-specific read-only
analysis programs are included as `summarize_r14_extras.py` and `r14_admission_stages.py`.

Raw run (not copied to this lightweight directory):
`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-prewarm-nohash-tool2s-20260912-r14`.
Read all `.log*` siblings: the request logs crossed an hourly rollover.

Full results: [main report](../../SWEBENCH_QWEN35_9B_TP1.md),
[middle window](../../SWEBENCH_QWEN35_9B_TP1_MIDWINDOW.md),
[H2D comparison](../../SWEBENCH_QWEN35_9B_H2D_DECOUPLING.md).
All21 pre-existing tables received R14 entries; main report adds five complete R14 tables.
No serving code or harness changed during analysis, and nothing was pushed to GitHub.
