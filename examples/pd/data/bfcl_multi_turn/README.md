# BFCL Multi-turn Base engineering probe

Source: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
Pinned checkout: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.

This probe uses the official Base questions (200 cases), function documentation,
classic plaintext/Python-call system prompt, stateful simulated tool execution,
and `multi_turn_checker`. No search credentials or external API are needed.
The official checker verifies each user turn's state and response requirements;
producing an answer or finishing the loop is not a correctness pass.

The local inference adapter is not an official leaderboard model submission.
It accepts only allowlisted calls with literal arguments (bare argument names
become strings, matching BFCL's parser), executes tools in
per-task network-disabled Docker containers, and keeps the message history.
Qwen's native chat template may drop older reasoning when a new user turn starts:
the probe records exact input/output token IDs and prefix comparisons rather
than assuming that retained messages imply reusable KV.

Configuration: `configs/experiments/bfcl_multi_turn_probe.json`.
Launch: `bash scripts/baseline/run_bfcl_multi_turn_probe.sh` from this PD directory.
Default: Qwen3.5-9B, GPU0 TP1, first 20 source-order cases, 4 concurrent agents,
temperature=0, 8192 tokens/call, 40960 context, at most 20 calls/user turn.
Context overflow is an explicit error, not silent history truncation.
Tool compute time and Docker RPC time are recorded separately.
Malformed calls end the current user turn, not the entire task, matching the
official Base handler. They are recorded separately from infrastructure errors.

This changes no PD ownership or transport path (invariants 1–7 unchanged/not
exercised). Relevant isolated executor/parser/checker/fault tests and an
independent audit are required before launch (criterion 8). This small dataset
probe is not a 300+1200-second serving performance acceptance experiment.
