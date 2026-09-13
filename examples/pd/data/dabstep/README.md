# DABstep real-task probe

Official sources:

- Dataset: https://huggingface.co/datasets/adyen/DABstep
- Baseline: https://huggingface.co/spaces/adyen/DABstep/tree/main/baseline
- Agent library: https://github.com/huggingface/smolagents

This engineering probe reuses original DABstep questions/guidelines/input files,
the official `chat_llm_task_prompt` constant, and smolagents `CodeAgent` 1.24.0.
It does not implement a new scientific solver or force slow computation. It is
an adaptation, not an exact reproduction of the older official baseline:
local Qwen3-8B replaces its hosted model, and a restricted persistent Docker
Python executor replaces host-local execution. The current smolagents system
prompt is used. Qwen reasoning remains in raw logs but is removed before
CodeAgent parses executable answer text. CodeAgent's code-marker stop strings
are applied locally after the reasoning block because they can occur within
reasoning. Only the first action goes to the interpreter; model-fabricated
observations and later actions in the same generation are discarded. Raw logs
and token counts still include that discarded generated text; this is another
reason not to treat this probe as a steady-state serving benchmark.

The source data is under `/homes/siqic/data/DABstep` and official baseline code
under `/homes/siqic/data/DABstep-upstream`. Exact Hub revisions, first-six-dev
task IDs, Docker image ID and input hashes are recorded. CC-BY-4.0 dataset
attribution remains with Adyen/Hugging Face. No leaderboard submission occurs.

## Run

From `/homes/siqic/slime`:

```bash
bash examples/pd/scripts/baseline/run_dabstep_probe.sh
```

Use a fresh `RUN_DIR`. Configuration is
`configs/experiments/dabstep_probe.json`. Single GPU0, Qwen3-8B, TP1,
mem_fraction_static0.80, context40960, temperature0,8192 maximum generated tokens
per model call, two concurrent agents, ten CodeAgent steps,60 seconds per Python
call. CodeAgent can make an additional final-answer generation after exhausting
its step limit; records distinguish `max_steps_error` from success. This probe
does not impose the SciAgentGym whole-trajectory32768 budget. It is not a
closed-loop300+1200 serving benchmark and is not yet a registered PD harness.

Inference uses unchanged `pd_baseline`. Agent-only dependencies are installed in
`/homes/siqic/.venvs/dabstep_smoke`, without changing baseline packages. The
existing pinned `slimerl/slime` image provides pandas3.0.1/numpy1.26.4; no new
per-task images are needed.

## Execution and measurement boundaries

- Each agent gets a persistent Python namespace in its own container. Only the
  seven official context files and read-only worker are mounted, never task
  answers, credentials, Docker socket or home directories.
- Network disabled, nonroot UID65534, all capabilities dropped, no privilege
  escalation, read-only root,4GiB RAM/no swap,2CPU quota,128PIDs, bounded tmpfs.
  Environment variables are cleared. `additional_authorized_imports` describes
  tool capabilities to the model; Docker, not that list, is the security boundary.
- Python execution timer covers actual code (including imports and file reads),
  not container/model startup. Parent wall time also includes RPC. Failed calls,
  timeouts and final-answer-only calls must be reported separately. No sleeps
  or inflated inputs are inserted.
- Exact per-task model messages, token-usage counters, Python code/output and
  durations are retained. This OpenAI-chat/CodeAgent probe does not claim exact
  token-prefix preservation or agentic KV transfer correctness.
- Each container is removed on success/error/timeout. Launcher uses a unique
  run label for cancellation cleanup and tracked process groups for its model
  and agent driver. Cleanup errors remain explicit in records.
- Independent audit GO plus CPU isolation, state persistence, timeout, startup,
  cancellation, cleanup-failure and mock-model CodeAgent tests are required
  before GPU runs. No existing PD ownership, router, allocator or TP logic changes.
