# SciAgentGym offline scientific tasks

This directory contains a CPU tool-level probe and a colocated-only model
harness. They are separate experiments: original `main()` demos are not model
trajectories. Neither runner reports official benchmark accuracy.

## Real-model functional run

`bash examples/pd/scripts/baseline/run_sciagentgym_colocated.sh` runs Qwen3-8B
on ten original questions (IDs 1,16,20,25,26,28,31,32,35,36), once each in source
order, with four agents in flight. Configuration is pinned in
`configs/experiments/sciagentgym_offline.yaml`: temperature 0, 8192 generated
tokens per turn, 32768 total added tokens including tools, 40960 context,
ten turns and a 60-second per-tool timeout. Case 34 is excluded following the
resource-limited CPU probe. This is a functional smoke, not steady-state
performance acceptance (which requires 300+1200 seconds).

The model chooses the tool names and arguments. `loader.py` whitelists original
questions and tool schemas without answers/reference solutions. `worker.py`
uses upstream `load_tools_for_case` and `GenericFunctionTool` in a private
per-agent directory; scientific functions are not rewritten. Original case 35
tools include symbolic derivation and answer comparison; comparison standards
must be supplied by the model, not injected by this adapter. Figures remain
artifacts: this text-only model does not see their contents. Errors are returned
as observations, and exact model/tool-suffix token IDs are recorded in metadata.

Tool execution is CPU-only, one call at a time per agent, with four agents
concurrent. Runtime and RPC wall time are recorded separately. Workers have
8 GiB address-space and 128 MiB file-size limits and are killed and reaped on
completion, cancellation or timeout. The audit hook is defense in depth for
reviewed tools, not an OS sandbox for arbitrary code. No PD lifecycle or serving
transport changes are made; custom PD mode is explicitly rejected.

### Additional original questions

`configs/experiments/sciagentgym_offline_more.yaml` selects ten different cases:
10,13,21,22,23,24,27,30,33,34. Run with the same model/limits:

```bash
WORKLOAD_CONFIG=/homes/siqic/slime/examples/pd/configs/experiments/sciagentgym_offline_more.yaml \
REQUESTS=10 \
RUN_DIR=/homes/siqic/slime/examples/pd/runs-host/baseline/sciagentgym-qwen3-8b-c4-n10-more-r1 \
bash examples/pd/scripts/baseline/run_sciagentgym_colocated.sh
```

These tools passed source review and isolated worker import checks. Cases
15/17/19 are excluded because their upstream tools evaluate model-provided
expression strings; 14/29 require the removed `scipy.special.sph_harm` symbol.
No baseline dependency versions or upstream scientific code were changed.
Case34's whole demo previously hit its CPU limit; real model-selected calls
remain bounded by the same 60-second per-call timeout. Case13's upstream query
may require a database that is not initialized by its declared tools. Tool
observations must be inspected for nested `metadata.error`, not just `ok`.

## Upstream reused without modification

- Repository: https://github.com/CMarsRover/SciAgentGYM
- Reviewed commit: `e9dbbea4369d67694e38bf8be67bedbcaf9e9300`
- Checkout on this node: `/homes/siqic/data/SciAgentGYM`
- Dataset: upstream `dataset/refine_merged_single_questions.json` (48 cases).
- Selected case IDs: `1,16,20,25,26,28,31,32,34,35,36`.
- Loader: upstream `gym.core.tool_loader.load_tools_for_case`.
- Computation and parameters: the corresponding upstream tool module's own
  `main()` demonstration. These demos can include additional scenarios beyond
  the original question; this is **not** a replay of an actual model trajectory.
- No hand-written scientific solver, synthetic task, artificial sleep, altered
  problem size, gold-output substitution, or network search.

The local wrapper times declared tools using `perf_counter`. Nested calls to
other declared tools are included in their outer call and not double counted.
Module loading is timed separately; dependency import and process startup are
not charged to individual tool calls. Demo duration contains tool duration:
do not add them together. `returned` only means execution returned normally,
not that the scientific answer was checked or judged correct.

## Run

From `/homes/siqic/slime`:

```bash
/homes/siqic/anaconda3/envs/pd_baseline/bin/python \
  examples/pd/data/sciagentgym/profile_offline.py \
  --output examples/pd/runs-host/baseline/sciagentgym-offline-tools-r2
```

Use a fresh output directory. `--cases` can select IDs from the reviewed list,
and `--repeats` controls complete repetitions of the upstream demos (default 3).
`--case` is the internal child-process entry point, not a standalone launcher.

Each case runs in a fresh temporary working directory, with GPU visibility
disabled, requested BLAS/OpenMP thread count 1, an 8 GiB address-space limit,
110 CPU-second limit and 120 wall-second timeout **for the whole child**, not
per call. Temporary tool-generated figures/data are removed after the case;
source hashes, timing records and textual logs remain in the result directory.
The runner kills and reaps a timed-out child. A resource-killed child may not
write its final case JSON; use the parent manifest to account for all cases.

Reviewed modules use local numerical libraries. A Python audit hook forbids
common network connections and subprocess creation during their execution.
This is defense in depth for reviewed upstream code, **not** a security sandbox
for arbitrary model-generated code. Trusted NumPy/Matplotlib initialization
precedes the hook so that local font discovery can run. No packages are installed
or changed by the probe; existing environment dependencies are reused.

## Harness correctness requirements

Reuse upstream task text, tool schemas, function loading and
`MinimalSciEnv`/`GenericFunctionTool`; do not feed `answer`, `golden_answer` or
reference solution steps to the model. Keep private per-agent files so paths
returned by one tool remain available to later calls without cross-agent
collisions. Preserve image-output semantics or explicitly select a text-only
tool subset; a text-only question may still offer plotting tools.

A slow upstream demonstration does not prove that Qwen will choose that tool,
or use the same parameters. Measure actual model-selected tool calls before
claiming this workload naturally has a given slow-path fraction. Long scientific
computations should have bounded CPU concurrency, with execution and queueing
time recorded separately.

No PD snapshot ownership, workset, transport, TP or scheduler code is touched.
