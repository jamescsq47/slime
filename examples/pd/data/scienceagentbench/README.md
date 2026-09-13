# ScienceAgentBench small real-task probe

Uses original **verified** ScienceAgentBench questions and datasets, Qwen3-8B,
and the upstream Self-debug loop. This is a functional/tool-latency probe, not
a formal serving benchmark and not a registered PD harness.

## Reuse and provenance

- Official source: https://github.com/OSU-NLP-Group/ScienceAgentBench
- Local source: `/homes/siqic/data/ScienceAgentBench-upstream`
- Source commit: `c26e151ed601ba109dc4d35e057ff8e73fec469d`
- Annotation: https://huggingface.co/datasets/osunlp/ScienceAgentBench
- Revision: `9c6e96c9e74572e979b0930ee735041cef528cb7`, verified split, 102 tasks.
- Data archive: official README's SharePoint `benchmark_verified.zip`.
  Password is published upstream: `scienceagentbench`. Do not redistribute
  extracted benchmark data, gold programs or grading artifacts online.
- Reuses official `get_sys_msg`, `write_program`, `solve_task` methods and all
  prompt constants directly through AST loading. Imports/host package installer
  are not loaded. `step` uses equivalent error-feedback/unchanged-program logic
  with isolated execution. The official loop allows up to ten executions.
- Reuses `data/dabstep/probe.py:ContainerPython` and its existing tested sandbox
  lifecycle. Only extension there is an optional image argument with unchanged
  default. No PD/SGLang/inference-engine logic is changed.

## Configuration

`configs/experiments/scienceagentbench_probe.json` fixes task IDs 5,6,7,8,9,20
in original order (CPU machine learning/statistics/plotting, dependencies selected
before model execution). Model TP=1, GPU0, static fraction0.80, context40960,
8192 maximum output per model call, temperature0, two concurrent agents.

Only official question, data tree/preview, input data path and expected output
path enter the model. Gold programs, rubrics, evaluator and domain knowledge
are not supplied. Container mounts only input dataset files and the generic
worker, never the repository or annotation table.

Sandbox is non-root, network disabled, read-only root and inputs, no capabilities,
no GPUs, 2 CPU cores, 4 GiB RAM, 512 MiB temporary workspace, BLAS/OMP threads=1.
Each complete generated program runs in a fresh Python subprocess. The existing
wrapper persists only for RPC. Timeout900s kills the task sandbox, including
descendants. Launcher tracks only its own model/client process groups and unique
container label, and cleans them on exit/interruption.

## Deliberate upstream differences

Full list is in the config's `official_adaptations` and copied into each run.
Most importantly: local temperature0 instead of0.2; fixed preinstalled scientific
packages instead of per-round installation; no input truncation; bounded error
text; timeout stops rather than retries; remove stale expected output before a
retry. Dependency installation time is not reported as tool execution time.
The model's completed reasoning is removed before official code extraction.
After the first probe exposed a sandbox OOM, signal-terminated programs also
stop the task and clean the container. An empty-stderr nonzero exit explicitly
reports the exit code, rather than misleadingly describing a missing output.

Execution success means zero exit code plus a newly produced expected file. It
does **not** mean the scientific answer/plot is correct. No paid visual judge or
official grader is used. Raw programs, execution output/errors and model calls
are saved, but sandbox output artifacts are temporary in this first probe.

Important for future PD integration: upstream Self-debug sends the original
task prompt, only the most recent assistant program, and the latest error on
each repair call. It does not append the full previous model-call history.
Therefore its next prompt is not generally an extension of the complete parent
KV. This probe deliberately preserves that behavior. It cannot be plugged into
the append-only reverse-KV workload as-is; a separately declared append-only
agent (e.g. a suitable official CodeAct implementation) would need identical
semantics on colocated and PD sides.

## Run and validation

```bash
docker build -t scienceagentbench-probe:local examples/pd/data/scienceagentbench
PYTHONPATH=examples/pd /homes/siqic/.venvs/dabstep_smoke/bin/python -m pytest -q \
  examples/pd/tests/test_scienceagentbench_probe.py examples/pd/tests/test_dabstep_probe.py
bash examples/pd/scripts/baseline/run_scienceagentbench_probe.sh
```

The run records actual image ID, upstream revision, annotation/source hashes,
configuration, model tokens, complete programs and execution timings. Input
files must be downloaded from official sources before launch. Code reuse does
not imply an exact reproduction of the original paper's environment or score.

Design-invariant check: no custom KV snapshots are created; criteria1–7 remain
unchanged/not exercised by this colocated-only probe. Criterion8 requires tests
of errors, stale output, timeout/cleanup, no-gold feedback and independent GO
before GPU use. This probe must not be reported as300+1200s acceptance.

## Qwen3.5-9B verified102 run

`configs/experiments/scienceagentbench_qwen35_9b_all.json` selects all102 tasks
in verified source order. It uses colocated GPU0/TP1, eight concurrent agents,
temperature0, context40960, output8192/call and static memory fraction0.80.
The Qwen3.5 hybrid model uses `page_size=1` required by this engine's
MambaRadixCache v1. `language_only=false` uses ordinary colocated loading;
SGLang's similarly named flag means encoder disaggregation, not text-only input.
The upstream ten-execution Self-debug limit is retained. There is no search
service, paid API, custom PD path, or GPU exposed to generated tool programs.

The full scientific image is Python3.10 with preinstalled CPU packages.
Tasks101/102 use the MODNet-compatible image because its pandas requirements
conflict with BioPsyKit. Each sandbox has two CPU cores,16GiB RAM and4GiB
writable workspace. Public Salem assets are downloaded during image build;
runtime networking remains disabled. Other dynamically requested downloads
are not allowed. Package availability does not guarantee scientific correctness.

Elapsed batch time starts immediately before submitting the102 tasks and ends
when all reach terminal states, including failures/timeouts. Model startup,
dataset downloads and image builds are excluded. Atomic summary/progress files
record incremental results. Output artifacts up to64MiB each are retained;
larger or invalid artifacts have explicit capture errors. A produced output
file is an execution result, not an official grader pass.

```bash
CONFIG_PATH="$PWD/examples/pd/configs/experiments/scienceagentbench_qwen35_9b_all.json" \
RUN_DIR="$PWD/examples/pd/runs-host/baseline/scienceagentbench-qwen35-9b-c8-n102-r3" \
bash examples/pd/scripts/baseline/run_scienceagentbench_probe.sh
```

Attempts r1/r2 were startup-only failures (encoder-disaggregation flag and
page-size compatibility, respectively); neither executed any task.
