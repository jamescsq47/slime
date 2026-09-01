# Agentic PD serving experiments

This directory contains pluggable agentic serving workloads, the clean SGLang
baselines, and the request-generation KV pipeline. Retool, BrowseComp and
Terminal-Bench now live under `data/`; thin root-level compatibility modules
keep existing experiment imports working.

## Layout

```text
examples/pd/
├── configs/                 environment and fixed workload schedules
│   ├── experiments/         dataset mixtures and reproducible sampling rules
│   └── profiles/            effective serving/topology defaults for reruns
├── data/                    dataset loaders and inference-only harness plugins
│   ├── retool/
│   ├── browsecomp/
│   ├── terminal_bench/
│   └── swe_bench/
├── docs/                    design and validation notes
├── patches/                 SGLang patch kept for reproducibility
├── requirement.txt          PD runtime, transport and analysis dependencies
├── scripts/
│   ├── baseline/            unmodified pd_baseline experiments
│   ├── new_method/          modified pd request-generation KV experiments
│   │   └── internal/        service lifecycle implementation; do not run directly
│   ├── bandwidth/           isolated NVLink/PCIe/Host/Mooncake microbenchmarks
│   ├── tools/               analysis, validation and correctness utilities
│   └── common/              shared process cleanup helpers
├── tests/                   focused unit tests for the PD extensions
├── runs-host/
│   ├── baseline/            retained formal baseline results only
│   └── new-method/          retained latest-method result only
└── *.py                     shared PD runtime and compatibility modules
```

Historical sweep, smoke-test, superseded launcher and temporary visualization
scripts have been removed.  See each `scripts/*/README.md` for supported entry
points and `runs-host/README.md` for the retained result inventory.

## Prerequisites

The validated setup uses Linux, NVIDIA GPUs with peer-to-peer access, Conda,
and a CUDA 12.x-compatible driver.  The launchers assume that the repository
root is on `PYTHONPATH`; Slime does not need to be installed as a wheel.

Set reusable locations before preparing the model and datasets:

```bash
export SLIME_ROOT=/path/to/slime
export PD_DATA_ROOT=/path/to/pd-data
export PD_MODEL_ROOT=/path/to/models
mkdir -p "${PD_DATA_ROOT}" "${PD_MODEL_ROOT}"
cd "${SLIME_ROOT}"
```

If Hugging Face requires authentication, run `hf auth login` first.  Keep the
model, datasets, Hugging Face cache and experiment outputs on local storage
rather than NFS when possible.

## Model and datasets

Download the generation model:

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3-8B \
  --local-dir "${PD_MODEL_ROOT}/Qwen3-8B"
```

Download the exact Slime-ready DAPO-Math-17k JSONL used by these experiments:

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir "${PD_DATA_ROOT}/dapo-math-17k"
```

Do not substitute the current default split of
`BytedTsinghua-SIA/DAPO-Math-17k`: its upstream contents have changed and no
longer represent the fixed 17k-row JSONL used in the retained runs.

The BrowseComp workload used by the retained experiments is the FoldAgent
`bc_train.parquet`, not the test-only `Tevatron/browsecomp-plus` dataset.  Fetch
the public FoldAgent archive and convert it as follows:

```bash
python -m pip install gdown pandas pyarrow
mkdir -p "${PD_DATA_ROOT}/browsecomp/raw"
python -m gdown 1aX5xXAN5R-gLKd8A0AY-troxXJRawyAM \
  -O "${PD_DATA_ROOT}/browsecomp/raw/browsecomp_data.zip"
unzip -o "${PD_DATA_ROOT}/browsecomp/raw/browsecomp_data.zip" \
  -d "${PD_DATA_ROOT}/browsecomp/raw"

python examples/pd/scripts/tools/prepare_workloads.py browsecomp \
  --input "${PD_DATA_ROOT}/browsecomp/raw/bc_train.parquet" \
  --output "${PD_DATA_ROOT}/browsecomp/bc_train.jsonl"
```

Some versions of the archive contain an extra directory level.  In that case,
locate the file with `find "${PD_DATA_ROOT}/browsecomp/raw" -name
'bc_train.parquet'` and pass the returned path to `--input`.

BrowseComp search additionally uses:

- `Qwen/Qwen3-Embedding-8B` for query embeddings;
- `Tevatron/browsecomp-plus-corpus` for the 100k-document corpus;
- `miaolu3/browsecomp-plus` for precomputed corpus embeddings.

`search_server.py` downloads these resources on first launch.  To prefetch
them into a persistent cache:

```bash
export HF_HOME="${PD_DATA_ROOT}/huggingface-cache"
hf download Qwen/Qwen3-Embedding-8B
hf download --repo-type dataset Tevatron/browsecomp-plus-corpus
hf download --repo-type dataset miaolu3/browsecomp-plus
```

All maintained launchers accept `MODEL_PATH` and `WORKLOAD_CONFIG`. Dataset
paths, harnesses, mixture weights and sampling semantics belong in that
YAML/JSON config. For example:

```bash
export MODEL_PATH="${PD_MODEL_ROOT}/Qwen3-8B"
export PD_DATA_ROOT=/path/to/pd-data
export WORKLOAD_CONFIG=examples/pd/configs/experiments/mixed_retool_browsecomp_1to1.yaml
```

The old `MATH_DATA`, `QA_DATA`, and `MATH_RATIO` variables remain compatible
when `WORKLOAD_CONFIG` is unset. The legacy path produces the same two-stage
shuffle as before. Every run records the resolved config and exact sample
order, so a new mixture can be replayed rather than randomly regenerated.
See `data/README.md` for the plugin contract and extension instructions.

Serving parameters that change the meaning or capacity of an experiment belong
in a checked profile rather than an ad-hoc shell command.  For example, the
Qwen3-8B BrowseComp 4P:4D profile records the topology, model limits, sampling,
closed-loop duration, exact-workset admission, D admission target, and both
Host arenas:

```bash
RUN_DIR=/path/to/result \
  bash examples/pd/scripts/new_method/run_qwen3_8b_tp1_browsecomp_4p4d.sh
```

The launcher reads
`configs/profiles/browsecomp_qwen3_8b_tp1_4p4d.yaml`, prints every effective
value before starting a GPU process, and marks caller-provided values as
`[override]`.  The same effective data-plane values are saved under
`serving_runtime` in the result's `config.json`.

BrowseComp retrieval and Terminal-Bench OpenEnv are deliberately external
services. Those harnesses only connect to configured endpoints; they never
start a GPU search worker or Terminal-Bench task container themselves. Existing
launchers may still manage BrowseComp search as an explicit service lifecycle
step. SWE-bench is different: each trajectory owns an official per-instance
Docker environment, so its harness creates and cleans that isolated container.

SWE-bench uses the official mini-SWE-agent control loop, official per-instance
environments, and the Miles/Harbor verifier contract.  The launcher pins
mini-SWE-agent v2.4.6 at commit
`25941c89cfbc91eb40b3f8756348c91d9977d57e`; it reads the upstream
`config/benchmarks/swebench.yaml` directly instead of maintaining a local copy
of the prompts or state machine.  Prepare that source checkout with:

```bash
bash examples/pd/scripts/tools/prepare_miniswe_agent.sh
```

Install the verifier dependencies in every environment that launches
`inference.py` (normally `pd`; also `pd_baseline` when evaluating a baseline):

```bash
python -m pip install -r examples/pd/requirements-swe-bench.txt
```

Export SWE-bench Verified into the harness format and prefetch the corresponding
images:

```bash
python examples/pd/scripts/tools/prepare_swe_bench.py \
  --dataset princeton-nlp/SWE-bench_Verified --split test \
  --output "${PD_DATA_ROOT}/swe-bench-verified/test.jsonl"

# Replace INSTANCE with e.g. astropy__astropy-12907. Repeat or parallelize for
# every instance selected for an experiment.
INSTANCE=astropy__astropy-12907
IMAGE_ID="${INSTANCE/__/_1776_}"
docker pull "swebench/sweb.eval.x86_64.${IMAGE_ID,,}:latest"
```

Use `configs/experiments/swe_bench_verified_eval.yaml` for correctness runs.
After the model stops, it validates the official repository baseline, captures
committed/staged/unstaged/untracked changes, injects the hidden official test
script, and records `swe_bench_verifier.resolved` plus `sample.reward`. Hidden
`test_patch` contents never enter the model prompt.

The verifier has three modes. `inline` runs hidden tests and produces a score;
`capture` only preserves the complete patch; `disabled` is intended for pure
serving-throughput profiles so CPU verification time does not occupy a
closed-loop slot. `verifier_max_concurrent` limits simultaneous CPU test suites
(default 4). The sandbox backend is independent: `docker` is the local
default, while `daytona` uses the same agent and verifier code through the
optional SDK:

```bash
python -m pip install -r examples/pd/requirements-swe-bench-daytona.txt
export DAYTONA_API_KEY=...
# Set sandbox_backend: daytona in the workload YAML.
```

The serving-only Lite workload remains in
`configs/experiments/swe_bench_lite.yaml`.  The maintained Qwen3-32B TP=2
colocated evaluation is:

```bash
bash examples/pd/scripts/baseline/run_qwen3_32b_tp2_swe_bench_colocated.sh
```

It runs four collocated TP=2 replicas on eight GPUs, 500 SWE-bench Verified
tasks with at most 128 active task containers, temperature 0, top-p 1, top-k
-1, an 8,192-token per-turn generation cap, and a 40,960-token model context.
The official agent keeps its 250-step limit and 60-second shell timeout; the
local evaluation disables the API cost limit and applies a one-hour wall-clock
limit per task.  The harness preserves repository state across shell turns,
disables container networking by default, runs the official verifier inline,
and removes the task container even when the load generator cancels an in-flight
trajectory.  Results are first written below `/tmp/pd-runs` so NFS cannot
perturb serving, then copied to `runs-host/baseline` after all 500 trajectories
finish.  Each result records the complete official trajectory plus per-turn
token/model timing, Docker start/exec/close timing, patch, and verifier fields.

The local and Daytona paths use the same agent and `resolved` grading
implementation; only sandbox provisioning and file transfer differ.

### OpenEnv-style SWE-bench harness (default for new evaluations)

New SWE-bench measurements should use the separate `swe_bench_openenv`
plugin. It ports the direct episode contract from Miles PR #51 while retaining
this repository's SGLang and sandbox instrumentation:

```text
official issue + pristine task image
  -> policy emits exactly one fenced shell command per turn
  -> repository state persists across turns
  -> policy stops / reaches max_turns
  -> capture the complete tracked + untracked patch
  -> inject the hidden official verifier
  -> record resolved, trajectory, model/tool/Docker/verifier timing
```

Unlike the historical mini-SWE-agent adapter, grading is not conditional on a
special submission marker. A patch left in the durable workspace is evaluated
after any normal episode termination. The hidden verifier remains unavailable
to the model. `sandbox_backend: docker` is the local default; the existing
Daytona transport remains selectable without changing the policy loop.

Qwen3.5-27B is the maintained SWE-bench model:

```bash
hf download Qwen/Qwen3.5-27B --local-dir /homes/siqic/Qwen3.5-27B
bash examples/pd/scripts/baseline/run_qwen35_27b_tp2_swe_bench_openenv_colocated.sh
```

The launcher uses four TP=2 replicas, c128, 131,072-token serving context,
8,192 tokens per model turn, temperature 0.6, top-p 0.95, top-k 20, min-p 0,
and inline canonical verification. Qwen3.5's hybrid Mamba radix cache requires
`page_size=1`; the reusable launcher keeps `page_size=64` for older Qwen3
models unless the model-specific launcher overrides it.

Before a model run, validate one official image with both expected outcomes:

```bash
python examples/pd/scripts/tools/smoke_swe_bench_verifier.py \
  --dataset "${PD_DATA_ROOT}/swe-bench-verified/test.jsonl" \
  --instance-id astropy__astropy-12907 --patch empty
python examples/pd/scripts/tools/smoke_swe_bench_verifier.py \
  --dataset "${PD_DATA_ROOT}/swe-bench-verified/test.jsonl" \
  --instance-id astropy__astropy-12907 --patch oracle
```

## SGLang source checkouts

SGLang is maintained in a separate repository.  Keep two independent source
checkouts so that changing the development branch cannot silently change the
baseline environment:

```bash
export SGLANG_GIT_URL=git@github.com:jamescsq47/sglang.git
export SGLANG_SRC_ROOT=/path/to/sglang-src
mkdir -p "${SGLANG_SRC_ROOT}"

git clone --filter=blob:none --single-branch --branch pd_baseline \
  "${SGLANG_GIT_URL}" "${SGLANG_SRC_ROOT}/pd_baseline"
git clone --filter=blob:none --single-branch --branch pd \
  "${SGLANG_GIT_URL}" "${SGLANG_SRC_ROOT}/pd"
```

The branch-to-environment mapping is intentional:

| Conda environment | SGLang branch | Purpose | Installation mode |
|---|---|---|---|
| `pd_baseline` | `pd_baseline` | Frozen clean baseline | regular, non-editable |
| `pd` | `pd` | Stable agentic PD implementation | editable |
| `pd` during node-local development | `pd_node_a`, `pd_node_b`, ... | Changes owned by one node | editable |

To develop on another node without colliding with `pd_node_a`, create and
publish a node-specific branch from the stable `pd` branch:

```bash
cd "${SGLANG_SRC_ROOT}/pd"
git switch -c pd_node_b
git push -u origin pd_node_b
```

If a node branch already exists, clone that branch instead of `pd`, or fetch
it and run `git switch --track origin/pd_node_b` before installing.  Do not
reuse one working tree for `pd_baseline` and `pd`: an editable installation
would follow whichever branch that working tree currently has checked out.

## Baseline environment (`pd_baseline`)

Create the environment, install the frozen SGLang baseline as a normal wheel,
then install the dependencies used by this directory:

```bash
conda create -n pd_baseline -y python=3.12 gxx_linux-64 pip
conda activate pd_baseline
python -m pip install --upgrade pip setuptools wheel
python -m pip install "${SGLANG_SRC_ROOT}/pd_baseline/python"
python -m pip install -r "${SLIME_ROOT}/examples/pd/requirement.txt"
```

A non-editable baseline install copies SGLang into the Conda environment, so
later edits or branch switches in a source checkout cannot contaminate a
running baseline experiment.  The validated package versions are SGLang
`0.5.10.post1`, SGLang Router `0.3.2`, Mooncake Transfer Engine
`0.3.12.post1`, NIXL `1.3.2`, PyTorch `2.9.1`, and Python 3.12.

Verify that the installed package contains no custom agentic modules:

```bash
cd "${SLIME_ROOT}"
python examples/pd/scripts/tools/check_environments.py --expect baseline
```

The scripts default to `/homes/siqic/anaconda3/envs/pd_baseline/bin` and
`/dataset/model/qwen3/Qwen3-8B`.  On another machine, provide both overrides:

```bash
PD_ENV_BIN="$(conda run -n pd_baseline which python | xargs dirname)" \
MODEL_PATH="${PD_MODEL_ROOT}/Qwen3-8B" \
MATH_DATA="${PD_DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl" \
QA_DATA="${PD_DATA_ROOT}/browsecomp/bc_train.jsonl" \
bash examples/pd/scripts/baseline/run_four_gpu_comparison_suite.sh
```

## Modified environment (`pd`)

Create `pd` with the same common dependencies, but install the modified SGLang
checkout in editable mode.  Source changes in that checkout then take effect
without copying files into `site-packages`:

```bash
conda create -n pd -y python=3.12 gxx_linux-64 pip
conda activate pd
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${SGLANG_SRC_ROOT}/pd/python"
python -m pip install -r "${SLIME_ROOT}/examples/pd/requirement.txt"
```

`requirement.txt` deliberately excludes `sglang`: the two environments need
different source branches even though their Python package version is both
`0.5.10.post1`.  `configs/environment-pd.yml` remains a Conda convenience
file for the common package layer, but installing SGLang from the Git checkout
is the authoritative way to select baseline versus modified code.

To update the modified environment later:

```bash
cd "${SGLANG_SRC_ROOT}/pd"
git pull --ff-only
conda activate pd
python -m pip install -e ./python
python -m pip install -r "${SLIME_ROOT}/examples/pd/requirement.txt"
```

For the frozen baseline, do not pull a moving branch during an experiment.  If
an intentional baseline update is required, record the commit, reinstall it,
and rerun the environment check:

```bash
cd "${SGLANG_SRC_ROOT}/pd_baseline"
git pull --ff-only origin pd_baseline
conda activate pd_baseline
python -m pip install --upgrade --force-reinstall --no-deps ./python
```

Validate without allocating GPUs:

```bash
cd "${SLIME_ROOT}"
conda run -n pd python \
  examples/pd/scripts/tools/check_environments.py --expect modified
conda run -n pd_baseline python \
  examples/pd/scripts/tools/check_environments.py --expect baseline
```

Also record the source/commit before a formal run:

```bash
conda run -n pd python -c \
  'import importlib.metadata as m, pathlib, sglang; print(m.version("sglang"), pathlib.Path(sglang.__file__).resolve())'
git -C "${SGLANG_SRC_ROOT}/pd" rev-parse HEAD

conda run -n pd_baseline python -c \
  'import importlib.metadata as m, pathlib, sglang; print(m.version("sglang"), pathlib.Path(sglang.__file__).resolve())'
git -C "${SGLANG_SRC_ROOT}/pd_baseline" rev-parse HEAD
```

## Formal experiments

Select a config-driven workload without changing the serving topology:

```bash
PD_DATA_ROOT="${PD_DATA_ROOT}" \
WORKLOAD_CONFIG=examples/pd/configs/experiments/mixed_retool_browsecomp_1to1.yaml \
bash examples/pd/scripts/new_method/run_1p3d_case.sh
```

Before using Terminal-Bench, start its OpenEnv service manually, set
`environment_url` in `configs/experiments/terminal_bench.yaml`, and use that
config in the same way.

```bash
OPENENV_ROOT=/path/to/openenv \
bash /homes/siqic/slime/examples/pd/scripts/tools/start_terminal_bench_env.sh
```

The service launcher defaults `MAX_CONCURRENT_ENVS` to 256 through
`configs/services/terminal_bench.env`, matching the formal c256 workload.  An
explicit environment value may override it for a different concurrency sweep.

Run the four-GPU colocated/1P:3D baseline comparison:

```bash
bash examples/pd/scripts/baseline/run_four_gpu_comparison_suite.sh
```

Run the four-GPU 2P:2D baseline comparison:

```bash
bash examples/pd/scripts/baseline/run_two_p_two_d_comparison_suite.sh
```

Run the current 1P:3D request-generation KV method:

```bash
bash examples/pd/scripts/new_method/run_1p3d_case.sh
```

All formal launchers default to a 300-second warmup and 1200-second
measurement window.  They use process groups and bounded TERM-to-KILL cleanup
so a failed run does not intentionally leave model services behind.

Runtime `.log` files, local search artifacts, profiler databases and temporary
`runs/` directories are ignored by Git.  Retained result summaries and plots
under `runs-host/` remain versionable.

## Tests

```bash
cd /homes/siqic/slime
PYTHONPATH=examples/pd \
  /homes/siqic/anaconda3/envs/pd/bin/python -m pytest -q \
  -p no:cacheprovider examples/pd/tests
```

Serving-level validation and bandwidth tests are intentionally separate from
unit tests; they require idle GPUs and are documented under `scripts/tools/`
and `scripts/bandwidth/`.
