"""Four finite SWE500 evaluations using the verified 27B tools contract.

Only the explicit shell(command) prompt example differs from the 27B harness.
Run under the independent systemd user service via tools/dualpd/swe9b_matrix.sh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import run_swe_aiohttp_matrix as matrix
from swe_prompt_reference import PROMPT_VERSION, matches_reference

PD = Path(__file__).resolve().parents[2]
ROOT = PD.parents[1]
ENGINE = ROOT.parent / 'sglang'
PYTHON = '/homes/siqic/anaconda3/envs/pd_multi_node/bin/python'
WORKLOAD = PD / 'configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml'
REFERENCE = Path('/tmp/pd-persist/swe-aiohttp-matrix-20260915-r1/qwen35-27b-tp2-c64-extension/qwen35-27b-tp2-c64-extension-01-27b-colocated-c64')
DOCUMENT = PD / 'runs-host/SWEBENCH_QWEN35_9B_TP1.md'
WORKLOAD_SHA = '9796e2d014cc7cdf49addddfb9b2f3711d7033636c98ea17a45dd30857fd7348'
DATA_SHA = 'f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c'


def cases():
    original = [x for x in matrix.cases() if x['model'] == '9b']
    first = json.loads(json.dumps(original[0]))
    first.update(concurrency=500, name='9b-colocated-c500', column='Colocated c500')
    first['env']['MAX_INFLIGHT'] = '500'
    result = [first, *original]
    for case in result:
        case['document'] = str(DOCUMENT)
        case['action_protocol'] = 'openai_tools'
        case['ideal_prefix_reference'] = False
        case['env'].update(
            WORKLOAD_CONFIG=str(WORKLOAD), MODEL_REASONING_PARSER='glm45',
            PD_MODEL_HTTP_TRANSPORT='aiohttp',
            PD_HARNESS_REFERENCE_RUN=str(REFERENCE),
            PD_HARNESS_PROMPT_VERSION=PROMPT_VERSION,
            PD_DATA_ROOT='/tmp/pd-data', MODEL_PATH='/homes/siqic/Qwen3.5-9B',
            PD_FUSED_ENV_BIN=str(Path(PYTHON).parent),
            SGLANG_OVERLAY_ROOT=str(ENGINE / 'python'),
            PD_RUN_QWEN_SCRIPT=str(ENGINE / 'validation/run_pd_servers.sh'),
            PD_SCRIPT_INTERNAL_DIR=str(PD / 'scripts/new_method/internal'))
    return result


def gpu_processes_available(approved):
    """Only the explicitly approved pre-existing GPU7 process may overlap."""
    listing = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid',
                                      '--format=csv,noheader,nounits'], text=True, timeout=20)
    indices = {uuid.strip(): int(index) for index, uuid in
               (line.split(',') for line in listing.splitlines())}
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'],text=True,timeout=20)
    for line in processes.splitlines():
        uuid, pid = [s.strip() for s in line.split(',')]
        key = f'{indices[uuid]}:{pid}'
        if key not in approved or matrix.identity(int(pid)) != approved[key]:
            return False
    return True


def validate_completed(run, case):
    records = [json.loads(s) for s in (run/'requests.completed.jsonl').open() if s.strip()]
    target = {json.loads(s)['instance_id'] for s in (run/'dataset.jsonl').open() if s.strip()}
    ids = [r['metadata']['instance_id'] for r in records]
    if len(ids) != 500 or len(set(ids)) != 500 or set(ids) != target:
        raise RuntimeError('Incomplete/duplicate/different SWE500 task set')
    for record in records:
        meta = record['metadata']
        if meta.get('action_protocol') != 'openai_tools':
            raise RuntimeError('Actual task protocol is not openai_tools')
        verifier = meta.get('swe_bench_verifier') or {}
        if record.get('status') == 'completed' and verifier.get('status') not in ('completed','timeout'):
            raise RuntimeError('Completed task without a completed verifier')
        if record.get('status') != 'completed' and not (
                record.get('error') or str(meta.get('stop_reason','')).startswith('environment_error:') or
                (meta.get('stop_reason') == 'verifier_infrastructure_error' and verifier.get('status') == 'infrastructure_error')):
            raise RuntimeError('Non-completed task has no explicit error')
    config = json.loads((run/'config.json').read_text())
    for key, expected in [('temperature',0.6),('top_p',0.95),('top_k',20)]:
        if config.get(key) != expected:
            raise RuntimeError(f'Effective sampling mismatch: {key}')


def preflight():
    for executable in ['rg','docker','rsync','nvidia-smi','setsid','curl','ss','timeout']:
        if not shutil.which(executable):raise RuntimeError(f'Missing dependency: {executable}')
    if hashlib.sha256(WORKLOAD.read_bytes()).hexdigest() != WORKLOAD_SHA:
        raise RuntimeError('27B structured workload drift')
    dataset = Path('/tmp/pd-data/swe-bench-verified/test.jsonl')
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != DATA_SHA:
        raise RuntimeError('SWE500 dataset/order drift')
    if WORKLOAD.read_bytes() != (REFERENCE/'workload.yaml').read_bytes():
        raise RuntimeError('27B workload snapshot mismatch')
    checked = 0
    for old in (REFERENCE/'source-snapshot/data/swe_bench_openenv').rglob('*.py'):
        rel = old.relative_to(REFERENCE/'source-snapshot')
        if not matches_reference(old.read_bytes(), (PD/rel).read_bytes(), old.name, PROMPT_VERSION):
            raise RuntimeError(f'27B harness mismatch: {rel}')
        checked += 1
    if not checked:raise RuntimeError('Missing 27B harness reference')
    # Launchers independently repeat environment, image, dataset and port checks.
    return {'workload_sha256': WORKLOAD_SHA, 'dataset_sha256': DATA_SHA,
            'reference': str(REFERENCE), 'harness_files_verified': checked,
            'prompt_version': PROMPT_VERSION,
            'harness_sha256': hashlib.sha256((PD/'data/swe_bench_openenv/harness.py').read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--allow-gpu-process',action='append',default=[])
    args = parser.parse_args()
    approved = {}
    for key in args.allow_gpu_process:
        gpu,pid = map(int,key.split(':'))
        if gpu not in range(8) or pid <= 0:raise ValueError('Invalid approved GPU process')
        ident = matrix.identity(pid)
        if ident is None:raise RuntimeError(f'Approved process no longer alive: {key}')
        approved[key] = ident
    checks = preflight()
    selected = cases()
    matrix.PY = PYTHON
    matrix.ENGINE = ENGINE
    matrix.APPROVED_GPU_PROCESSES = approved
    if args.action == 'plan':
        print(json.dumps(dict(preflight=checks,cases=selected,approved=approved),indent=2))
        return
    root = args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    disk = os.statvfs(root)
    if disk.f_bavail * disk.f_frsize < 100 * 2**30:
        raise RuntimeError('Output filesystem has less than 100 GiB free')
    # Snapshot launcher/harness/runtime state; never modify the baseline env.
    matrix.atomic_json(root/'structured_plan.json',dict(preflight=checks,cases=selected,approved=approved))
    matrix.run_matrix(root, selected=selected, strict=True)
    if matrix.read_json(root/'sequence_status.json',{}).get('state') != 'finished':
        raise SystemExit(1)


if __name__ == '__main__':main()
