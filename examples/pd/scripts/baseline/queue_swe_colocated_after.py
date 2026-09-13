"""Wait for one finite SWE run and its cleanup, then launch the next concurrency.

Does not cancel the predecessor or change harness/engine behavior. Failed or
partial predecessor runs and changed Python sources fail closed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def validate_previous(previous, pd_dir):
    summary = json.loads((previous / 'summary.json').read_text())
    assert summary['requests'] == 500, 'Predecessor did not produce 500 results'
    rows = [json.loads(s) for s in (previous / 'requests.jsonl').open()]
    assert len(rows) == len({x['sample_index'] for x in rows}) == 500
    assert all(x['status'] in {'completed', 'failed'} for x in rows)
    for old in (previous / 'source-snapshot').rglob('*.py'):
        current = pd_dir / old.relative_to(previous / 'source-snapshot')
        assert current.exists() and hashlib.sha256(current.read_bytes()).digest() == hashlib.sha256(old.read_bytes()).digest(), f'Source changed: {current}'
    for saved, current in [('launcher.sh', 'scripts/baseline/run_qwen35_9b_tp1_swe_verified_500_colocated.sh'),
                           ('scripts/common/runtime.sh', 'scripts/common/runtime.sh')]:
        assert (previous / 'source-snapshot' / saved).read_bytes() == (pd_dir / current).read_bytes(), f'Shell source changed: {current}'
    assert (previous / 'workload.yaml').read_bytes() == (pd_dir / 'configs/experiments/swe_bench_verified_miles_pr51_8k_t64.yaml').read_bytes()
    assert (previous / 'dataset.jsonl').read_bytes() == Path('/tmp/pd-data/swe-bench-verified/test.jsonl').read_bytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-pid', type=int, required=True)
    parser.add_argument('--previous-run', type=Path, required=True)
    parser.add_argument('--next-run', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    args = parser.parse_args()
    assert 1 <= args.concurrency <= 500
    pd_dir = Path(__file__).resolve().parents[2]
    assert os.uname().nodename == 'a10.mit.edu', 'This queue is for the a10 local run'
    args.next_run.mkdir(parents=True, exist_ok=True)
    with (args.next_run / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (args.next_run / 'environment.json').exists(), 'Next run already started'
        identity = process_identity(args.previous_pid)
        if identity is not None:
            cmdline = Path(f'/proc/{args.previous_pid}/cmdline').read_bytes()
            assert b'run_qwen35_9b_tp1_swe_verified_500_colocated.sh' in cmdline
        print(f'Waiting for launcher PID {args.previous_pid} and its cleanup', flush=True)
        while identity is not None and process_identity(args.previous_pid) == identity:
            time.sleep(30)
        validate_previous(args.previous_run, pd_dir)
        free = os.statvfs(args.next_run)
        assert free.f_bavail * free.f_frsize >= 150 * 1024**3, 'Less than 150 GiB free on result/Docker disk'
        env = dict(os.environ)
        env.update(MAX_INFLIGHT=str(args.concurrency), RUN_DIR=str(args.next_run),
                   RESULTS_DIR=str(pd_dir / 'runs-host/baseline' / args.next_run.name),
                   PD_SWE_RUN_ID=args.next_run.name, PD_IDLE_GPU_MAX_MIB='1536',
                   WORKLOAD_CONFIG=str(args.previous_run / 'workload.yaml'),
                   MODEL_REASONING_PARSER='qwen3')
        print(f'Predecessor has 500 terminal results. Launching c{args.concurrency}', flush=True)
        with (args.next_run / 'supervisor.log').open('x') as log:
            # Keep the queue lock open through exec and until launcher exits.
            os.set_inheritable(lock.fileno(), True)
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            os.chdir(pd_dir.parents[1])
            launcher = pd_dir / 'scripts/baseline/run_qwen35_9b_tp1_swe_verified_500_colocated.sh'
            os.execvpe('bash', ['bash', str(launcher)], env)


if __name__ == '__main__':
    main()
