"""Supervise c256 then c500; never advance after failure or cancellation.

Owns only launchers it creates. Their existing cleanup owns GPU processes and
run-labelled Docker containers. All input runs are fresh, full500 evaluations.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

LAUNCHER = Path(__file__).with_name('run_qwen35_fused_27b_tp2_swe500_4p4d.sh')

FATAL_MARKERS = ('Scheduler hit an exception', 'NIXL_ERR_REMOTE_DISCONNECT',
                 'Unable to fence NIXL sender handle', 'TP P->D shard submission failed',
                 'Agentic CUDA worker failed')


class RunHealth:
    """Stop a failed evaluation, never reclaim transport-owned buffers here.

    Long tools/verifiers alone are NOT a stall. Require a previously active
    workload, no observed model progress for five minutes AND repeated HTTP500.
    Lifecycle/fence errors fail immediately, even when schedulers remain alive.
    """
    def __init__(self):
        self.last_progress = None
        self.http_errors = 0

    def observe(self, worker_data, inference_data, now):
        fatal = [marker for marker in FATAL_MARKERS
                 if any(marker in data for data in worker_data)]
        if fatal:
            return 'transport_or_scheduler_failure: ' + ', '.join(fatal)
        if any('Prefill batch' in data or 'Decode batch' in data for data in worker_data):
            self.last_progress = now
            self.http_errors = 0
        self.http_errors += inference_data.count("Error: Server error '500 Internal Server Error'")
        if (self.last_progress is not None and now - self.last_progress >= 300
                and self.http_errors >= 3):
            return 'no_forward_300s_with_repeated_http500'
        return None


@contextmanager
def owned_child(*args, **kwargs):
    process = subprocess.Popen(*args, **kwargs)
    try:
        yield process
    finally:
        # Even if progress/log/status I/O raises, keep the owning launcher
        # alive long enough to clean up its service groups and containers.
        if process.poll() is None:
            process.terminate()
            process.wait()


def validate_repeat_inputs(previous):
    """A different agent may edit shared harness files while c256 runs."""
    pd_dir = LAUNCHER.parents[2]
    record = json.loads((previous / 'preflight.json').read_text())
    for relative, expected in record['harness_sha256'].items():
        assert hashlib.sha256((pd_dir / relative).read_bytes()).hexdigest() == expected, relative
    config = pd_dir / 'configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml'
    assert config.read_bytes() == (previous / 'workload.yaml').read_bytes()
    assert LAUNCHER.read_bytes() == (previous / 'source-snapshot/launcher.sh').read_bytes()
    dataset = Path('/tmp/pd-data/swe-bench-verified/test.jsonl')
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == record['dataset_sha256']
    source = record['source']
    revision = subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()
    diff = subprocess.check_output(['git','-C',source,'diff','HEAD'],text=True)
    assert revision == record['engine_revision']
    assert hashlib.sha256(diff.encode()).hexdigest() == record['engine_diff_sha256']
    untracked = subprocess.check_output(['git','-C',source,'ls-files','--others','--exclude-standard','-z'],text=True).split('\0')
    current = {name: hashlib.sha256((Path(source)/name).read_bytes()).hexdigest()
               for name in untracked if name.startswith('python/sglang/') and name.endswith('.py')}
    assert current == record['engine_untracked_source_sha256'], 'untracked engine sources changed'


def validate_finished(run):
    with (run / 'requests.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 500
    assert len({r['metadata']['instance_id'] for r in rows}) == 500
    assert all(r['status'] in {'completed', 'failed'} for r in rows)
    registration = json.loads((run / 'host_register_prewarm.json').read_text())
    records = registration['participants']
    assert registration['status'] == 'complete' and len(records) == 8
    expected = {(role, f'{role}-{group}', rank)
                for role in ('prefill', 'decode') for group in (0, 1) for rank in (0, 1)}
    assert {(r['role'], r['engine_id'], r['tp_rank']) for r in records} == expected
    assert all(r['registered_bytes'] == (288 if r['role'] == 'prefill' else 320) * 2**30
               for r in records)
    for name in ('host.json', 'p2d-host.json'):
        final = json.loads((run / 'control-final' / name).read_text())
        assert not final['entries'], f'Unreclaimed Host entries in {name}'
    for path in (run / 'logs').glob('*.log*'):
        with path.open(errors='replace') as stream:
            assert not any(any(marker in line for marker in FATAL_MARKERS)
                           for line in stream), path
    return {'ended': len(rows), 'completed': sum(r['status'] == 'completed' for r in rows),
            'resolved': sum(bool(r.get('metadata', {}).get('swe_bench_verifier', {}).get('resolved'))
                            for r in rows), 'registration_verified': True,
            'final_host_empty': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'sequence_status.json'
    if status_path.exists():
        raise RuntimeError('Refusing to reuse a sequence root')
    cancelled = False
    child = None
    termination_requested = False
    def stop(signum, frame):
        nonlocal cancelled, termination_requested
        cancelled = True
        if child is not None and child.poll() is None and not termination_requested:
            child.terminate()
            termination_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state = {'started_at': time.time(), 'runs': []}
    def save():
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(status_path)
    save()
    for concurrency in (256, 500):
        if cancelled:
            break
        if concurrency == 500:
            try:
                validate_repeat_inputs(root / 'c256')
            except Exception as exc:
                state.update(state='inputs_changed', error=repr(exc)); save()
                return 2
        run = root / f'c{concurrency}'
        run.mkdir(exist_ok=False)
        env = os.environ.copy()
        env.update(RUN_DIR=str(run), MAX_INFLIGHT=str(concurrency), REQUESTS='500')
        offsets = {}
        health = RunHealth()
        termination_requested = False
        with (run / 'supervisor.log').open('w') as log, owned_child(
            ['bash', str(LAUNCHER)], env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        ) as child:
            # SIGTERM may arrive during Popen, before child assignment. The
            # cancellation flag is sticky: immediately stop that new child.
            if cancelled:
                stop(signal.SIGTERM, None)
            item = {'concurrency': concurrency, 'run': str(run), 'pid': child.pid,
                    'started_at': time.time(), 'state': 'starting'}
            state['runs'].append(item); save()
            while child.poll() is None:
                if cancelled:
                    stop(signal.SIGTERM, None)
                worker_data = []
                inference_data = ''
                for path in [*(run / 'logs').glob('*.log'), run / 'inference.log']:
                    if not path.exists():
                        continue
                    with path.open(errors='replace') as stream:
                        if path.stat().st_size < offsets.get(path, 0):
                            offsets[path] = 0
                        stream.seek(offsets.get(path, 0)); data = stream.read()
                        offsets[path] = stream.tell()
                    if path.name == 'inference.log':
                        inference_data = data
                    else:
                        worker_data.append(data)
                failure = health.observe(worker_data, inference_data, time.monotonic())
                if failure:
                    item['health_failure'] = failure
                    stop(signal.SIGTERM, None)
                progress = run / 'episode_progress.jsonl'
                item['ended_so_far'] = sum(1 for _ in progress.open()) if progress.exists() else 0
                item['state'] = 'stopping' if cancelled else ('running' if progress.exists() else 'starting')
                item['checked_at'] = time.time(); save()
                time.sleep(30)
            code = child.wait()
        item.update(exit_code=code, ended_at=time.time())
        if cancelled or code:
            item['state'] = 'cancelled' if cancelled else 'failed'; save()
            return 130 if cancelled else code
        try:
            item['acceptance'] = validate_finished(run)
        except Exception as exc:
            item.update(state='acceptance_failed', error=repr(exc)); save()
            return 2
        item['state'] = 'finished'; save()
        child = None
    state['state'] = 'cancelled' if cancelled else 'finished'
    save()
    return 130 if cancelled else 0


if __name__ == '__main__':
    raise SystemExit(main())
