"""Own one native-baseline launch and persist health/progress every 30 seconds."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

FATAL = ('Scheduler hit an exception', 'CUDA error:', 'CUDA out of memory',
         'Fatal Python error: Segmentation fault', 'Agentic CUDA worker failed')


def progress(run):
    path = run / 'episode_progress.jsonl'
    rows = {}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A writer may be appending the final line.
                rows[row['instance_id']] = row
    return dict(ended=len(rows), resolved=sum(float(r.get('reward') or 0) > 0 for r in rows.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--launcher', type=Path,
                        default=Path(__file__).with_name('run_qwen35_27b_tp2_swe500_mamba_baseline.sh'))
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    status_path = run / 'monitor.json'
    if status_path.exists() or (run / 'environment.json').exists():
        raise RuntimeError('Refusing an already-started run')
    launcher = args.launcher.resolve()
    assert launcher.is_file(), launcher
    state = dict(state='starting', started_at=time.time(), run=str(run))
    child = None
    cancelled = False
    termination_requested = False

    def stop(signum, frame):
        nonlocal cancelled, termination_requested
        cancelled = True
        if child is not None and child.poll() is None and not termination_requested:
            termination_requested = True
            child.terminate()  # The launcher owns cleanup of its service groups.

    def save():
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(status_path)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    offsets = {}
    save()
    env = dict(os.environ, RUN_DIR=str(run))
    with (run / 'supervisor.log').open('x') as log:
        try:
            child = subprocess.Popen(['bash', str(launcher)], env=env,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state['launcher_pid'] = child.pid
            if cancelled:
                stop(None, None)
            while child.poll() is None:
                for path in (run / 'logs').glob('*.log'):
                    with path.open(errors='replace') as stream:
                        stream.seek(offsets.get(path, 0))
                        data = stream.read()
                        offsets[path] = stream.tell()
                    if any(marker in data for marker in FATAL):
                        state['failure'] = f'Fatal engine error in {path.name}'
                        if not cancelled:
                            stop(None, None)
                state.update(progress(run), checked_at=time.time(),
                             state='stopping' if cancelled else 'running')
                save()
                time.sleep(30)
            state.update(progress(run), exit_code=child.returncode, ended_at=time.time())
            state['state'] = ('cancelled' if cancelled else
                              'finished' if child.returncode == 0 and state['ended'] == 500
                              else 'failed')
            save()
        finally:
            if child is not None and child.poll() is None:
                stop(None, None)
                child.wait()


if __name__ == '__main__':
    main()
