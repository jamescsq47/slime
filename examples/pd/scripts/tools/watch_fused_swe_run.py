"""Stop only this run's supervisor on a fatal scheduler error; retain all logs.

Read-only health polling never changes serving configuration. The existing
supervisor performs process-group/Docker cleanup, not this watcher.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def start_identity(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("pid", type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    identity = start_identity(args.pid)
    env = Path(f"/proc/{args.pid}/environ").read_bytes().split(b"\0")
    assert f"RUN_DIR={run}".encode() in env, "supervisor run mismatch"
    cmd = Path(f"/proc/{args.pid}/cmdline").read_bytes()
    assert b"run_qwen35_fused_swe500_2p6d.sh" in cmd, "not fused SWE supervisor"
    pidfd = os.pidfd_open(args.pid)
    assert start_identity(args.pid) == identity, "supervisor changed during verification"
    offsets = {}
    while True:
        try:
            if start_identity(args.pid) != identity:
                return
        except FileNotFoundError:
            return
        fatal = []
        for role in ("prefill", "decode"):
            for path in (run / "logs").glob(f"{role}-*.log"):
                with path.open(errors="replace") as stream:
                    stream.seek(offsets.get(path, 0))
                    data = stream.read()
                    offsets[path] = stream.tell()
                if "Scheduler hit an exception" in data:
                    fatal.append(path.name)
        print(json.dumps(dict(time=time.time(), supervisor=args.pid,
                              fatal_scheduler_logs=fatal)), flush=True)
        if fatal:
            # pidfd targets this exact process, even if the numeric PID is
            # reused. Never signal GPU PIDs or arbitrary containers here.
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
