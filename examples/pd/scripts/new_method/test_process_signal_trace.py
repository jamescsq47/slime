"""No-GPU coverage for the optional isolated server process tracer."""
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


@pytest.mark.parametrize('command,expected,marker', [
    ('exit 37',37,'exited with 37'),
    ('kill -TERM $$',-signal.SIGTERM,'killed by SIGTERM'),
])
def test_trace_preserves_exit_status(tmp_path,command,expected,marker):
    result=subprocess.run(['setsid','strace','-ff','--seccomp-bpf','-ttt',
        '-e','trace=process,signal,prctl','-o',str(tmp_path/'process'),
        '/bin/sh','-c',command],capture_output=True,timeout=10)
    assert result.returncode==expected
    assert any(marker in p.read_text() for p in tmp_path.glob('process.*'))


def test_trace_group_cleanup_covers_children(tmp_path):
    child=subprocess.Popen(['setsid','strace','-ff','--seccomp-bpf','-ttt',
        '-e','trace=process,signal,prctl','-o',str(tmp_path/'process'),
        '/bin/sh','-c','sleep 30 & wait'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while len(list(tmp_path.glob('process.*')))<2 and time.monotonic()<deadline:
            time.sleep(.01)
        assert len(list(tmp_path.glob('process.*')))>=2
        os.killpg(child.pid,signal.SIGTERM)
        child.wait(timeout=5)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            rows=subprocess.check_output(['ps','-e','-o','pgid=,stat='],text=True).splitlines()
            live=[x for x in rows if x.split()[0]==str(child.pid) and not x.split()[1].startswith('Z')]
            if not live:break
            time.sleep(.05)
        assert not live
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=5)
