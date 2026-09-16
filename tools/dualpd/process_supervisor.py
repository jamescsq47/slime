"""One foreground supervisor, one owned process session, authenticated local stop.

Stop never trusts a PID file or kills a pattern. It contacts this live supervisor
through a run-scoped abstract Unix socket. The session leader is not reaped until
its group is cleaned, preventing PID/group reuse during termination.
"""
import ctypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import time
import uuid


def atomic_record(path, payload):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(payload, out, indent=2)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def socket_name(identity):
    digest = hashlib.sha256(identity.encode()).hexdigest()[:40]
    return "dualpd-" + digest


def _signal_owned_group(child, signum):
    if child is None or child.returncode is not None:
        return
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def _exited_without_reap(child):
    return os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def supervise(command, env, directory, identity, *, stop_grace=30.0):
    """Run until exit/stop; does not daemonize. No global process enumeration."""
    if not 0 < stop_grace <= 300:
        raise ValueError("stop grace must be positive and <=300 seconds")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    address = socket_name(identity)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    child = None
    old_handlers = {}
    stopped = [False]
    deadline = [None]
    token = uuid.uuid4().hex
    record = {"identity": identity, "socket": address, "token": token,
              "boot_id": boot_id(), "supervisor_pid": os.getpid(), "started_at": time.time()}
    def request_stop(_signum=None, _frame=None):
        stopped[0] = True
        if deadline[0] is None:
            deadline[0] = time.monotonic() + stop_grace
        _signal_owned_group(child, signal.SIGTERM)
    try:
        # Binding is the process-independent single-supervisor lock, no stale
        # lock-file replacement. The kernel frees it when this supervisor exits.
        listener.bind("\0" + address)
        listener.listen(4)
        listener.setblocking(False)
        # Reap SGLang descendants after their parent exits instead of leaving
        # adopted workers behind. This process is dedicated to one component.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers[sig] = signal.signal(sig, request_stop)
        with (directory / "service.log").open("ab", buffering=0) as logfile:
            if stopped[0]:
                return 130
            child = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            if stopped[0]:
                request_stop()
            record["child_pid"] = child.pid
            # All code after Popen remains inside the same cleanup finally.
            atomic_record(directory / "process.json", record)
            while not _exited_without_reap(child):
                if stopped[0] and deadline[0] is not None and time.monotonic() >= deadline[0]:
                    _signal_owned_group(child, signal.SIGKILL)
                readable, _, _ = select.select([listener], [], [], 0.1)
                if not readable:
                    continue
                conn, _ = listener.accept()
                with conn:
                    conn.settimeout(0.2)
                    try:
                        request = json.loads(conn.recv(4096).decode())
                        valid = (request.get("identity") == identity
                                 and hmac.compare_digest(str(request.get("token", "")), token))
                        if not valid:
                            response = {"ok": False, "error": "identity/token mismatch"}
                        elif request.get("action") == "stop":
                            request_stop()
                            response = {"ok": True, "stopping": True}
                        elif request.get("action") == "status":
                            response = {"ok": True, "stopping": stopped[0], "child_pid": child.pid}
                        else:
                            response = {"ok": False, "error": "unknown action"}
                        conn.sendall(json.dumps(response).encode())
                    except (OSError, ValueError, TypeError):
                        continue
            # Keep the leader zombie held while terminating its remaining group.
            _signal_owned_group(child, signal.SIGTERM)
            _signal_owned_group(child, signal.SIGKILL)
            result = child.wait()
            return result
    finally:
        if child is not None and child.returncode is None:
            _signal_owned_group(child, signal.SIGTERM)
            until = time.monotonic() + min(stop_grace, 5.0)
            while time.monotonic() < until and not _exited_without_reap(child):
                time.sleep(0.05)
            _signal_owned_group(child, signal.SIGKILL)
            child.wait()
        # The parent is now reaped; do not signal its numeric group again.
        if child is not None:
            until = time.monotonic() + 2.0
            while time.monotonic() < until:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    time.sleep(0.02)
            record.update(stopped_at=time.time(), exit_code=child.returncode)
            atomic_record(directory / "stopped.json", record)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        listener.close()


def control(directory, identity, action):
    """Talk only to a matching live supervisor; stale files never trigger kill."""
    record = json.loads((Path(directory) / "process.json").read_text())
    if record.get("identity") != identity or record.get("boot_id") != boot_id():
        raise RuntimeError("process record belongs to a different run/host boot")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(2)
        conn.connect("\0" + record["socket"])
        conn.sendall(json.dumps({"identity": identity, "token": record["token"], "action": action}).encode())
        response = json.loads(conn.recv(4096).decode())
    if not response.get("ok"):
        raise RuntimeError("supervisor rejected command: " + repr(response))
    return response
