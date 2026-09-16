"""CPU subprocess tests only. No models, SSH, CUDA, or RDMA initialization."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

from process_supervisor import control


ROOT = Path(__file__).resolve().parent


class SupervisorTests(unittest.TestCase):
    def launch(self, directory, identity, child_code, *, fail_record=False):
        driver = (
            "import sys,os; sys.path.insert(0,sys.argv[1]); import process_supervisor as s; "
            + ("s.atomic_record=lambda *a: (_ for _ in ()).throw(RuntimeError('record failed')); " if fail_record else "")
            + "sys.exit(s.supervise([sys.executable,'-c',sys.argv[4]],os.environ,sys.argv[2],sys.argv[3],stop_grace=0.2))"
        )
        return subprocess.Popen([sys.executable, "-c", driver, str(ROOT), str(directory), identity, child_code],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def wait_record(self, directory, process):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                return json.loads((Path(directory) / "process.json").read_text())
            except FileNotFoundError:
                if process.poll() is not None:
                    self.fail("supervisor exited: " + repr(process.communicate()))
                time.sleep(0.02)
        self.fail("no supervisor record")

    def test_stop_escalates_only_owned_session(self):
        with tempfile.TemporaryDirectory() as d:
            identity = uuid.uuid4().hex
            innocent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            process = self.launch(d, identity, "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)")
            try:
                record = self.wait_record(d, process)
                self.assertTrue(control(d, identity, "status")["ok"])
                time.sleep(0.1)
                self.assertTrue(control(d, identity, "stop")["stopping"])
                process.wait(timeout=5)
                self.assertFalse(Path("/proc/" + str(record["child_pid"])).exists())
                self.assertIsNone(innocent.poll())
                with self.assertRaises(OSError):
                    control(d, identity, "stop")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                process.communicate()
                innocent.terminate()
                innocent.wait()

    def test_wrong_identity_does_not_stop(self):
        with tempfile.TemporaryDirectory() as d:
            identity = uuid.uuid4().hex
            process = self.launch(d, identity, "import time; time.sleep(30)")
            try:
                self.wait_record(d, process)
                with self.assertRaisesRegex(RuntimeError, "different"):
                    control(d, "unrelated", "stop")
                self.assertTrue(control(d, identity, "status")["ok"])
            finally:
                process.terminate()
                process.wait(timeout=5)
                process.communicate()

    def test_child_exit_cleans_descendants(self):
        with tempfile.TemporaryDirectory() as d:
            child = ("import subprocess,sys,time; from pathlib import Path; "
                     "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                     "Path(" + repr(str(Path(d) / "grandchild.pid")) + ").write_text(str(p.pid)); time.sleep(.15)")
            process = self.launch(d, uuid.uuid4().hex, child)
            process.wait(timeout=5)
            process.communicate()
            grandchild = int((Path(d) / "grandchild.pid").read_text())
            self.assertFalse(Path("/proc/" + str(grandchild)).exists())

    def test_publication_failure_does_not_orphan_child(self):
        with tempfile.TemporaryDirectory() as d:
            pidfile = Path(d) / "child.pid"
            code = "import os,time; from pathlib import Path; Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid())); time.sleep(30)"
            process = self.launch(d, uuid.uuid4().hex, code, fail_record=True)
            process.wait(timeout=5)
            process.communicate()
            if pidfile.exists():
                self.assertFalse(Path("/proc/" + pidfile.read_text()).exists())
            self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
