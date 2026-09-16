"""Offline script tests: mock timeout so no hardware probes are executed."""

import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("collect_hardware_topology.sh")


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dualpd-inventory-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bindir = self.root / "bin"
        bindir.mkdir()
        # Do not invoke the probe: inject a timeout for NVLink, success elsewhere.
        scripts = {
            "timeout": "#!/bin/bash\nshift 3\n"
            'if [[ "$1" == nvidia-smi && "${2:-}" == nvlink ]]; then exit 124; fi\n'
            "printf 'MOCK: %s\\n' \"$*\"\n",
            "nvidia-smi": "#!/bin/bash\necho 'probe unexpectedly executed' >&2\nexit 99\n",
            "ip": "#!/bin/bash\nexit 99\n",
        }
        for name, content in scripts.items():
            path = bindir / name
            path.write_text(content)
            path.chmod(0o700)
        self.env = dict(os.environ, PATH=f"{bindir}:/usr/bin:/bin")

    def run_script(self, *args):
        return subprocess.run(
            ["/bin/bash", str(SCRIPT), *map(str, args)], env=self.env,
            text=True, capture_output=True, timeout=30,
        )

    def test_help(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--peer", result.stdout)

    def test_rejects_bad_timeout_before_creating_output(self):
        output = self.root / "invalid"
        result = self.run_script("--output", output, "--timeout", "0")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())

    def test_reports_missing_timeout_and_packages_without_failing_run(self):
        output = self.root / "report with spaces"
        result = self.run_script(
            "--output", output, "--peer", "10.0.0.2",
            "--python", "dualpd-deliberately-absent-python",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        status = (output / "status.tsv").read_text()
        self.assertIn("gpu_nvlink\t124\tTIMEOUT", status)
        self.assertIn("python_packages\t127\tMISSING", status)
        self.assertIn("gpu_summary\t0\tOK", status)
        self.assertIn("10.0.0.2", (output / "peer_0_route.command.txt").read_text())
        self.assertIn("not measured bandwidth", (output / "SUMMARY.md").read_text())
        with tarfile.open(str(output) + ".tar.gz") as archive:
            self.assertIn(output.name + "/SUMMARY.md", archive.getnames())

    def test_existing_directory_is_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("keep")
        result = self.run_script("--output", output)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertFalse((output / "status.tsv").exists())

    def test_peer_requires_numeric_address(self):
        result = self.run_script("--peer", "bad;command")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
