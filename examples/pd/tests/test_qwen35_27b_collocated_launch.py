"""Contract checks for the isolated 27B native-baseline launcher."""
from pathlib import Path
import importlib.util
import json
import signal
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/baseline/run_qwen35_27b_tp2_swe500_mamba_baseline.sh'


def test_shell_syntax():
    subprocess.run(['bash', '-n', str(SCRIPT)], check=True)


def test_isolated_native_environment():
    text = SCRIPT.read_text()
    assert 'PD_ENV_BIN=/homes/siqic/anaconda3/envs/pd_mamba_baseline/bin' in text
    assert 'unset SGLANG_OVERLAY_ROOT PD_P_READY_DIR' in text
    assert 'compgen -v SGLANG_AGENTIC_' in text
    assert 'compgen -v SGLANG_PD_' in text
    assert '--expect baseline' in text
    for forbidden in ('--disaggregation-mode', '--enable-hierarchical-cache', '--disaggregation-decode-enable-offload-kvcache'):
        assert forbidden not in text


def test_alignment_and_full_dataset():
    text = SCRIPT.read_text()
    for required in ("groups=('0,4' '1,5' '2,6' '3,7')", '--tp-size 2',
                     '--mamba-full-memory-ratio 0.9 --mem-fraction-static 0.80',
                     '--page-size 64 --mamba-track-interval 64',
                     '--reasoning-parser glm45 --tool-call-parser qwen3_coder',
                     'swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml',
                     '--requests 500 --warmup-requests 0', '--preserve-source-order',
                     '--temperature 0.6 --top-p 0.95 --top-k 20', 'export MIN_P=0',
                     '${MAX_INFLIGHT:-128}'):
        assert required in text
    assert '--closed-loop' not in text


def test_tp2_performance_settings_are_explicit():
    text = SCRIPT.read_text()
    assert '--enable-deterministic-inference' not in text
    assert '--disable-custom-all-reduce' not in text
    for required in ('--attention-backend triton --sampling-backend flashinfer',
                     '--numa-node 0 1 --mamba-radix-cache-strategy extra_buffer',
                     'unset SGLANG_ENABLE_DETERMINISTIC_INFERENCE NCCL_ALGO',
                     "runtime_versions=", 'mamba_resumed_chunk_{alignment,capacity}.patch'):
        assert required in text


def test_measurements_and_owned_cleanup():
    text = SCRIPT.read_text()
    for required in ('SGLANG_ENABLE_METRICS_DEVICE_TIMER=true', '--enable-metrics',
                     '--metrics-interval 2', '--log-requests-level 3',
                     'pd_track_group "${worker_pid}"', 'pd_track_group "${inference_pid}"',
                     'trap baseline_cleanup EXIT', 'label=pd.swe.run_id=${PD_SWE_RUN_ID}',
                     'pd_cleanup_all', 'Refusing reused run'):
        assert required in text


def test_monitor_progress_deduplicates_and_tolerates_partial_write(tmp_path):
    path = SCRIPT.with_name('monitor_qwen35_27b_swe500.py')
    spec = importlib.util.spec_from_file_location('native_swe_monitor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.progress(tmp_path) == {'ended': 0, 'resolved': 0}
    (tmp_path / 'episode_progress.jsonl').write_text(
        '{"instance_id":"a","reward":0}\n'
        '{"instance_id":"a","reward":1}\n'
        '{"instance_id":"b","reward":0}\n'
        '{"instance_id":'
    )
    assert module.progress(tmp_path) == {'ended': 2, 'resolved': 1}


@pytest.mark.parametrize('cancel_during_popen', [False, True])
def test_monitor_sends_one_term_even_with_repeated_signals(tmp_path, monkeypatch, cancel_during_popen):
    path = SCRIPT.with_name('monitor_qwen35_27b_swe500.py')
    spec = importlib.util.spec_from_file_location('native_swe_monitor_signals', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handlers = {}

    class Process:
        pid = 123
        returncode = None
        terms = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terms += 1
            if self.terms == 1:
                handlers[signal.SIGINT](signal.SIGINT, None)

        def wait(self):
            return self.returncode

    child = Process()

    def popen(*args, **kwargs):
        if cancel_during_popen:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return child

    def sleep(seconds):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        handlers[signal.SIGINT](signal.SIGINT, None)
        child.returncode = 130

    monkeypatch.setattr(module.signal, 'signal', lambda sig, callback: handlers.update({sig: callback}))
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(module.time, 'sleep', sleep)
    monkeypatch.setattr(sys, 'argv', [str(path), str(tmp_path)])
    module.main()
    assert child.terms == 1
    assert json.loads((tmp_path / 'monitor.json').read_text())['state'] == 'cancelled'
