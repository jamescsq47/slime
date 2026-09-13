import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('mixed_matrix', Path(__file__).parents[1] / 'scripts/tools/run_mixed_matrix.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_order_and_environment_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv('SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH', '100000')
    monkeypatch.setenv('FAST_TOOL_THRESHOLD_SECONDS', '999')
    monkeypatch.setenv('PREFILL_GPUS', '7')
    assert runner.CASES == [
        ('native_mooncake', 2, 384), ('native_mooncake', 2, 512),
        ('native_mooncake', 2, 640), ('native_mooncake', 4, 512),
        ('no_reverse', 2, 384), ('no_reverse', 2, 512),
        ('no_reverse', 2, 640), ('no_reverse', 4, 512),
        ('full', 2, 384), ('full', 2, 512), ('full', 2, 640),
    ]
    for case in runner.CASES:
        env, cmd = runner.environment(case, tmp_path/'run')
        assert env['TEMPERATURE'] == '0'
        assert 'n8192' in env['SCHEDULE_FILE']
        assert env['MODEL_PATH'] == '/homes/siqic/Qwen3-8B'
        assert env['WARMUP_SECONDS'] == '300'
        assert env['MEASURE_SECONDS'] == '1200'
        assert Path(cmd[1]).is_file()
        if case[0] == 'full':
            assert env['FAST_TOOL_THRESHOLD_SECONDS'] == '1'
            assert 'SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH' not in env
            assert 'PREFILL_GPUS' not in env
        else:
            assert env['PYTHONPATH'] == ''
            assert 'pd_baseline' in env['PD_ENV_BIN']
            assert not any(k.startswith('SGLANG_AGENTIC') for k in env)
            assert env['DECODE_MEM_FRACTION_STATICS'].split()[-1] == '0.60'
            for key, value in env.items():
                if key.endswith(('_PORT', '_PORTS')):
                    assert all(1024 <= int(port) < 32768 for port in value.split())


def test_failed_or_short_run_not_accepted(tmp_path):
    p = tmp_path/'closed_loop_boundaries.json'
    p.write_text(json.dumps(dict(measurement_seconds=500,warmup_seconds=300)))
    with pytest.raises(RuntimeError,match='incomplete'):
        runner.summarize(tmp_path)
    p.write_text(json.dumps(dict(measurement_seconds=1200,warmup_seconds=300,
        state_at_measurement_start={'failures':0},state_at_measurement_end={'failures':1})))
    with pytest.raises(RuntimeError,match='failed agents'):
        runner.summarize(tmp_path)


@pytest.mark.parametrize('message', [
    'Fatal Python error: Segmentation fault', 'Segfault encountered',
    'Subprocess scheduler_0 crashed with exit code -11',
    'error while attempting to bind on address',
])
def test_native_failure_detected(message):
    assert runner.FATAL_LOG.search(message)


def test_cleanup_signal_is_not_a_serving_failure():
    term = 'Subprocess detokenizer crashed with exit code -15. Triggering SIGQUIT'
    assert runner.fatal_service_error(term)
    cutoff = 1789075080.47  # 2026-09-10 21:18:00.47 UTC
    assert runner.fatal_service_error(term, cutoff)  # No timestamp: fail closed.
    assert not runner.fatal_service_error('[2026-09-10 21:18:19] '+term, cutoff)
    assert runner.fatal_service_error('[2026-09-10 21:17:59] '+term, cutoff)
    assert runner.fatal_service_error('[2026-09-10 21:18:00] '+term, cutoff)
    assert runner.fatal_service_error('Fatal Python error: Segmentation fault', cutoff)
    disconnect = '[2026-09-10 21:18:43] Scheduler hit an exception: Traceback\nRuntimeError: NIXL_ERR_REMOTE_DISCONNECT'
    assert not runner.fatal_service_error(disconnect, cutoff)
    assert runner.fatal_service_error(disconnect)


def test_post_window_timestamp_does_not_hide_untimestamped_failure():
    cutoff = 1789075080.47
    chunk = (
        '[2026-09-10 21:18:19] harmless cleanup line\n'
        'RuntimeError: genuine later line without timestamp'
    )
    assert runner.fatal_service_error(chunk, cutoff)


def test_update_keeps_other_sections(tmp_path,monkeypatch):
    p=tmp_path/'table.md';p.write_text('# Mixed\n\nExisting baseline.\n')
    monkeypatch.setattr(runner,'TABLE',p)
    runner.update_table({});runner.update_table({'native_mooncake-2p6d-c512':{'status':'运行中'}})
    text=p.read_text()
    assert text.count(runner.BEGIN)==1
    assert 'Existing baseline.' in text
    assert '运行中' in text
