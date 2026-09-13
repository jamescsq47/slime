import json
from pathlib import Path

import pytest

from data.dabstep.probe import ContainerPython
from data.scienceagentbench import probe

UPSTREAM = '/homes/siqic/data/ScienceAgentBench-upstream/agent.py'
IMAGE = 'scienceagentbench-probe:local'


@pytest.fixture
def sandbox(tmp_path):
    tmp_path.chmod(0o755)
    executor = ContainerPython(tmp_path, 20, image=IMAGE)
    try:
        yield executor
    finally:
        executor.cleanup()
    assert not executor.cleanup_errors
    assert executor.proc.poll() is not None


def test_execution_and_stale_output(sandbox):
    out = probe.execute_program(sandbox, "open('pred_results/x.txt','w').write('test')", 'pred_results/x.txt', 5)
    assert out['returncode'] == 0 and out['output_exists'] and out['output_bytes'] == 4
    out = probe.execute_program(sandbox, 'print(42)', 'pred_results/x.txt', 5)
    assert out['returncode'] == 0 and not out['output_exists']
    out = probe.execute_program(sandbox, 'raise ValueError("test")', 'pred_results/x.txt', 5)
    assert out['returncode'] != 0 and 'ValueError' in out['stderr']


def test_timeout(sandbox):
    out = probe.execute_program(sandbox, 'while True: pass', 'pred_results/x.txt', 0.1)
    assert out['timed_out'] and not out['output_exists']


def test_official_loop_with_real_executor(monkeypatch, tmp_path):
    class Model:
        def __init__(self, config):
            self.calls = []
            self.llm_engine_name = 'local'
        def respond(self, messages, **kwargs):
            assert 'SECRET_GOLD' not in str(messages)
            self.calls.append(messages)
            code = 'raise ValueError("repair me")' if len(self.calls) == 1 else "open('pred_results/x.txt','w').write('valid')"
            return '```python\n' + code + '\n```', 1, 1
    monkeypatch.setattr(probe, 'LocalEngine', Model)
    tmp_path.chmod(0o755)
    row = dict(instance_id=1, task_inst='Write output', dataset_folder_tree='|-- empty/',
               dataset_preview='', output_fname='pred_results/x.txt', gold_program_name='SECRET_GOLD')
    cfg = dict(datasets=str(tmp_path), image='unused-invalid-image', task_images={'1': IMAGE}, context_length=40960, tool_timeout_seconds=5,
               save_artifacts=True, sandbox_memory='1g', sandbox_workspace='64m')
    result = probe.run_task(row, cfg, tmp_path, probe.upstream_agent(UPSTREAM))
    assert result['state'] == 'executed_output_exists', result
    assert len(result['executions']) == 2 and len(result['model_calls']) == 2
    assert not result['cleanup_errors']
    assert 'SECRET_GOLD' not in json.dumps(result)
    assert result['executions'][1]['output_bytes'] == 5
    assert Path(result['artifact']['path']).read_text() == 'valid'
    checkpoint = json.loads((tmp_path / 'progress-1.json').read_text())
    assert checkpoint['state'] == 'executed_output_exists'
    assert len(checkpoint['executions']) == 2
    assert result['executor_image'] == IMAGE


def test_upstream_extraction_ignores_reasoning(tmp_path):
    base = probe.upstream_agent(UPSTREAM)
    agent = base()
    path = tmp_path / 'program.py'
    text = probe.action_text('<think>```python\nbad()\n```</think>```python\nprint(42)\n```', [])
    assert agent.write_program(text, str(path)) is False
    assert path.read_text() == 'print(42)'
    assert agent.write_program(text, str(path)) is True


@pytest.mark.parametrize('signal_exit', [True, False])
def test_process_failure_semantics(monkeypatch, tmp_path, signal_exit):
    class Model:
        def __init__(self, config):
            self.calls = []
            self.llm_engine_name = 'local'
        def respond(self, messages, **kwargs):
            self.calls.append(messages)
            if len(self.calls) == 1:
                code = 'import os, signal; os.kill(os.getpid(), signal.SIGKILL)' if signal_exit else 'import sys; sys.exit(3)'
            else:
                assert 'exited with code 3' in messages[-1]['content']
                code = "open('pred_results/x.txt','w').write('valid')"
            return '```python\n' + code + '\n```', 1, 1
    monkeypatch.setattr(probe, 'LocalEngine', Model)
    tmp_path.chmod(0o755)
    row = dict(instance_id=1, task_inst='Write output', dataset_folder_tree='|-- empty/',
               dataset_preview='', output_fname='pred_results/x.txt')
    cfg = dict(datasets=str(tmp_path), image=IMAGE, context_length=40960, tool_timeout_seconds=5)
    result = probe.run_task(row, cfg, tmp_path, probe.upstream_agent(UPSTREAM))
    assert not result['cleanup_errors']
    if signal_exit:
        assert result['state'] == 'program_signal' and result['signal'] == 9
        assert len(result['model_calls']) == 1
        assert result['executions'][0]['returncode'] == -9
    else:
        assert result['state'] == 'executed_output_exists'
        assert len(result['model_calls']) == 2


def test_artifact_rejects_paths_and_symlinks(sandbox, tmp_path):
    with pytest.raises(ValueError):
        probe.save_artifact(sandbox, '../escape', tmp_path, 1)
    sandbox("import os; os.makedirs('pred_results',exist_ok=True); os.symlink('/etc/passwd','pred_results/link')")
    with pytest.raises(ValueError, match='symlink'):
        probe.save_artifact(sandbox, 'pred_results/link', tmp_path, 1)
    assert not (tmp_path / 'artifacts/1/link').exists()
    sandbox("os.mkfifo('pred_results/fifo')")
    with pytest.raises(ValueError, match='regular'):
        probe.save_artifact(sandbox, 'pred_results/fifo', tmp_path, 1)
    assert not (tmp_path / 'artifacts/1/fifo').exists()
