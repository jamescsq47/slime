import json
import subprocess

import pytest

from data.dabstep.probe import ContainerPython, upstream_task_template
from data.dabstep import probe
from smolagents import Model, ChatMessage
from types import SimpleNamespace


@pytest.fixture
def sandbox():
    executor = ContainerPython('/homes/siqic/data/DABstep/data/context')
    try:
        yield executor
    finally:
        executor.cleanup()
    assert executor.proc.poll() is not None
    result = subprocess.run(['docker', 'inspect', executor.name], capture_output=True)
    assert result.returncode != 0


def test_persistent_state_data_and_final(sandbox):
    out = sandbox('import pandas as pd\nx=pd.read_csv("/data/context/payments.csv")\nprint(x.shape)')
    assert '(' in out.logs
    assert sandbox('final_answer(len(x))').is_final_answer
    assert len(sandbox.events) == 2


def test_isolation_and_errors(sandbox):
    inspection = json.loads(subprocess.check_output(['docker', 'inspect', sandbox.name]))[0]
    assert inspection['HostConfig']['NetworkMode'] == 'none'
    assert inspection['HostConfig']['ReadonlyRootfs']
    assert inspection['Config']['User'] == '65534:65534'
    assert {m['Destination'] for m in inspection['Mounts'] if m['Type'] == 'bind'} == {'/data/context', '/worker.py'}
    for code in ['open("/data/context/payments.csv","w")',
                 'open("/homes/siqic/data/DABstep/data/tasks/dev.jsonl")',
                 'import socket; socket.create_connection(("1.1.1.1",80),timeout=0.5)',
                 'raise ValueError("test error")']:
        with pytest.raises(ValueError):
            sandbox(code)
    assert sandbox('final_answer("still alive")').output == 'still alive'


def test_timeout_cleanup(sandbox):
    sandbox.timeout = 0.1
    with pytest.raises(TimeoutError):
        sandbox('while True: pass')
    assert sandbox.closed
    assert sandbox.proc.poll() is not None


def test_prompt_constant():
    template = upstream_task_template('/homes/siqic/data/DABstep-upstream/baseline/prompts.py')
    prompt = template.format(ctx_path='/data/context', question='QUESTION', guidelines='GUIDELINE')
    assert 'QUESTION' in prompt and 'GUIDELINE' in prompt


def test_reasoning_and_action_boundary():
    stops = ['Observation:', 'Calling tools:', '</code>']
    text = '<think>example </code> Observation: fake</think><code>print(1)</code>Observation: fabricated<code>bad()</code>'
    assert probe.action_text(text, stops) == '<code>print(1)'
    assert probe.action_text('<think><code>bad()</code>', stops) == ''
    assert probe.action_text('<code>print(1)</code>', stops) == '<code>print(1)'


def test_codeagent_real_executor_mock_model(monkeypatch):
    class FakeModel(Model):
        def __init__(self):
            super().__init__()
            self.calls = []
        def generate(self, messages, **kwargs):
            assert 'SECRET_GOLD' not in str(messages)
            return ChatMessage(role='assistant', content='<code>final_answer("TEST")</code>')
    monkeypatch.setattr(probe, 'RecordedModel', lambda endpoint: FakeModel())
    row = {'task_id': 'fake', 'question': 'Test', 'guidelines': 'Test', 'level': 'easy', 'answer': 'SECRET_GOLD'}
    result = probe.run_task(row, {'endpoint': 'unused', 'context': '/homes/siqic/data/DABstep/data/context',
                                'tool_timeout_seconds': 60, 'max_steps': 2}, '{question} {guidelines}')
    assert result.get('answer') == 'TEST', result
    assert result['state'] == 'success'
    assert not result['cleanup_errors']
    assert 'SECRET_GOLD' not in json.dumps(result)


def test_cleanup_failure_still_reaps(monkeypatch):
    executor = ContainerPython.__new__(ContainerPython)
    events = []
    executor.closed = False
    executor.name = 'mock-no-container'
    executor.cleanup_errors = []
    executor.proc = SimpleNamespace(poll=lambda: None, kill=lambda: events.append('kill'),
                                    wait=lambda **kw: events.append('wait'))
    executor.selector = SimpleNamespace(close=lambda: events.append('close'))
    def fail(*a, **kw):
        raise subprocess.TimeoutExpired('docker', 30)
    monkeypatch.setattr(probe.subprocess, 'run', fail)
    executor.cleanup()
    assert events == ['kill', 'wait', 'close']
    assert executor.cleanup_errors


def test_startup_failure_removes_container(monkeypatch):
    original = ContainerPython.read_response
    names = []
    def fail(self, timeout):
        original(self, timeout)
        names.append(self.name)
        raise RuntimeError('bad startup handshake')
    monkeypatch.setattr(ContainerPython, 'read_response', fail)
    with pytest.raises(RuntimeError):
        ContainerPython('/homes/siqic/data/DABstep/data/context')
    assert names
    assert subprocess.run(['docker', 'inspect', names[0]], capture_output=True).returncode != 0


def test_cancellation_cleanup(monkeypatch):
    original = ContainerPython.cleanup
    cleaned = []
    def cleanup(self):
        original(self)
        cleaned.append(self.name)
    monkeypatch.setattr(ContainerPython, 'cleanup', cleanup)
    monkeypatch.setattr(probe, 'RecordedModel', lambda endpoint: SimpleNamespace(calls=[]))
    def interrupt(**kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(probe, 'CodeAgent', interrupt)
    row = dict(task_id='cancel', question='test', guidelines='', level='easy')
    with pytest.raises(KeyboardInterrupt):
        probe.run_task(row, dict(endpoint='unused', context='/homes/siqic/data/DABstep/data/context',
                                tool_timeout_seconds=60, max_steps=2), '{question}')
    assert cleaned
    assert subprocess.run(['docker', 'inspect', cleaned[0]], capture_output=True).returncode != 0
