"""CPU-only sequence acceptance and launch-default regression tests."""
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('sequence27b', HERE/'run_qwen35_27b_tp2_swe500_sequence.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def finished(tmp_path):
    rows = [dict(status='completed', metadata={'instance_id': str(i),
            'swe_bench_verifier': {'resolved': i < 200}}) for i in range(500)]
    (tmp_path/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    parts = [dict(role=role, engine_id=f'{role}-{group}', tp_rank=rank,
                  registered_bytes=(288 if role=='prefill' else 320)*2**30)
             for role in ('prefill','decode') for group in (0,1) for rank in (0,1)]
    (tmp_path/'host_register_prewarm.json').write_text(json.dumps(dict(status='complete',participants=parts)))
    (tmp_path/'control-final').mkdir()
    for name in ('host.json','p2d-host.json'):
        (tmp_path/'control-final'/name).write_text('{"entries":{}}')
    (tmp_path/'logs').mkdir()
    return tmp_path


def test_full500_acceptance(tmp_path):
    result=module.validate_finished(finished(tmp_path))
    assert result['ended']==500 and result['resolved']==200


@pytest.mark.parametrize('bad', ['missing_rank','duplicate_rank','wrong_capacity','not500','host_residue','fatal'])
def test_fail_closed(tmp_path,bad):
    run=finished(tmp_path)
    if bad in ('missing_rank','duplicate_rank','wrong_capacity'):
        f=run/'host_register_prewarm.json';d=json.loads(f.read_text())
        if bad=='missing_rank':d['participants'].pop()
        elif bad=='duplicate_rank':d['participants'][-1]=d['participants'][-2]
        else:d['participants'][0]['registered_bytes']=1
        f.write_text(json.dumps(d))
    elif bad=='not500':(run/'requests.jsonl').write_text('')
    elif bad=='host_residue':(run/'control-final/host.json').write_text('{"entries":{"left":{}}}')
    else:(run/'logs/prefill-0.log').write_text('Scheduler hit an exception')
    with pytest.raises(AssertionError):module.validate_finished(run)


def test_launcher_contract():
    s=module.LAUNCHER.read_text()
    for text in ["PREFILL_GPU_GROUPS='0,4;1,5'", "DECODE_GPU_GROUPS='2,6;3,7'",
                 'PREFILL_TP_SIZE=2 DECODE_TP_SIZE=2', 'MODEL_REASONING_PARSER=glm45',
                 'MAMBA_FULL_MEMORY_RATIO=0.9', 'PD_PAGE_SIZE=64 MAMBA_TRACK_INTERVAL=64',
                 'SGLANG_AGENTIC_KV_P_H2D_DECOUPLED=false',
                 'SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH=false',
                 'SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA=1',
                 'SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER=1',
                 'SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD=1',
                 'SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT=1',
                 'SGLANG_AGENTIC_KV_EARLY_CLAIM=true',
                 'SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE=false',
                 'host_physical_arena_gib=640',
                 'swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml']:
        assert text in s,text
    assert '--enable-int8-mamba-checkpoint' not in s and '--speculative-algorithm' not in s


@pytest.mark.parametrize('mode', ['success','first_fails','cancelled'])
def test_sequence_does_not_advance_after_failure_or_cancel(tmp_path,monkeypatch,mode):
    calls=[];handlers={}
    class Child:
        pid=123
        def __init__(self,*args,**kwargs):calls.append(kwargs['env']['MAX_INFLIGHT'])
        def poll(self):return 0
        def wait(self):
            if mode=='cancelled':handlers[module.signal.SIGTERM](15,None)
            return 2 if mode=='first_fails' else 0
        def terminate(self):raise AssertionError('finished child must not be signalled')
    monkeypatch.setattr(module.subprocess,'Popen',Child)
    monkeypatch.setattr(module.signal,'signal',lambda k,v:handlers.update({k:v}))
    monkeypatch.setattr(module,'validate_finished',lambda run:dict(ended=500))
    monkeypatch.setattr(module,'validate_repeat_inputs',lambda run:None)
    monkeypatch.setattr('sys.argv',['runner',str(tmp_path)])
    code=module.main()
    assert calls==(['256','500'] if mode=='success' else ['256'])
    assert code==({'success':0,'first_fails':2,'cancelled':130}[mode])


@pytest.mark.parametrize('cancel_launch', [1, 2])
def test_cancel_during_popen_assignment_stops_new_child(tmp_path,monkeypatch,cancel_launch):
    calls=[];handlers={};terminated=[]
    class Child:
        pid=123
        def __init__(self,*args,**kwargs):
            calls.append(kwargs['env']['MAX_INFLIGHT']);self.number=len(calls)
            self.running=self.number==cancel_launch
            if self.running:handlers[module.signal.SIGTERM](15,None)
        def poll(self):return None if self.running else 0
        def wait(self):assert not self.running;return 0
        def terminate(self):self.running=False;terminated.append(self.number)
    monkeypatch.setattr(module.subprocess,'Popen',Child)
    monkeypatch.setattr(module.signal,'signal',lambda k,v:handlers.update({k:v}))
    monkeypatch.setattr(module,'validate_finished',lambda run:dict(ended=500))
    monkeypatch.setattr(module,'validate_repeat_inputs',lambda run:None)
    monkeypatch.setattr('sys.argv',['runner',str(tmp_path)])
    assert module.main()==130
    assert len(calls)==cancel_launch and terminated==[cancel_launch]


def test_monitor_io_exception_cleans_owned_child(monkeypatch):
    events=[]
    class Child:
        def __init__(self,*a,**kw):pass
        def poll(self):return None
        def terminate(self):events.append('terminate')
        def wait(self):events.append('wait');return 0
    monkeypatch.setattr(module.subprocess,'Popen',Child)
    with pytest.raises(OSError):
        with module.owned_child(['dummy']):raise OSError('status write failed')
    assert events==['terminate','wait']


@pytest.mark.parametrize('marker',module.FATAL_MARKERS)
def test_transport_error_stops_without_scheduler_crash(marker):
    assert module.RunHealth().observe([marker], '', 10)


def test_stall_requires_model_progress_then_http_failures():
    health=module.RunHealth()
    errors="Error: Server error '500 Internal Server Error'\n"*3
    assert health.observe([], errors, 0) is None  # startup, not yet serving
    assert health.observe(['Prefill batch, tokens=123'], '', 10) is None
    assert health.observe([], '', 999) is None  # tools/verifier may be slow
    assert health.observe([], errors, 1000)=='no_forward_300s_with_repeated_http500'


def test_ongoing_forward_clears_stall_errors():
    health=module.RunHealth()
    errors="Error: Server error '500 Internal Server Error'\n"*3
    health.observe(['Decode batch'], errors, 0)
    assert health.observe(['Decode batch'], '', 299) is None
    assert health.observe([], '', 600) is None
    assert health.observe([], errors, 601)
