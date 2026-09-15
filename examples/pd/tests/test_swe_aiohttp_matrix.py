import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

TOOLS=Path(__file__).resolve().parents[1]/'scripts/tools'
sys.path.insert(0,str(TOOLS))
import run_swe_aiohttp_matrix as queue
import swe_matrix_report as report


def test_case_order_and_http():
    cs=queue.cases()
    assert [(c['model'],c['mode'],c['concurrency']) for c in cs]==[
        ('9b','colocated',256),('9b','pd',256),('9b','pd',500),
        ('27b','colocated',128),('27b','colocated',256),('27b','pd',128),('27b','pd',256)]
    assert all(c['env']['PD_MODEL_HTTP_TRANSPORT']=='aiohttp' for c in cs)
    assert cs[1]['column']=='新方法2P:6D c256'
    assert cs[-1]['column']=='新方法4P:4D c256'
    assert cs[1]['env']['PD_FUSED_CONGESTION_RECOMPUTE']=='false'


def test_document_update_preserves_other_cells(tmp_path):
    for c in queue.cases():
        p=tmp_path/Path(c['document']).name;p.write_text(Path(c['document']).read_text())
        before=p.read_text();case=dict(c,document=str(p));item={'state':'starting','run':'/tmp/test'}
        report.update_document(case,item)
        after=p.read_text()
        before_lines=[l for l in before.splitlines() if l.startswith('|')]
        after_lines=[l for l in after.splitlines() if l.startswith('|')]
        assert len(before_lines)==len(after_lines)
        assert '启动中' in after
        column=None;section=0
        for a,b in zip(before_lines,after_lines):
            aa=[v.strip() for v in a.strip('|').split('|')];bb=[v.strip() for v in b.strip('|').split('|')]
            if c['column'] in aa[1:]:column=aa.index(c['column'])
            if aa[0]==c['column']:assert aa[:-1]==bb[:-1]
            elif column is not None and len(aa)>column:assert aa[:column]+aa[column+1:]==bb[:column]+bb[column+1:]
            else:assert aa==bb


def test_tp_timers_normalize_but_tokens_do_not(tmp_path):
    metrics={'sglang_forward_execution_seconds_total|category=extend':0,
             'sglang_forward_execution_seconds_total|category=decode':0,
             'sglang_realtime_tokens_total|mode=prefill_compute':0,
             'sglang_realtime_tokens_total|mode=decode':0,
             'sglang_kv_used_tokens':20,'sglang_kv_evictable_tokens':30,'sglang_kv_available_tokens':50}
    records=[]
    for ts in [0,1600]:
        m=dict(metrics)
        m['sglang_forward_execution_seconds_total|category=extend']=ts*.4
        m['sglang_forward_execution_seconds_total|category=decode']=ts*1.2
        m['sglang_realtime_tokens_total|mode=prefill_compute']=ts*100
        m['sglang_realtime_tokens_total|mode=decode']=ts*200
        for role in ['prefill','decode']:
            records.append(dict(ts=ts,role=role,endpoint_metrics=[dict(endpoint=f'same{i}',metrics=m) for i in range(4)]))
    (tmp_path/'engine_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    _,_,ws=report.summarize_windows(tmp_path,[dict(started_ts=0,finished_ts=1600,status='completed')],dict(tp=2,mode='colocated'))
    m=ws['full']['roles']['decode']
    assert m['p_forward']==.2 and m['d_forward']==.6
    assert m['d_tps']==800 and m['groups']==4
    assert m['kv_used']==.2 and m['kv_resident']==.5


def test_pid_marker_and_pidfd_do_not_touch_unowned():
    marker='matrix-cpu-test-'+str(os.getpid())
    own=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=dict(os.environ,PD_MATRIX_CASE_ID=marker))
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        time.sleep(.1);owned=queue.owned_processes(marker)
        assert own.pid in owned and other.pid not in owned
        queue.signal_owned(own.pid,'wrong-start-time',signal.SIGTERM)
        assert own.poll() is None
        queue.signal_owned(own.pid,owned[own.pid],signal.SIGTERM);own.wait(timeout=5)
        assert other.poll() is None
    finally:
        for p in [own,other]:
            if p.poll() is None:p.terminate()
            p.wait()


def test_failed_case_advances_to_next(tmp_path,monkeypatch):
    launcher=tmp_path/'fail.sh';launcher.write_text('exit 9\n')
    cs=[dict(c,launcher=str(launcher)) for c in queue.cases()[:2]]
    monkeypatch.setattr(queue,'cases',lambda:cs)
    monkeypatch.setattr(queue,'source_fingerprint',lambda:{})
    monkeypatch.setattr(queue,'resources_available',lambda:True)
    monkeypatch.setattr(queue,'clean_owned',lambda *a:None)
    monkeypatch.setattr(queue,'update_report',lambda *a:None)
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *a,**k:'')
    original_sleep=time.sleep
    monkeypatch.setattr(queue.time,'sleep',lambda n:original_sleep(.01))
    root=tmp_path/'queue';queue.run_matrix(root)
    state=json.loads((root/'sequence_status.json').read_text())
    assert len(state['cases'])==2
    assert all(c['state']=='failed' and c['exit_code']==9 for c in state['cases'])
    assert state['state']=='finished'
