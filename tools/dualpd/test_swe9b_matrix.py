import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'examples/pd/scripts/tools'))
import swe9b_structured_matrix as queue
from swe_prompt_reference import EXAMPLE, PROMPT_VERSION, matches_reference
from swe_matrix_report import value_for, update_document


def test_four_cases_and_protocol():
    cases = queue.cases()
    assert [(x['mode'], x['concurrency']) for x in cases] == [
        ('colocated',500),('colocated',256),('pd',256),('pd',500)]
    for case in cases:
        env=case['env']
        assert case['action_protocol']=='openai_tools'
        assert not case['ideal_prefix_reference']
        assert env['MODEL_REASONING_PARSER']=='glm45'
        assert env['PD_HARNESS_PROMPT_VERSION']==PROMPT_VERSION
        assert env['PD_SCRIPT_INTERNAL_DIR'].startswith(str(ROOT))
        assert env['SGLANG_OVERLAY_ROOT']==str(ROOT.parent/'sglang/python')
        if case['mode']=='pd':
            assert env['PD_FUSED_CONGESTION_RECOMPUTE']=='false'
            assert env['PD_FUSED_FAST_TOOL_THRESHOLD']=='2'
            assert env['PD_FUSED_P_H2D_MAX_INFLIGHT']=='4'


def test_only_prompt_example_differs_from_reference():
    old=(queue.REFERENCE/'source-snapshot/data/swe_bench_openenv/harness.py').read_bytes()
    current=(queue.PD/'data/swe_bench_openenv/harness.py').read_bytes()
    assert matches_reference(old,current,'harness.py',PROMPT_VERSION)
    assert not matches_reference(old,current,'harness.py')
    assert not matches_reference(old,current+b'\n# other change','harness.py',PROMPT_VERSION)
    assert not matches_reference(old,current+EXAMPLE,'harness.py',PROMPT_VERSION)
    assert not matches_reference(old,current,'harness.py','unknown')
    assert matches_reference(b'unchanged',b'unchanged','__init__.py',PROMPT_VERSION)
    import ast
    def assignments(source):
        tree=ast.parse(source)
        return {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
                if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                and n.targets[0].id in ('_SYSTEM_PROMPT','_TOOL_SYSTEM_PROMPT')}
    before,after=assignments(old),assignments(current)
    assert after['_SYSTEM_PROMPT']==before['_SYSTEM_PROMPT']
    assert after['_TOOL_SYSTEM_PROMPT']==before['_TOOL_SYSTEM_PROMPT']+EXAMPLE.decode()


def fixture_run(tmp_path):
    records=[dict(status='completed',metadata=dict(instance_id=f'task-{i}',
             action_protocol='openai_tools',swe_bench_verifier=dict(status='completed',resolved=False)))
             for i in range(500)]
    (tmp_path/'dataset.jsonl').write_text(''.join(json.dumps({'instance_id':f'task-{i}'})+'\n' for i in range(500)))
    (tmp_path/'config.json').write_text(json.dumps(dict(temperature=.6,top_p=.95,top_k=20)))
    return records


@pytest.mark.parametrize('fault',['none','environment_failure','verifier_timeout','verifier_infrastructure',
                                  'duplicate','missing','wrong_protocol','verifier','sampling'])
def test_completed_validation(tmp_path,fault):
    records=fixture_run(tmp_path)
    if fault=='duplicate':records[-1]=records[0]
    if fault=='missing':records.pop()
    if fault=='wrong_protocol':records[0]['metadata']['action_protocol']='fenced_shell'
    if fault=='verifier':records[0]['metadata']['swe_bench_verifier']['status']='running'
    if fault=='sampling':(tmp_path/'config.json').write_text('{}')
    if fault=='environment_failure':
        records[0]['status']='failed'
        records[0]['metadata']['stop_reason']='environment_error:RuntimeError'
        records[0]['metadata']['swe_bench_verifier']=None
    if fault=='verifier_timeout':records[0]['metadata']['swe_bench_verifier']['status']='timeout'
    if fault=='verifier_infrastructure':
        records[0]['status']='failed'
        records[0]['metadata']['stop_reason']='verifier_infrastructure_error'
        records[0]['metadata']['swe_bench_verifier']['status']='infrastructure_error'
    (tmp_path/'requests.completed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    if fault in ('none','environment_failure','verifier_timeout','verifier_infrastructure'):queue.validate_completed(tmp_path,queue.cases()[0])
    else:
        with pytest.raises(RuntimeError):queue.validate_completed(tmp_path,queue.cases()[0])


def test_gpu_identity(monkeypatch):
    def query(cmd,**kwargs):
        return '7, gpu-seven\n0, gpu-zero\n' if '--query-gpu=index,uuid' in cmd else 'gpu-seven, 42\n'
    monkeypatch.setattr(queue.subprocess,'check_output',query)
    monkeypatch.setattr(queue.matrix,'identity',lambda pid:'start1')
    assert queue.gpu_processes_available({'7:42':'start1'})
    assert not queue.gpu_processes_available({'7:42':'start2'})
    assert not queue.gpu_processes_available({'0:42':'start1'})
    assert not queue.gpu_processes_available({})


def test_nonforward_is_not_decode_forward():
    r={'case':{'mode':'colocated'},'begin':0,'end':100,'raw':{'totals':{'actual':0},'mismatches':[], 'unmatched':[]},
       'windows':{'full':{'seconds':100,'roles':{'prefill':{'p_forward':.2},'decode':{'d_forward':.7}}}}}
    assert value_for(2,'非Forward时间占比 / 卡',r)=='10.00%'


def test_report_only_changes_selected_column(tmp_path):
    p=tmp_path/'report.md'
    p.write_text('## 2. 全程结果\n\n| 指标 | Colocated c500 | Colocated c256 |\n|---|---|---|\n| Decode吞吐 | — | 123 |\n')
    case=dict(queue.cases()[0],document=str(p))
    update_document(case,{'state':'starting','run':'test'})
    assert '| Decode吞吐 | 启动中 | 123 |' in p.read_text()


def test_service_is_independent_and_scoped():
    source=(ROOT/'tools/dualpd/swe9b_matrix.sh').read_text()
    assert 'systemd-run --user' in source
    assert 'KillMode=mixed' in source and 'TimeoutStopSec=900' in source
    assert '--setenv="PATH=${PATH}"' in source
    assert '--allow-gpu-process 7:1868643' in source
    assert 'pkill' not in source and 'gpu-reset' not in source
