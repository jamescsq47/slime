import importlib.util
import json
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[1]/'scripts/tools/summarize_native_swe_colocated.py'
spec=importlib.util.spec_from_file_location('native_report',path)
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_interval_integral_and_reset():
    rows=[(0,{'g':0,'c':0}),(10,{'g':2,'c':40})]
    assert report.interval_value(rows,'g',0,10)==1
    assert report.interval_value(rows,'c',0,10,True)==4
    with pytest.raises(ValueError,match='reset'):
        report.interval_value([(0,{'c':10}),(10,{'c':1})],'c',0,10,True)


def test_roles_not_doubled_but_rank_timer_normalized(tmp_path):
    tasks=[dict(started_ts=0,finished_ts=100,generation_turns=2,response_tokens=3,
                metadata=dict(instance_id=str(i),swe_bench_verifier={'resolved':i%2==0})) for i in range(500)]
    (tmp_path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
    (tmp_path/'preflight.json').write_text('{"tp":2,"pd":false,"concurrency":128}')
    records=[]
    for t in (0,100):
        metrics={'sglang_realtime_tokens_total|mode=prefill_compute':200*t,
                 'sglang_realtime_tokens_total|mode=decode':100*t,
                 'sglang_forward_execution_seconds_total|category=extend':.4*t,
                 'sglang_forward_execution_seconds_total|category=decode':.6*t,
                 'sglang_full_token_usage':.5,'sglang_mamba_usage':.7,'sglang_num_running_reqs':10}
        for role in ('prefill','decode'):
            records.append(dict(ts=t,role=role,endpoint_metrics=[dict(endpoint=str(i),metrics=metrics) for i in range(4)]))
    (tmp_path/'engine_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    result=report.summarize(tmp_path)
    m=result['windows']['full']
    assert m['prefill_token_s']==800
    assert m['decode_token_s']==400
    assert m['prefill_forward_fraction']==pytest.approx(.2)
    assert m['decode_forward_fraction']==pytest.approx(.3)
    assert m['attention_kv_fraction']==.5
    assert m['running_per_tp_group']==10
    assert result['actual_prefill_tokens_per_agent']==160
    assert result['resolved']==250
    assert '50.0%' in report.render(result)
    assert '/ c128' in report.render(result)
