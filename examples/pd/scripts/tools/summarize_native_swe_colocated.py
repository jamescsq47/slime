"""Native TP2 SWE500 report; select one role alias, count both rank timers.

Does not change a server or harness. Optional waiting is for this run's monitor
to finish, not a licence to treat a failed/partial run as a complete evaluation.
"""
import argparse
import bisect
from collections import defaultdict
import json
import math
from pathlib import Path
import time


def quantile(values, q):
    values = sorted(values)
    p = (len(values)-1)*q
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] + (values[hi]-values[lo])*(p-lo)


def interval_value(rows, key, begin, end, counter=False):
    points = sorted((t, m[key]) for t, m in rows if key in m and math.isfinite(m[key]))
    if not points:
        return None
    times = [t for t, _ in points]
    if counter and any(y < x for (_, x), (_, y) in zip(points, points[1:])):
        raise ValueError(f'Counter reset: {key}')

    def at(t):
        i = bisect.bisect_right(times, t)-1
        if i < 0:
            return 0.0 if counter else points[0][1]
        if i >= len(points)-1:
            return points[-1][1]
        a, x = points[i]; b, y = points[i+1]
        return x+(y-x)*(t-a)/(b-a)

    if counter:
        return (at(end)-at(begin))/(end-begin)
    clipped = [(begin, at(begin))]+[(t,v) for t,v in points if begin<t<end]+[(end,at(end))]
    return sum((b-a)*(x+y)/2 for (a,x),(b,y) in zip(clipped,clipped[1:]))/(end-begin)


def summarize(run):
    tasks = [json.loads(line) for line in (run/'requests.jsonl').open()]
    assert len(tasks) == len({r['metadata']['instance_id'] for r in tasks}) == 500
    preflight = json.loads((run/'preflight.json').read_text())
    assert preflight['tp'] == 2 and not preflight['pd']
    series = defaultdict(list)
    for line in (run/'engine_metrics.jsonl').open():
        row = json.loads(line)
        if row['role'] != 'prefill':
            continue  # decode is an alias of these same four servers.
        for endpoint in row.get('endpoint_metrics', []):
            series[endpoint['endpoint']].append((row['ts'], endpoint['metrics']))
    assert len(series) == 4
    begin = min(r['started_ts'] for r in tasks)
    end = max(r['finished_ts'] for r in tasks)
    wall = end-begin
    latencies = [r['finished_ts']-r['started_ts'] for r in tasks]
    ends = sorted(r['finished_ts'] for r in tasks)
    result = dict(run=str(run), requests=500, concurrency=preflight['concurrency'],
                  resolved=sum(bool(r['metadata'].get('swe_bench_verifier',{}).get('resolved')) for r in tasks),
                  wall_seconds=wall, latency_p50=quantile(latencies,.5), latency_p90=quantile(latencies,.9),
                  T450_seconds=ends[449]-begin,
                  mean_turns=sum(r['generation_turns'] for r in tasks)/500,
                  mean_decode_tokens=sum(r['response_tokens'] for r in tasks)/500,
                  windows={})
    for name,a,b in [('full',begin,end)]+([('middle_300_1500',begin+300,begin+1500)] if wall>=1500 else []):
        record = {}
        for label,key,is_counter,divisor in [
            ('prefill_token_s','sglang_realtime_tokens_total|mode=prefill_compute',True,1),
            ('decode_token_s','sglang_realtime_tokens_total|mode=decode',True,1),
            ('prefill_forward_fraction','sglang_forward_execution_seconds_total|category=extend',True,8),
            ('decode_forward_fraction','sglang_forward_execution_seconds_total|category=decode',True,8),
            ('attention_kv_fraction','sglang_full_token_usage',False,4),
            ('mamba_fraction','sglang_mamba_usage',False,4),
            ('running_per_tp_group','sglang_num_running_reqs',False,4),
        ]:
            values = [interval_value(rows,key,a,b,is_counter) for rows in series.values()]
            if any(v is None for v in values):
                raise ValueError(f'Missing required metric: {key}')
            record[label] = sum(values)/divisor
        record['decode_token_s_per_physical_gpu'] = record['decode_token_s']/8
        record['seconds'] = b-a
        record['max_scrape_gap_seconds'] = max(
            (t2-t1 for rows in series.values() for (t1,_),(t2,_) in zip(rows,rows[1:]) if t1<b and t2>a),default=0)
        result['windows'][name] = record
    result['actual_prefill_tokens_per_agent'] = result['windows']['full']['prefill_token_s']*wall/500
    return result


def render(result):
    lines = [f"# Qwen3.5-27B / TP2 / native collocated / c{result['concurrency']}", '',
             '| 全程指标 | 结果 |', '|---|---:|',
             f"| 完成记录 | {result['requests']} |",
             f"| 正确率 | {result['resolved']}/500 = {result['resolved']/5:.1f}% |"]
    for label,key in [('全完成墙钟秒','wall_seconds'),('T450秒','T450_seconds'),
                      ('题目延迟P50秒','latency_p50'),('题目延迟P90秒','latency_p90'),
                      ('平均轮数','mean_turns'),('平均输出tokens/题','mean_decode_tokens'),
                      ('实际Prefill计算tokens/题（计数器）','actual_prefill_tokens_per_agent')]:
        lines.append(f"| {label} | {result[key]:,.2f} |")
    lines += ['', '| 指标 | 全程 | 中段300–1500秒 |','|---|---:|---:|']
    fields = [('Prefill token/s','prefill_token_s',1),('Decode token/s','decode_token_s',1),
              ('Decode token/s/物理卡','decode_token_s_per_physical_gpu',1),
              ('Prefill Forward占比%','prefill_forward_fraction',100),
              ('Decode Forward占比%','decode_forward_fraction',100),
              ('Attention KV不可驱逐占用%','attention_kv_fraction',100),
              ('Mamba池不可驱逐占用%','mamba_fraction',100),
              ('Running/TP2组','running_per_tp_group',1),('最大采样间隔秒','max_scrape_gap_seconds',1)]
    for label,key,mult in fields:
        values=[f"{result['windows'][w][key]*mult:,.2f}" if w in result['windows'] else '—' for w in ('full','middle_300_1500')]
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines += ['', '口径：同服务P/D角色去重，只取prefill角色；Forward计数器包含两TP rank，除以8卡×窗口时间。',
              'KV/Mamba此处仅为各自池不可驱逐占用，不含可驱逐前缀缓存，也不是占整张HBM比例。总占用见主对照文档的缓存组成表。中段只是有限500题测评的固定窗口，不代表持续补充的稳态。',
              '实际Prefill计数器包含重算，不等于每轮完整Prompt累计，也不等于理想必要增量。',
              f"原始目录：`{result['run']}`", '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--wait',action='store_true')
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.wait:
        while True:
            state=json.loads((args.run/'monitor.json').read_text())
            if state['state'] in ('failed','cancelled'):
                raise RuntimeError(f"Run {state['state']}; refusing complete-result table")
            if state['state']=='finished':
                break
            time.sleep(30)
    result=summarize(args.run)
    (args.run/'native_colocated_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(render(result))


if __name__=='__main__':
    main()
