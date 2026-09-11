#!/usr/bin/env python3
"""Offline finite SWE evaluation metrics, with explicit wall and common windows.

Counter differences and gauge integrals are computed per endpoint; missing
scrapes are interpolated and their gaps reported. Not a replenished benchmark.
"""
import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path

from summarize_swe_h2d_window import stats


def window_metrics(series, begin, end):
    result = {}
    for (role, endpoint), rows in sorted(series.items()):
        metrics = {}
        for key in sorted({key for _, row in rows for key in row}):
            if not (key.startswith('sglang_realtime_tokens_total|') or
                    key.startswith('sglang_gpu_execution_seconds_total|') or
                    key in {'sglang_full_token_usage', 'sglang_mamba_usage',
                            'sglang_num_running_reqs', 'sglang_num_queue_reqs',
                            'sglang_num_decode_prealloc_queue_reqs',
                            'sglang_num_decode_transfer_queue_reqs',
                            'sglang_num_prefill_inflight_queue_reqs'}):
                continue
            valid = sorted((t, m[key]) for t, m in rows if key in m)
            times = [t for t, _ in valid]
            def value(t):
                i = bisect.bisect_right(times, t) - 1
                if i < 0:
                    return valid[0][1]
                if i >= len(valid)-1:
                    return valid[-1][1]
                a, x = valid[i]; b, y = valid[i+1]
                return x + (y-x)*(t-a)/(b-a)
            points = [(begin, value(begin))]
            points += [(t, v) for t, v in valid if begin < t < end]
            points += [(end, value(end))]
            counter = '_total|' in key
            metrics[key] = ((points[-1][1]-points[0][1]) if counter else
                            sum((b-a)*(x+y)/2 for (a,x),(b,y)
                                in zip(points, points[1:]))) / (end-begin)
        times = sorted(t for t, _ in rows)
        gaps = [b-a for a,b in zip(times,times[1:]) if a < end and b > begin]
        result.setdefault(role, {})[endpoint] = dict(
            metrics=metrics, scrape_max_gap_seconds=max(gaps, default=0),
            missing_window_start=times[0] > begin,
            missing_window_end=times[-1] < end)
    roles = {}
    for role, endpoints in result.items():
        values = defaultdict(list)
        for record in endpoints.values():
            for key, value in record['metrics'].items():
                values[key].append(value)
        roles[role] = dict(endpoints=len(endpoints), mean_per_gpu={
            k: sum(v)/len(v) for k,v in values.items()}, sum={
            k: sum(v) for k,v in values.items()})
    return dict(begin=begin, end=end, seconds=end-begin, roles=roles,
                endpoint_diagnostics=result)


def analyze(run):
    rows = []
    for line in (run/'requests.jsonl').open():
        r = json.loads(line); m = r.get('metadata', {})
        rows.append(dict(start=r['started_ts'], end=r['finished_ts'],
                         status=r['status'], instance=m.get('instance_id'),
                         turns=r['generation_turns'], decode=r['response_tokens'],
                         stop=m.get('stop_reason'),
                         verifier=m.get('swe_bench_verifier', {}),
                         tool_seconds=[t['command_seconds'] for t in m.get('turn_metrics', [])
                                       if t.get('command') and t.get('command_seconds') is not None]))
    start = min(r['start'] for r in rows); end = max(r['end'] for r in rows)
    milestones = {}
    for name, chosen in [('ended', rows), ('completed', [r for r in rows if r['status']=='completed'])]:
        ends = sorted(r['end'] for r in chosen)
        milestones[name] = {str(n): ends[n-1]-start for n in (10,20,50,90,100,250,450,500) if n <= len(ends)}
    series = defaultdict(list)
    for line in (run/'engine_metrics.jsonl').open():
        r = json.loads(line)
        for ep in r.get('endpoint_metrics', []):
            series[(r['role'], ep['endpoint'])].append((r['ts'], ep['metrics']))
    return dict(run=str(run), requests=len(rows), unique_instances=len({r['instance'] for r in rows}),
                status_counts=dict(Counter(r['status'] for r in rows)),
                stop_counts=dict(Counter(r['stop'] for r in rows)),
                first_start=start, last_end=end, wall_seconds=end-start,
                milestones_seconds=milestones,
                latency_seconds=stats([r['end']-r['start'] for r in rows]),
                turns=stats([r['turns'] for r in rows]),
                total_decode_tokens=sum(r['decode'] for r in rows),
                decode_per_agent=stats([r['decode'] for r in rows]),
                shell_seconds=stats([v for r in rows for v in r['tool_seconds']]),
                full=window_metrics(series,start,end),
                middle=window_metrics(series,start+300,start+1500))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))
