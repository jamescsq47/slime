#!/usr/bin/env python3
"""Offline comparison of the five retained P2D Host experiments; no GPU use."""
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'runs-host'
OUT = ROOT / 'analysis/p2d-host-c512-c576'
RUNS = {
    '512-full': 'current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-slow-congestion-1s-20260909-r1',
    '512-off': 'current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-host-disabled-20260910-r1',
    '576-full': 'current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-slow-congestion-1s-20260909-r1',
    '576-off': 'current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-p2d-direct-only-20260910-r1',
    '576-pool': 'current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-numa-pool-288g-r1',
}
COUNTERS = {'p_tokens': 'sglang_realtime_tokens_total|mode=prefill_compute',
            'd_tokens': 'sglang_realtime_tokens_total|mode=decode',
            'p_gpu': 'sglang_gpu_execution_seconds_total|category=forward_extend',
            'd_gpu': 'sglang_gpu_execution_seconds_total|category=forward_decode'}
GAUGES = {'kv': 'sglang_token_usage', 'running': 'sglang_num_running_reqs',
          'queue': 'sglang_num_queue_reqs', 'inflight': 'sglang_num_prefill_inflight_queue_reqs',
          'transfer': 'sglang_num_decode_transfer_queue_reqs', 'used': 'sglang_num_used_tokens'}
STAMP = re.compile(r'^\[([^\]]+)\]')
EVENT = re.compile(r'AgenticKV (\w+)')
FIELD = re.compile(r'(\w+)=([^\s,]+)')
BATCH = re.compile(r'Decode batch, #running-req: (\d+), #token: (\d+)')
BYTE_PER_TOKEN = 147456

def integrate(bucket, a, b, name, value):
    a, b = max(a, 0), min(b, 1200)
    if b <= a:
        return
    for i in range(int(a // 30), min(40, int((b - 1e-8) // 30) + 1)):
        bucket[i][name] += max(0, min(b, (i + 1) * 30) - max(a, i * 30)) * value

def parse_run(name, relative):
    path = ROOT / relative
    bounds = json.loads((path / 'closed_loop_boundaries.json').read_text())
    start, end = bounds['measurement_start_wall'], bounds['measurement_end_wall']
    bins = [defaultdict(float) for _ in range(40)]
    previous = {}
    conditional = defaultdict(lambda: defaultdict(float))
    joint = defaultdict(lambda: defaultdict(float))
    gauges = []
    for line in (path / 'engine_metrics.jsonl').open():
        row = json.loads(line)
        role = row['role'][0]
        for rank, endpoint in enumerate(row.get('endpoint_metrics', [])):
            key = role, rank
            old = previous.get(key)
            previous[key] = row['ts'], endpoint['metrics']
            if old is None:
                continue
            t0, m0 = old
            t1, m1 = previous[key]
            dt = t1 - t0
            if dt <= 0:
                continue
            for short, metric in COUNTERS.items():
                if short[0] == role and metric in m0 and metric in m1:
                    rate = max(0, m1[metric] - m0[metric]) / dt
                    integrate(bins, t0-start, t1-start, short, rate)
            for short, metric in GAUGES.items():
                integrate(bins, t0-start, t1-start, role+'_'+short, m0.get(metric, 0) / 4)
            weight = max(0, min(t1, end) - max(t0, start))
            if role == 'd' and weight:
                running = m0.get(GAUGES['running'], 0)
                g = conditional[int(running // 16) * 16]
                g['seconds'] += weight
                g['running'] += weight * running
                g['used'] += weight * m0.get(GAUGES['used'], 0)
                for short in ('d_tokens', 'd_gpu'):
                    metric = COUNTERS[short]
                    g[short] += max(0, m1.get(metric, 0)-m0.get(metric, 0)) / dt * weight
                gauges.append((running, weight))
                cell = f'{rank}:{int(running//8)*8}:{int(m0.get(GAUGES["used"],0)//32768)*32768}'
                j = joint[cell]
                j['seconds'] += weight
                for short in ('d_tokens', 'd_gpu'):
                    metric = COUNTERS[short]
                    j[short] += max(0,m1.get(metric,0)-m0.get(metric,0))/dt*weight

    events = defaultdict(dict)
    p_releases = {}
    seen_path = set()
    for log in sorted((path / 'logs').glob('*.log')):
        if not (log.name.startswith('prefill') or log.name.startswith('decode')):
            continue
        for line in log.open(errors='replace'):
            if not ('Decode batch' in line or 'AgenticKV' in line):
                continue
            match = STAMP.search(line)
            if not match:
                continue
            ts = datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
            batch = BATCH.search(line)
            if batch and start <= ts < end:
                b = bins[min(39, int((ts-start)//30))]
                b['step_estimate'] += 40
                b['logged_batch_sum'] += int(batch[1])
                b['logged_batch_count'] += 1
            event = EVENT.search(line)
            if not event:
                continue
            f = dict(FIELD.findall(line))
            if event[1] == 'p_to_d_release' and 'req' in f:
                p_releases.setdefault(f['req'], (ts, float(f.get('tokens',0))))
            snapshot = f.get('snapshot')
            if not snapshot:
                continue
            ev = event[1]
            f['ts'] = ts
            events[snapshot].setdefault(ev, f)
            if start <= ts < end:
                b = bins[min(39, int((ts-start)//30))]
                if ev in ('direct_rank_send_complete', 'direct_fallback', 'fast_direct_failure_recompute'):
                    if (snapshot, ev) not in seen_path:
                        b[ev] += 1
                        seen_path.add((snapshot, ev))

    transfer = defaultdict(lambda: defaultdict(float))
    host_requests = {e['p2d_host_prefill_release'].get('req') for e in events.values() if 'p2d_host_prefill_release' in e}
    for req,(ts,tokens) in p_releases.items():
        if req not in host_requests and start <= ts < end:
            b = bins[min(39,int((ts-start)//30))]
            b['delivery_count'] += 1
            b['delivery_tokens'] += tokens
    for snap, evs in events.items():
        p2d = snap.startswith('p2d:')
        write = 'p2d_host_d2h_complete' if p2d else 'shared_host_d2h_complete'
        read = 'p2d_host_h2d_complete' if p2d else 'shared_host_h2d_complete'
        if write not in evs:
            continue
        tokens = next((float(e['tokens']) for e in evs.values() if 'tokens' in e), 0)
        byte_size = next((float(e['bytes']) for e in evs.values() if 'bytes' in e), tokens*BYTE_PER_TOKEN)
        gib = byte_size / 2**30
        t0 = evs[write]['ts']
        t1 = evs.get(read, {}).get('ts', end)
        integrate(bins, t0-start, t1-start, ('p2d' if p2d else 'd2p')+'_host_gib', gib)
        for event, direction in ((write, 'p_d2h' if p2d else 'd_d2h'), (read, 'd_h2d' if p2d else 'p_h2d')):
            if event not in evs:
                continue
            e = evs[event]
            ts = e['ts']
            if not start <= ts < end:
                continue
            b = bins[min(39, int((ts-start)//30))]
            if event == 'p2d_host_h2d_complete':
                b['delivery_count'] += 1
                b['delivery_tokens'] += tokens
                b['host_delivery_count'] += 1
            b[direction+'_gib'] += gib
            gpu_ms = float(e.get('gpu_ms', e.get('elapsed_ms', 0)))
            wall_ms = float(e.get('wall_ms', e.get('elapsed_ms', 0)))
            t = transfer[direction]
            t['count'] += 1
            t['gib'] += gib
            t['gpu_seconds'] += gpu_ms/1000
            t['wall_seconds'] += wall_ms/1000
            b[direction+'_gpu_seconds'] += gpu_ms/1000

    completed = []
    admissions = {}
    for line in (path / 'requests.jsonl').open():
        r = json.loads(line)
        if r.get('error') or not isinstance(r.get('finished_ts'), (int, float)):
            continue
        m = r.get('metadata', {})
        item = {k: r.get(k, 0) for k in ('generation_turns', 'model_prompt_tokens', 'model_completion_tokens', 'response_tokens')}
        item['source'] = m.get('source_position')
        admissions[str(m.get('closed_loop_admission_id', r.get('sample_index')))] = item
        if start <= r['finished_ts'] < end:
            completed.append(item)
            b = bins[min(39, int((r['finished_ts']-start)//30))]
            b['completed'] += 1
            for k,v in item.items():
                if k != 'source' and isinstance(v, (float,int)):
                    b['completed_'+k] += v
    result_bins = []
    for i,b in enumerate(bins):
        z = dict(b)
        z.update(run=name, start_s=i*30, end_s=(i+1)*30)
        for k in list(z):
            if k.startswith(('p_', 'd_')) and not k.endswith('_gib'):
                # Gauges and counters were integrated as sums over a 30-second bin.
                if k not in ('p_d2h_gpu_seconds','d_h2d_gpu_seconds','d_d2h_gpu_seconds','p_h2d_gpu_seconds'):
                    z[k] /= 30
        for k in ('p2d_host_gib', 'd2p_host_gib'):
            z[k] = b.get(k, 0)/30
        result_bins.append(z)
    weights = sum(w for _,w in gauges)
    stats = {'running_lt32_fraction': sum(w for r,w in gauges if r<32)/weights,
             'running_ge96_fraction': sum(w for r,w in gauges if r>=96)/weights,
             'step_tokens': sum(b['d_tokens'] for b in bins)/sum(b['step_estimate'] for b in bins),
             'step_gpu_ms': 1000*sum(b['d_gpu'] for b in bins)/sum(b['step_estimate'] for b in bins),
             'conditional': dict(conditional), 'joint_running_used_rank':dict(joint), 'transfer': dict(transfer),
             'completed_count':len(completed),
             'completed_means':{k:float(np.mean([x[k] for x in completed])) for k in ('generation_turns','model_prompt_tokens','model_completion_tokens','response_tokens')}}
    return result_bins, stats, admissions

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_bins, stats, admissions = [], {}, {}
    for run,path in RUNS.items():
        b,s,a = parse_run(run,path)
        all_bins += b
        stats[run] = s
        admissions[run] = a
        print(run, 'done', flush=True)
    columns = sorted(set().union(*(r.keys() for r in all_bins)))
    with (OUT/'time_slices_30s.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_bins)
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2))
    matched = {}
    for a,b in [('512-full','512-off'),('576-full','576-off'),('576-full','576-pool')]:
        ids = [k for k in admissions[a].keys() & admissions[b].keys()
               if admissions[a][k]['source']==admissions[b][k]['source']]
        matched[a+' vs '+b] = {'n':len(ids), 'same_decode_fraction':float(np.mean([admissions[a][k]['model_completion_tokens']==admissions[b][k]['model_completion_tokens'] for k in ids])),
            'means':{r:{field:float(np.mean([admissions[r][k][field] for k in ids])) for field in ('generation_turns','model_prompt_tokens','model_completion_tokens')} for r in (a,b)}}
    (OUT/'matched_admissions.json').write_text(json.dumps(matched,indent=2))
    fig,axes=plt.subplots(6,2,figsize=(16,18),sharex=True)
    panels=[('d_tokens','Decode total token/s',1),('d_running','D running / GPU',1),('d_kv','D KV %',100),('p_gpu','P Forward % / GPU',25),('p_kv','P KV %',100),('p2d_host_gib','P-to-D Host durable GiB',1)]
    for col,run_names in enumerate([['512-full','512-off'],['576-full','576-off','576-pool']]):
        for name in run_names:
            rows=[r for r in all_bins if r['run']==name]
            for row,(key,title,scale) in enumerate(panels):
                axes[row,col].plot([r['start_s']+15 for r in rows],[r.get(key,0)*scale for r in rows],label=name)
                axes[row,col].set_ylabel(title)
                axes[row,col].grid(alpha=.2)
        axes[0,col].legend()
        axes[0,col].set_title('c512' if col==0 else 'c576')
        axes[-1,col].set_xlabel('Seconds after measurement start (30 s bins)')
    fig.tight_layout()
    fig.savefig(OUT/'time_slices_30s.png',dpi=160)

if __name__=='__main__':
    main()
