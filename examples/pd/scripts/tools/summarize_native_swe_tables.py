"""Offline native SWE counters, pool composition, tool and raw-call accounting.

No server/harness changes. Logs contain rendered text but not exact prompt IDs;
re-tokenization failed the length check, so no ideal-Prefill number is inferred.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from summarize_native_swe_colocated import interval_value, quantile


def dist(values):
    return dict(count=len(values), mean=sum(values)/len(values),
                p50=quantile(values, .5), p90=quantile(values, .9),
                p99=quantile(values, .99), total=sum(values)) if values else None


def scan_group(args):
    directory = args
    records, seen = [], set()
    for path in sorted(Path(directory).glob('*.log*')):
        for line in path.open():
            if '"event": "request.finished"' not in line:
                continue
            raw = json.loads(line[line.index('{'):])
            if raw['rid'].startswith('HEALTH_CHECK') or raw['rid'] in seen:
                continue
            meta = raw['out']['meta_info']
            text = raw['obj'].get('text')
            if not isinstance(text, str) or 'prompt_tokens' not in meta:
                continue
            seen.add(raw['rid'])
            records.append((meta['request_received_ts'], text,
                            meta))
    records.sort(key=lambda x:x[0])
    result = []
    for ts, text, meta in records:
        user = text.split('<|im_start|>user\n', 1)[1].split('<|im_end|>', 1)[0]
        user = user.replace('\r\n', '\n').strip()
        result.append(dict(user=user, ts=ts, prompt=meta['prompt_tokens'],
                           decode=meta['completion_tokens'], cached=meta.get('cached_tokens',0),
                           retractions=meta.get('num_retractions',0)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tasks=[json.loads(line) for line in (args.run/'requests.jsonl').open()]
    assert len(tasks)==len({x['metadata']['instance_id'] for x in tasks})==500
    begin=min(x['started_ts'] for x in tasks)
    end=max(x['finished_ts'] for x in tasks)
    series=defaultdict(list)
    for line in (args.run/'engine_metrics.jsonl').open():
        row=json.loads(line)
        if row['role']=='prefill':
            for endpoint in row.get('endpoint_metrics',[]):
                m=dict(endpoint['metrics'])
                for pool in ['kv','mamba']:
                    keys=[f'sglang_{pool}_{s}_tokens' for s in ['used','evictable','available']]
                    if all(k in m for k in keys):
                        cap=sum(m[k] for k in keys)
                        if cap:
                            m[f'{pool}_capacity']=cap
                            for s in ['used','evictable','available']:
                                m[f'{pool}_{s}_fraction']=m[f'sglang_{pool}_{s}_tokens']/cap
                            m[f'{pool}_occupied_fraction']=1-m[f'{pool}_available_fraction']
                series[endpoint['endpoint']].append((row['ts'],m))
    # Metrics are updated asynchronously. A transient sum of the three pool
    # components is not a change in configured capacity; use its modal value.
    for rows in series.values():
        for pool in ['kv','mamba']:
            counts=Counter(m[f'{pool}_capacity'] for _,m in rows if f'{pool}_capacity' in m)
            capacity=counts.most_common(1)[0][0]
            for _,m in rows:
                if f'{pool}_capacity' not in m:
                    continue
                m[f'{pool}_capacity']=capacity
                for s in ['used','evictable','available']:
                    m[f'{pool}_{s}_fraction']=m[f'sglang_{pool}_{s}_tokens']/capacity
                m[f'{pool}_occupied_fraction']=1-m[f'{pool}_available_fraction']
    windows={}
    for label,a,b in [('full',begin,end),('middle_300_1500',begin+300,begin+1500)]:
        w=dict(completed=sum(a<=t['finished_ts']<=b for t in tasks), seconds=b-a)
        for key in ['sglang_num_running_reqs','sglang_num_queue_reqs',
                    'sglang_num_decode_prealloc_queue_reqs','sglang_num_decode_transfer_queue_reqs']+[
                        f'{p}_{s}' for p in ['kv','mamba'] for s in [
                            'capacity','used_fraction','evictable_fraction','available_fraction','occupied_fraction']]:
            vals=[interval_value(rows,key,a,b) for rows in series.values()]
            w[key]=sum(vals)/4 if all(v is not None for v in vals) else None
            w[key+'_max']=max((m[key] for rows in series.values() for t,m in rows if a<=t<=b and key in m), default=None)
        for mode in ['prefill_compute','prefill_cache','decode']:
            vals=[interval_value(rows,'sglang_realtime_tokens_total|mode='+mode,a,b,True) for rows in series.values()]
            w[mode+'_tokens']=sum(vals)*(b-a) if all(v is not None for v in vals) else None
        windows[label]=w
    events=[e for t in tasks for e in t['metadata']['openenv_trajectory']['turn_events']]
    tools=[e['command_seconds'] for e in events if e.get('command') and e.get('command_seconds') is not None]
    result=dict(run=str(args.run),begin=begin,end=end,windows=windows,
                prompt=dist([e['prompt_tokens'] for e in events if 'prompt_tokens' in e]),
                decode=dist([e['output_tokens'] for e in events if 'output_tokens' in e]),
                initial_prompt=dist([t['metadata']['openenv_trajectory']['turn_events'][0]['prompt_tokens'] for t in tasks]),
                tools=dist(tools),
                tool_bins=[sum(test(s) for s in tools) for test in [lambda s:s<=1,lambda s:1<s<=2,lambda s:2<s<=10,lambda s:s>10]],
                compacted_events=sum(bool(e.get('context_compacted')) for e in events),
                structured_action_errors=sum(bool(e.get('structured_action_error')) for e in events))
    print('Counters/tool summaries ready; checking raw per-call accounting.',flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        groups=list(pool.map(scan_group,[str(d) for d in sorted(args.run.glob('raw-[0-9]'))]))
    owners=defaultdict(set)
    for i,g in enumerate(groups):
        for c in g:
            owners[c['user']].add(i)
    result['cross_worker_tasks']=sum(len(v)>1 for v in owners.values())
    calls=[x for g in groups for x in g]
    taskmap={t['metadata']['openenv_trajectory']['instruction'].replace('\r\n','\n').strip():t for t in tasks}
    per_task=defaultdict(Counter)
    fallback_users=set()
    for call in calls:
        task=taskmap.get(call['user'])
        if task is None:
            # Some engine-rendered issue text has normalized special characters.
            # Require a unique issue header, then validate all turn/token totals.
            candidates=[t for user,t in taskmap.items() if user[:128]==call['user'][:128]]
            assert len(candidates)==1, call['user'][:128]
            task=candidates[0]
            fallback_users.add(call['user'])
        row=per_task[task['metadata']['instance_id']]
        for k in ['prompt','decode','cached','retractions']:
            row[k]+=call[k]
        row['calls']+=1
    assert len(per_task)==500
    for t in tasks:
        r=per_task[t['metadata']['instance_id']]
        assert r['calls']==t['generation_turns'], (t['metadata']['instance_id'],r['calls'],t['generation_turns'])
        # Top-level model_prompt_tokens is zero in this harness. Read events.
        r['harness_prompt_tokens']=sum(e['prompt_tokens'] for e in t['metadata']['openenv_trajectory']['turn_events'])
        assert r['prompt']==r['harness_prompt_tokens'], t['metadata']['instance_id']
        r['uncached_prompt_tokens']=r['prompt']-r['cached']
    result['raw_totals']={k:sum(r[k] for r in per_task.values()) for k in next(iter(per_task.values()))}
    result['issue_header_fallback_tasks']=len(fallback_users)
    result['raw_windows']={label:dict(calls=len(v),retractions=sum(c['retractions'] for c in v)) for label,v in [
        ('full',calls),('middle_300_1500',[c for c in calls if begin+300<=c['ts']<begin+1500])]}
    result['per_task']=per_task
    result['ideal_prefill_tokens']=None
    result['ideal_prefill_limitation']='Exact prompt token IDs/checkpoint insertion events not recorded. Rendered-text re-tokenization length check failed (example: 26523 vs engine 26531 tokens); no precise LCP/ideal/excess inferred.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_task'},indent=2))


if __name__=='__main__':
    main()
