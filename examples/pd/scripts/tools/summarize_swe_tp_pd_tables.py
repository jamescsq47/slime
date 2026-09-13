"""Read-only TP PD finite-evaluation accounting. Never modifies serving state.

Token metrics are rank-0 counters; GPU timers contain all TP ranks. Pool usage
gauges are non-evictable occupancy, not total allocator occupancy. Log-based
stage intervals use second-resolution timestamps and are explicitly approximate.
"""
import argparse
import base64
from collections import Counter, defaultdict
import datetime as dt
import json
from pathlib import Path
import re

from summarize_native_swe_colocated import interval_value
from summarize_native_swe_tables import dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    tasks = [json.loads(x) for x in (run/'requests.jsonl').open()]
    assert len(tasks) == len({r['metadata']['instance_id'] for r in tasks}) == 500
    start = min(r['started_ts'] for r in tasks)
    end = max(r['finished_ts'] for r in tasks)
    events = [e for r in tasks for e in r['metadata']['openenv_trajectory']['turn_events']]
    tools = [e['command_seconds'] for e in events if e.get('command') and e.get('command_seconds') is not None]
    series = defaultdict(list)
    for line in (run/'engine_metrics.jsonl').open():
        row = json.loads(line)
        for ep in row.get('endpoint_metrics', []):
            series[(row['role'], ep['endpoint'])].append((row['ts'], ep['metrics']))
    result = dict(run=str(run), tp=args.tp, begin=start, end=end, seconds=end-start,
                  completed=len(tasks), resolved=sum(bool(r['metadata']['swe_bench_verifier'].get('resolved')) for r in tasks),
                  milestones={str(n): sorted(r['finished_ts'] for r in tasks)[n-1]-start for n in [10,20,50,90,100,250,450,490,500]},
                  latency=dist([r['finished_ts']-r['started_ts'] for r in tasks]),
                  prompt=dist([e['prompt_tokens'] for e in events]),
                  decode=dist([e['output_tokens'] for e in events]),
                  initial=dist([r['metadata']['openenv_trajectory']['turn_events'][0]['prompt_tokens'] for r in tasks]),
                  tools=dist(tools),
                  tool_bins=[sum(test(v) for v in tools) for test in [lambda v:v<=1,lambda v:1<v<=2,lambda v:2<v<=10,lambda v:v>10]],
                  compacted_events=sum(bool(e.get('context_compacted')) for e in events),
                  structured_action_errors=sum(bool(e.get('structured_action_error')) for e in events),
                  stop_counts=dict(Counter(r['metadata']['stop_reason'] for r in tasks)),
                  stop_resolved=dict(Counter(r['metadata']['stop_reason'] for r in tasks if r['metadata']['swe_bench_verifier'].get('resolved'))),
                  profile=json.loads((run/'swe_bench_profile_summary.json').read_text()), windows={})
    for label,a,b in [('full',start,end),('middle',start+300,start+1500)]:
        window = dict(seconds=b-a,completed=sum(a<=r['finished_ts']<=b for r in tasks),roles={})
        for role in ['prefill','decode']:
            groups = {ep:rows for (r,ep),rows in series.items() if r==role}
            rr = dict(groups=len(groups),metrics={},peaks={},capacity={})
            keys = ['sglang_full_token_usage','sglang_mamba_usage','sglang_num_running_reqs',
                    'sglang_num_queue_reqs','sglang_num_decode_prealloc_queue_reqs',
                    'sglang_num_decode_transfer_queue_reqs','sglang_num_prefill_inflight_queue_reqs']
            keys += ['sglang_realtime_tokens_total|mode='+m for m in ['prefill_compute','prefill_cache','decode']]
            keys += ['sglang_gpu_execution_seconds_total|category='+m for m in ['forward_extend','forward_decode']]
            for key in keys:
                counter = '_total|' in key
                values = [interval_value(rows,key,a,b,counter) for rows in groups.values()]
                divisor = len(groups)*args.tp if 'execution_seconds' in key else (1 if counter else len(groups))
                rr['metrics'][key] = sum(values)/divisor if all(v is not None for v in values) else None
                if not counter:
                    rr['peaks'][key] = max((m[key] for rows in groups.values() for t,m in rows if a<=t<=b and key in m),default=None)
            for ep,rows in groups.items():
                rr['capacity'][ep] = Counter(m['sglang_max_total_num_tokens'] for _,m in rows if m.get('sglang_max_total_num_tokens')).most_common(1)[0][0]
            rr['max_scrape_gap_s'] = max(t2-t1 for rows in groups.values() for (t1,_),(t2,_) in zip(rows,rows[1:]) if t1<b and t2>a)
            window['roles'][role]=rr
        result['windows'][label] = window
    print('Counter and trajectory accounting ready.',flush=True)
    # Small per-call records only: do not retain multi-GB rendered log text.
    records=[];seen=set(); skipped=Counter(); skipped_reasons=Counter(); decode_totals=Counter()
    owner_to_instance={r['metadata']['agentic_request_id']:r['metadata']['instance_id'] for r in tasks}
    for directory in sorted((run/'raw').iterdir()):
        if not directory.name.startswith(('prefill-','decode-')):continue
        for path in sorted(directory.glob('*.log*')):
            for line in path.open():
                if '"event": "request.finished"' not in line:continue
                raw=json.loads(line[line.index('{'):])
                identity=(directory.name.split('-')[0],raw['rid'])
                if identity in seen:continue
                extra=raw['obj'].get('extra_key','') or ''
                if not extra.startswith('agentic-v1e:'):continue
                payload=extra.split(':')[2]
                payload=json.loads(base64.urlsafe_b64decode(payload+'='*(-len(payload)%4)))
                owner=payload['agentic_request_id']
                if owner not in owner_to_instance:continue
                m=raw['out']['meta_info']
                if 'request_received_ts' not in m or 'prompt_tokens' not in m:
                    skipped[directory.name]+=1
                    skipped_reasons[str(m.get('finish_reason'))]+=1
                    continue
                seen.add(identity)
                if directory.name.startswith('decode-'):
                    decode_totals['calls']+=1
                    decode_totals['prompt']+=m['prompt_tokens']
                    decode_totals['decode']+=m.get('completion_tokens',0)
                    decode_totals['uncached']+=m['prompt_tokens']-m.get('cached_tokens',0)
                    decode_totals['retractions']+=m.get('total_retractions',m.get('num_retractions',0))
                    continue
                records.append(dict(instance=owner_to_instance[owner],generation=payload['agentic_generation'],
                                    ts=m['request_received_ts'],prompt=m['prompt_tokens'],cached=m.get('cached_tokens',0),
                                    retractions=m.get('total_retractions',m.get('num_retractions',0))))
    per_task=defaultdict(Counter)
    for r in records:
        c=per_task[r['instance']];c['calls']+=1
        for k in ['prompt','cached','retractions']:c[k]+=r[k]
        c['uncached']+=r['prompt']-r['cached']
    mismatches=[]
    for task in tasks:
        instance=task['metadata']['instance_id'];c=per_task[instance]
        prompt=sum(e['prompt_tokens'] for e in task['metadata']['openenv_trajectory']['turn_events'])
        if c['calls']!=task['generation_turns'] or c['prompt']!=prompt:
            mismatches.append(dict(instance=instance,raw=dict(c),harness_calls=task['generation_turns'],harness_prompt=prompt))
    result['raw_prefill']=dict(totals={k:sum(c[k] for c in per_task.values()) for k in ['calls','prompt','cached','uncached','retractions']},
                               mismatches=mismatches,per_task=per_task,
                               middle_retractions=sum(r['retractions'] for r in records if start+300<=r['ts']<start+1500))
    result['raw_decode']=dict(decode_totals)
    result['raw_missing_metrics_skipped']=dict(skipped)
    result['raw_skipped_finish_reasons']=dict(skipped_reasons)
    print('Raw Prefill accounting ready.',flush=True)
    # Keep all ranks and use the last completion in an atomic TP group. These
    # counts are snapshots, not TP shards or retry attempts.
    stages=defaultdict(lambda:defaultdict(list)); arrivals={}; event_counts=Counter()
    relevant={'shared_host_d2h_complete','shared_host_h2d_complete','shared_host_materialize_complete',
              'shared_host_group_commit_release','shared_host_final_release','early_direct_group_complete'}
    for path in (run/'logs').glob('*.log*'):
        for line in path.open(errors='replace'):
            if 'PD_EARLY_CLAIM_ARRIVAL snapshot=' in line:
                v=dict(re.findall(r'(\w+)=([^\s]+)',line));sid=v['snapshot'];t=float(v['arrived_at'])
                arrivals[sid]=min(arrivals.get(sid,t),t)
            match=re.search(r'AgenticKV (\w+)',line)
            if not match:continue
            event=match[1];event_counts[event]+=1
            if event not in relevant:continue
            v=dict(re.findall(r'(\w+)=([^\s]+)',line));sid=v.get('snapshot')
            if not sid:continue
            v['time']=dt.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()+.5
            stages[sid][event].append(v)
    intervals=defaultdict(list); missing=0
    for sid,s in stages.items():
        done=s.get('shared_host_h2d_complete');durable=s.get('shared_host_d2h_complete')
        if not done or not durable:continue
        if sid not in arrivals:missing+=1;continue
        ready=max(float(v['completed_at']) for v in durable)
        eligible=max(ready,arrivals[sid])
        hbm=max(v['time'] for v in done)
        io=min(v['time']-float(v['wall_ms'])/1000 for v in done)
        intervals['host_ready_arrived_wait_io'].append((eligible,max(eligible,io)))
        intervals['host_ready_arrived_wait_hbm'].append((eligible,max(eligible,hbm)))
        intervals['host_wait_next_arrival'].append((ready,max(ready,arrivals[sid])))
        intervals['h2d_io_wall_envelope'].append((max(eligible,io),max(eligible,hbm)))
    result['paths']={'event_counts':dict(event_counts),'missing_host_arrivals':missing,'windows':{}}
    for label,a,b in [('full',start,end),('middle',start+300,start+1500)]:
        paths={}
        for name,ev in [('direct','early_direct_group_complete'),('slow','shared_host_group_commit_release')]:
            paths[name]=sum(a<=max(v['time'] for v in s[ev])<b for s in stages.values() if ev in s)
        paths['mean_snapshot_occupancy_approx']={name:sum(max(0,min(y,b)-max(x,a)) for x,y in spans)/(b-a) for name,spans in intervals.items()}
        result['paths']['windows'][label]=paths
    result['tail']=[dict(instance=r['metadata']['instance_id'],finished_after_s=r['finished_ts']-start,
                         latency=r['agent_latency_seconds'],model=r['metadata']['model_time'],tool=r['tool_time_seconds'],
                         verifier=r['metadata']['swe_bench_verifier'].get('duration_seconds'),
                         verifier_status=r['metadata']['swe_bench_verifier'].get('status')) for r in sorted(tasks,key=lambda r:r['finished_ts'])[-3:]]
    result['limitations']=['Pool gauges are non-evictable usage; full occupied/free split was not exported.',
                           'No ideal increment inferred from prompt lengths alone: exact token/checkpoint alignment requires separate audit.',
                           'Host stage occupancy uses one-second log timestamps; TP ranks are grouped. No byte-occupancy history inferred from final tombstones.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Wrote',args.output,flush=True)


if __name__=='__main__':
    main()
