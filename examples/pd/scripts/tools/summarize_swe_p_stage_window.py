#!/usr/bin/env python3
"""Offline TP=1 P-side snapshot occupancy and SWE tool distributions.

No serving mutations. Counts integrate unique snapshot lifetimes, clipping at
the window boundaries, rather than applying Little's law to completed cohorts.
Worker log times use the center of their one-second timestamp bucket.
"""
import argparse
import bisect
import datetime as dt
import json
import re
import statistics
import zlib
from collections import defaultdict
from pathlib import Path

from summarize_swe_h2d_window import stats


def overlap(start, finish, begin, end):
    return max(0.0, min(finish, end) - max(start, begin))


def occupancy(intervals, begin, end):
    edges = defaultdict(float)
    touched = 0
    for a, b in intervals:
        a, b = max(a, begin), min(b, end)
        if b > a:
            edges[a] += 1
            edges[b] -= 1
            touched += 1
    active = area = occupied = 0.0
    maximum = 0
    last = begin
    segments = []
    for time in sorted(set(edges) | {begin, end}):
        duration = time - last
        area += duration * active
        occupied += duration * (active > 0)
        if duration:
            segments.append((active, duration))
        active += edges[time]
        maximum = max(maximum, int(active))
        last = time
    def quantile(q):
        acc = 0
        for count, duration in sorted(segments):
            acc += duration
            if acc >= q * (end - begin):
                return count
        return 0
    return dict(mean=area/(end-begin), request_seconds=area,
                nonempty_time_fraction=occupied/(end-begin), p50=quantile(.5),
                p90=quantile(.9), maximum=maximum, intersecting_snapshots=touched)


def distribution(values):
    result = stats(values)
    if not values:
        return result
    for q in (.95, .99):
        x = sorted(values); k=(len(x)-1)*q; i=int(k)
        result[f'p{int(q*100)}'] = x[i]+(x[min(i+1,len(x)-1)]-x[i])*(k-i)
    bounds = [0, .5, 1, 2, 5, 10, 60, 300, float('inf')]
    result['bins'] = [dict(lower=a, upper=None if b==float('inf') else b,
                          count=sum(a < x <= b for x in values),
                          fraction=sum(a < x <= b for x in values)/len(values))
                      for a,b in zip(bounds,bounds[1:])]
    result['sum_seconds'] = sum(values)
    return result


def analyze(path):
    commands={}; task_ends={}; all_starts=[]; agent_intervals=[]
    for line in (path/'requests.jsonl').open():
        r=json.loads(line);m=r.get('metadata',{});rid=m.get('agentic_request_id')
        all_starts.append(r['started_ts']);task_ends[rid]=r['finished_ts']
        agent_intervals.append((r['started_ts'],r['finished_ts']))
        for t in m.get('turn_metrics',[]):
            if t.get('command') and t.get('command_seconds') is not None:
                commands[f"{rid}:{int(t['turn'])-1}"]=float(t['command_seconds'])
    begin,end=min(all_starts)+300,min(all_starts)+1500
    root=path/'control-final/ready'
    metadata={}
    metadata_tombstones=0
    for f in (root/'snapshot-metadata').iterdir():
        if f.name.endswith('.lock'):continue
        data=f.read_bytes()
        if data.startswith(b'fallback:'):
            metadata_tombstones+=1
            continue
        v=json.loads(data if data.lstrip().startswith(b'{') else zlib.decompress(data))
        metadata[f"{v['request_id']}:{v['generation']}"]=v
    def markers(kind):
        return {v['snapshot_id']:v for f in (root/'early-claims'/kind).glob('*.json')
                for v in [json.loads(f.read_text())]}
    arrivals=markers('arrivals'); finals=markers('finals')
    retained_arrivals=len(arrivals)
    router_arrivals={}
    for f in (path/'logs').glob('router.log*'):
        for line in f.open(errors='replace'):
            if 'PD_EARLY_CLAIM_ARRIVAL snapshot=' not in line:continue
            values=dict(re.findall(r'(\w+)=([^\s]+)',line))
            sid=values['snapshot'];time=float(values['arrived_at'])
            router_arrivals[sid]=min(router_arrivals.get(sid,time),time)
    for sid,time in router_arrivals.items():
        if sid not in arrivals or time<arrivals[sid]['arrived_at']:
            arrivals[sid]={'arrived_at':time,'source':'router_log_first'}
    stages=defaultdict(dict); fast_offset={}; controls=[]
    relevant={'shared_host_d2h_complete','shared_host_h2d_start','shared_host_h2d_complete',
              'shared_host_h2d_release','shared_host_final_release',
              'shared_host_materialize_complete','early_direct_start',
              'early_direct_complete','early_direct_bind','fast_arrival_seen'}
    for f in sorted((path/'logs').glob('*.log*')):
        if not f.name.startswith(('prefill-','decode-')):continue
        for line in f.open(errors='replace'):
            match=re.search(r'AgenticKV (\w+)',line)
            if not match or match[1] not in relevant:continue
            values=dict(re.findall(r'(\w+)=([^\s]+)',line))
            sid=values.get('snapshot')
            if not sid:continue
            time=dt.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()+.5
            values.update(time=time,worker=f.name)
            stages[sid][match[1]]=values
            if match[1]=='fast_arrival_seen':fast_offset[sid]=float(values['tool_elapsed_s'])
    # Fast arrival logs retain the exact offset from manifest.created_at.
    recovered_arrivals=0
    for sid,offset in fast_offset.items():
        if sid not in arrivals and sid in metadata:
            arrivals[sid]={'arrived_at':metadata[sid]['created_at']+offset,'source':'fast_log'}
            recovered_arrivals+=1
    intervals=defaultdict(list);missing=[];matched=0;duration_inconsistent=[]
    queue_lower=[];queue_upper=[]
    direct_recovered=0
    for sid,e in stages.items():
        d=e.get('shared_host_d2h_complete')
        if not d:continue
        durable=float(d['completed_at'])
        done=e.get('shared_host_h2d_complete'); release=e.get('shared_host_h2d_release')
        final_release=e.get('shared_host_final_release')
        if done and release:
            hbm=done['time'];released=max(hbm,release['time'])
            io=hbm-float(done['wall_ms'])/1000
            arrival=arrivals.get(sid,{}).get('arrived_at')
            if arrival is None:
                missing.append(sid)
                continue
            matched+=1
            eligible=max(durable,arrival)
            for dest, delta in ((queue_lower,-.5),(queue_upper,.5)):
                dest.append((eligible,max(eligible,io+delta)))
            intervals['host_eligible_not_yet_hbm_ready'].append((eligible,max(eligible,hbm)))
            # Equivalent in-I/O occupancy uses its measured duration directly;
            # pointwise concurrency cannot be inferred at sub-second precision.
            intervals['io_wall_equivalent'].append((io,hbm))
            io=max(eligible,io);hbm=max(io,hbm);released=max(hbm,released)
            intervals['host_wait_next_arrival'].append((durable,arrival))
            intervals['arrived_wait_host_durable'].append((arrival,durable))
            intervals['host_eligible_wait_io'].append((eligible,io))
            mat=e.get('shared_host_materialize_complete',{}).get('time',io)
            mat=min(io,max(eligible,mat))
            intervals['eligible_before_materialize'].append((eligible,mat))
            intervals['materialized_before_io'].append((mat,io))
            intervals['host_io'].append((io,hbm))
            intervals['hbm_ready_before_radix_release'].append((hbm,released))
        elif final_release:
            final=finals.get(sid,{}).get('arrived_at')
            if final is not None:
                intervals['host_wait_final_marker'].append((durable,final))
                intervals['final_marker_before_host_release'].append((max(durable,final),final_release['time']))
        else:missing.append(sid)
    # All actual commands, with observable brackets for their execution. The
    # D-created timestamp precedes HTTP return; arrival/final is after tool end.
    # We do not invent precise shell start/end timestamps inside this bracket.
    low=high=0.0; bracketed=0;missing_tools=[];mid_tools=[];proxy_intervals=[]
    missing_tool_max_seconds=0.0
    for sid,seconds in commands.items():
        meta=metadata.get(sid)
        if not meta:
            missing_tools.append(sid);missing_tool_max_seconds+=seconds;continue
        a=meta['created_at']
        if begin<=a<end:mid_tools.append(seconds)
        b=arrivals.get(sid,{}).get('arrived_at',finals.get(sid,{}).get('arrived_at'))
        if b is None:
            missing_tools.append(sid);missing_tool_max_seconds+=seconds;continue
        bracketed+=1
        if b-a+.002<seconds:
            duration_inconsistent.append(dict(snapshot=sid,seconds=seconds,bracket=b-a))
        inside=overlap(a,b,begin,end)
        outside=max(0,b-a)-inside
        low+=max(0,seconds-outside)
        high+=min(seconds,inside)
        proxy_intervals.append((a,b))
    # Exact endpoint counter deltas and piecewise-linear gauge averages.
    series=defaultdict(list)
    for line in (path/'engine_metrics.jsonl').open():
        r=json.loads(line)
        if r['role']!='prefill':continue
        for ep in r.get('endpoint_metrics',[]):
            series[ep['endpoint']].append((r['ts'],ep['metrics']))
    forward_key='sglang_gpu_execution_seconds_total|category=forward_extend'
    names=[forward_key,'sglang_num_queue_reqs','sglang_num_prefill_prealloc_queue_reqs',
           'sglang_num_prefill_inflight_queue_reqs','sglang_full_token_usage','sglang_mamba_usage']
    metrics={}; busy_bins=[]
    for ep,rows in series.items():
        metrics[ep]={}
        for key in names:
            valid=sorted((t,m[key]) for t,m in rows if key in m)
            ts=[t for t,v in valid]
            def interp(t):
                i=bisect.bisect_right(ts,t)-1
                if i<0:return valid[0][1]
                if i>=len(valid)-1:return valid[-1][1]
                a,x=valid[i];b,y=valid[i+1]
                return x+(y-x)*(t-a)/(b-a)
            points=[(begin,interp(begin))]+[(t,v) for t,v in valid if begin<t<end]+[(end,interp(end))]
            if key==forward_key:
                metrics[ep]['forward_fraction']=(points[-1][1]-points[0][1])/(end-begin)
                for (a,x),(b,y) in zip(points,points[1:]):
                    busy_bins.append((a,b,(y-x)/(b-a),ep))
            else:
                metrics[ep][key]=sum((b-a)*(x+y)/2 for (a,x),(b,y) in zip(points,points[1:]))/(end-begin)
    # Coarse coexistence check, not attribution of GPU idle milliseconds.
    backlog=intervals['host_eligible_wait_io']
    coexist_seconds=total_seconds=0
    for a,b,busy,ep in busy_bins:
        avg=sum(overlap(x,y,a,b) for x,y in backlog)/(b-a)
        total_seconds+=b-a
        if avg>=1 and busy<.95:coexist_seconds+=b-a
    stage_stats={k:occupancy(v,begin,end) for k,v in intervals.items()}
    for name in ('host_io','io_wall_equivalent','hbm_ready_before_radix_release','materialized_before_io'):
        if name in stage_stats:
            # One-second timestamp bucketing can align independent short I/O
            # intervals and invent a peak larger than the physical lane count.
            for field in ('maximum','p50','p90','nonempty_time_fraction'):
                stage_stats[name].pop(field,None)
            stage_stats[name]['subsecond_absolute_timing_not_measured']=True
    return dict(run=str(path),window=[begin,end],
                metadata_tombstones=metadata_tombstones,
                arrivals_from_retained_markers=retained_arrivals,
                arrivals_from_router_log=len(router_arrivals),
                arrivals_reconstructed_fast=recovered_arrivals,
                slow_matched=matched,slow_unmatched=len(missing),
                stage_occupancy_two_p_total=stage_stats,
                eligible_wait_io_mean_timestamp_bounds=[occupancy(v,begin,end)['mean'] for v in (queue_lower,queue_upper)],
                p_metrics=metrics,
                coarse_endpoint_time_backlog_ge1_and_forward_lt95=coexist_seconds/total_seconds,
                active_agents=occupancy(agent_intervals,begin,end),
                tool_full=distribution(list(commands.values())),
                tool_d_completion_window_cohort=distribution(mid_tools),
                tool_bracketed_count=bracketed,tool_unbracketed_count=len(missing_tools),
                tool_duration_bracket_violations=duration_inconsistent[:10],
                tool_concurrency_bracketed_lower=low/(end-begin),
                tool_concurrency_bracketed_upper=high/(end-begin),
                tool_concurrency_all_conservative_upper=(high+missing_tool_max_seconds)/(end-begin),
                model_end_to_next_arrival_or_final=occupancy(proxy_intervals,begin,end),
                notes=['All occupancy counts are system totals, not per-P averages.',
                       'Host wait-next-arrival includes tool and application/HTTP gap, not exact tool execution.',
                       'Host and load timestamp buckets have 1s precision; I/O duration uses wall_ms.',
                       'Short sub-second stage counts are coarse estimates; their instantaneous maxima can be artifacts of rounded log times.',
                       'Only complete matched lifetimes enter stage occupancy; unmatched are excluded.',
                       'Tool window cohort uses preceding D manifest creation, not exact shell start.',
                       'Tool concurrency bounds cover matched brackets only, and require duration<=bracket.',
                       'Coarse backlog/Forward coexistence is not causal idle-time attribution.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    args=parser.parse_args()
    print(json.dumps([analyze(p) for p in args.runs],indent=2))
