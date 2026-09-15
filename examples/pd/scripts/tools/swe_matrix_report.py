"""Offline finite SWE500 report and narrow, locked updates of the result matrix.

Never reads live CUDA state. Missing measurements stay missing, and unsuccessful
runs are not relabelled as completed evaluations. TP counters count logical
groups; device timers count ranks. Stable-prefix ideal is only a counterfactual.
"""
import argparse
import base64
from collections import Counter, defaultdict
import datetime as dt
import fcntl
import json
from pathlib import Path
import re

from summarize_native_swe_colocated import interval_value
from summarize_native_swe_tables import dist


def rows(path):
    if path.exists():
        with path.open(errors='replace') as f:
            for line in f:
                try: yield json.loads(line)
                except ValueError: continue


def events(task):
    return task.get('metadata', {}).get('openenv_trajectory', {}).get('turn_events', [])


def mean(values):
    return sum(values)/len(values) if values and all(v is not None for v in values) else None


def summarize_windows(run, tasks, case):
    series = defaultdict(list)
    for row in rows(run/'engine_metrics.jsonl'):
        if case['mode']=='colocated' and row['role']!='prefill': continue
        for ep in row.get('endpoint_metrics', []):
            series[(row['role'], ep['endpoint'])].append((row['ts'], dict(ep['metrics'])))
    for samples in series.values():
        samples.sort(key=lambda x:x[0])
        for pool in ['kv','mamba']:
            names=[f'sglang_{pool}_{s}_tokens' for s in ['used','evictable','available']]
            caps=Counter(sum(m[k] for k in names) for _,m in samples if all(k in m for k in names))
            cap=caps.most_common(1)[0][0] if caps else 0
            for _,m in samples:
                if cap and all(k in m for k in names):
                    for s in ['used','evictable','available']:m[f'{pool}_{s}']=m[f'sglang_{pool}_{s}_tokens']/cap
                    m[f'{pool}_resident']=1-m[f'{pool}_available']
                else:
                    key='sglang_full_token_usage' if pool=='kv' else 'sglang_mamba_usage'
                    if key in m:m[f'{pool}_used']=m[key]
    begin=min(t['started_ts'] for t in tasks);end=max(t['finished_ts'] for t in tasks)
    windows={}
    for name,a,b in [('full',begin,end),('middle',begin+300,begin+1500)]:
        if b>end:windows[name]=None;continue
        w=dict(seconds=b-a,ended=sum(a<=t['finished_ts']<=b for t in tasks),
               completed=sum(a<=t['finished_ts']<=b and t['status']=='completed' for t in tasks),
               active_agents=sum(max(0,min(b,t['finished_ts'])-max(a,t['started_ts'])) for t in tasks)/(b-a),roles={})
        for role in ['prefill','decode']:
            selected='prefill' if case['mode']=='colocated' else role
            groups=[ss for (r,_),ss in series.items() if r==selected]
            expected=(8//case['tp'] if case['mode']=='colocated' else
                      (2 if role=='prefill' or case['tp']==2 else 6))
            m={'groups':len(groups)}
            m['expected_groups']=expected
            m['coverage']=[dict(first=ss[0][0],last=ss[-1][0],covers_begin=ss[0][0]<=a,covers_end=ss[-1][0]>=b) for ss in groups if ss]
            for label,key,counter in [
                ('p_tps','sglang_realtime_tokens_total|mode=prefill_compute',True),
                ('d_tps','sglang_realtime_tokens_total|mode=decode',True),
                ('cached_tps','sglang_realtime_tokens_total|mode=prefill_cache',True),
                ('running','sglang_num_running_reqs',False),('waiting','sglang_num_queue_reqs',False),
                ('prealloc','sglang_num_decode_prealloc_queue_reqs',False),
                ('transfer','sglang_num_decode_transfer_queue_reqs',False),
            ]+[(f'{p}_{s}',f'{p}_{s}',False) for p in ['kv','mamba'] for s in ['used','resident','evictable','available']]:
                values=[interval_value(ss,key,a,b,counter) for ss in groups]
                avg=mean(values) if len(groups)==expected else None
                m[label]=avg*len(groups) if counter and avg is not None else avg
            for label,category in [('p_forward','extend'),('d_forward','decode')]:
                values=[]
                for ss in groups:
                    native=f'sglang_forward_execution_seconds_total|category={category}'
                    custom=f'sglang_gpu_execution_seconds_total|category=forward_{category}'
                    key=native if any(native in x for _,x in ss) else custom
                    v=interval_value(ss,key,a,b,True)
                    values.append(v/case['tp'] if v is not None else None)
                m[label]=mean(values) if len(groups)==expected else None
            m['gap']=max((t2-t1 for ss in groups for (t1,_),(t2,_) in zip(ss,ss[1:]) if t1<b and t2>a),default=None)
            w['roles'][role]=m
        windows[name]=w
    return begin,end,windows


def raw_accounting(run,tasks,case):
    per=defaultdict(Counter);seen=set();unknown=0
    owners={t['metadata'].get('agentic_request_id'):t['metadata']['instance_id'] for t in tasks}
    users={t['metadata'].get('openenv_trajectory',{}).get('instruction','').replace('\r\n','\n').strip():t['metadata']['instance_id'] for t in tasks}
    directories=list(run.glob('raw-[0-9]*')) if case['mode']=='colocated' else list((run/'raw').glob('prefill-*'))
    for directory in directories:
        for path in sorted(directory.glob('*.log*')):
            with path.open(errors='replace') as f:
                for line in f:
                    if '"event": "request.finished"' not in line:continue
                    r=json.loads(line[line.index('{'):]);rid=r['rid']
                    if rid in seen or rid.startswith('HEALTH_CHECK'):continue
                    m=r['out']['meta_info']
                    if 'prompt_tokens' not in m or 'request_received_ts' not in m:continue
                    if case['mode']=='pd':
                        extra=r['obj'].get('extra_key') or ''
                        if not extra.startswith('agentic-v1e:'):continue
                        p=extra.split(':')[2];payload=json.loads(base64.urlsafe_b64decode(p+'='*(-len(p)%4)))
                        iid=owners.get(payload['agentic_request_id'])
                    else:
                        text=r['obj'].get('text') or ''
                        if '<|im_start|>user\n' not in text:continue
                        user=text.split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0].replace('\r\n','\n').strip()
                        iid=users.get(user)
                        if iid is None:
                            candidates=[v for u,v in users.items() if u[:128]==user[:128]]
                            iid=candidates[0] if len(candidates)==1 else None
                    if iid is None:unknown+=1;continue
                    seen.add(rid);c=per[iid];c['calls']+=1;c['prompt']+=m['prompt_tokens'];c['cached']+=m.get('cached_tokens',0)
                    c['retractions']+=m.get('total_retractions',m.get('num_retractions',0))
    mismatches=[]
    for t in tasks:
        c=per[t['metadata']['instance_id']]
        if c['calls']!=len(events(t)) or c['prompt']!=sum(e.get('prompt_tokens',0) for e in events(t)):
            mismatches.append(t['metadata']['instance_id'])
    totals={k:sum(c[k] for c in per.values()) for k in ['prompt','cached','retractions','calls']}
    totals['actual']=totals['prompt']-totals['cached']
    return dict(totals=totals,mismatches=mismatches,unmatched=unknown,per_task=dict(per))


def path_accounting(run,case,begin,end):
    if case['mode']=='colocated':return {}
    names={'direct':'early_direct_bind' if case['tp']==1 else 'early_direct_group_complete',
           'slow':'shared_host_h2d_release' if case['tp']==1 else 'shared_host_group_commit_release'}
    stamps=defaultdict(dict);counts=Counter();ownership=defaultdict(set)
    ownership_events={'host_staging_offer','shared_host_d2h_complete','d_release_after_p_host',
                      'p2d_host_d2h_queued','p2d_host_d2h_complete','p2d_host_prefill_release'}
    for path in (run/'logs').glob('*.log*'):
        for line in path.open(errors='replace'):
            match=re.search(r'AgenticKV (\w+)',line)
            if not match:continue
            ev=match[1];counts[ev]+=1
            sid=re.search(r'\bsnapshot=([^\s]+)',line)
            if ev in ownership_events and sid:ownership[ev].add(sid[1])
            if ev not in names.values():continue
            if not sid:continue
            try:ts=dt.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()+.5
            except ValueError:continue
            stamps[ev][sid[1]]=max(ts,stamps[ev].get(sid[1],ts))
    windows={}
    for label,a,b in [('full',begin,end),('middle',begin+300,begin+1500)]:
        windows[label]={kind:sum(a<=t<=b for t in stamps[ev].values()) for kind,ev in names.items()}
    from watch_swe_host_progress import ledger_entries
    final={}
    for name in ['host','p2d-host']:
        entries=ledger_entries(run/'control-final'/f'{name}.json')
        active={'host_reserved','host_writing','host_ready','h2d_loading','hbm_ready','retry_pending','aborting','evicting'}
        final[name]=None if entries is None else dict(states=dict(Counter(e.get('state','unknown') for e in entries.values())),
            outstanding=[dict(snapshot=sid,state=e.get('state'),byte_size=e.get('byte_size'),recovery_claims=e.get('recovery_claims')) for sid,e in entries.items() if e.get('state') in active])
    queued=ownership['host_staging_offer'];durable=ownership['shared_host_d2h_complete'];released=ownership['d_release_after_p_host']
    return dict(windows=windows,event_counts=dict(counts),final=final,
                ownership=dict(queued=len(queued),durable=len(durable),source_release=len(released),
                    queued_not_durable=sorted(queued-durable),durable_not_released=sorted(durable-released),
                    unqueued_durable=sorted(durable-queued),release_without_durable=sorted(released-durable),
                    note='唯一generation日志集合，另保留逐rank事件次数；TP原子性和取消例外仍需人工复核'),
                p2d_ownership={k:sorted(ownership[ev]) for k,ev in [('queued','p2d_host_d2h_queued'),('durable','p2d_host_d2h_complete'),('source_release','p2d_host_prefill_release')]},
                overlap=len(set(stamps[names['direct']])&set(stamps[names['slow']])))


def summarize(run,case):
    tasks=list(rows(run/'requests.jsonl'))
    assert len(tasks)==len({t['metadata']['instance_id'] for t in tasks})==500
    begin,end,windows=summarize_windows(run,tasks,case)
    ev=[e for t in tasks for e in events(t)];shell=[e['command_seconds'] for e in ev if e.get('command') and e.get('command_seconds') is not None]
    raw=raw_accounting(run,tasks,case)
    errors=Counter()
    for path in [*(run/'logs').glob('*.log*'),run/'inference.log']:
        if not path.exists():continue
        for line in path.open(errors='replace'):
            for key,needle in [('http_retry','Retrying'),('oom','CUDA out of memory'),('scheduler','Scheduler hit an exception')]:
                errors[key]+=line.count(needle)
    ends=sorted(t['finished_ts'] for t in tasks);normal=sorted(t['finished_ts'] for t in tasks if t['status']=='completed')
    ideal=aligned=0;valid=case['model']=='9b'
    for t in tasks:
        previous=0
        for e in events(t):
            stable=max(previous-2,0);p=e.get('prompt_tokens',0)
            valid=valid and not e.get('context_compacted') and p>=stable
            ideal+=p-stable;aligned+=p-stable//64*64;previous=p
    return dict(run=str(run),case=case,begin=begin,end=end,windows=windows,raw=raw,error_markers=dict(errors),
        ended=500,completed=len(normal),resolved=sum(bool(t['metadata'].get('swe_bench_verifier',{}).get('resolved')) for t in tasks),
        milestones={str(n):ends[n-1]-begin for n in [250,450,500]},normal450=normal[449]-begin if len(normal)>=450 else None,
        latency=dist([t['finished_ts']-t['started_ts'] for t in tasks]),turns=dist([len(events(t)) for t in tasks]),
        decode=sum(t.get('response_tokens',0) for t in tasks),prompt=sum(e.get('prompt_tokens',0) for e in ev),
        initial=dist([events(t)[0]['prompt_tokens'] for t in tasks if events(t)]),
        prompt_dist=dist([e['prompt_tokens'] for e in ev]),decode_dist=dist([e['output_tokens'] for e in ev]),
        ideal=ideal if valid else None,aligned=aligned if valid else None,
        ideal_note='仅9B未压缩稳定Prompt的L-2/page64反事实；非精确token-LCP物理下界，超额未全部归因',
        observations=sum(e.get('observation_tokens',0) for e in ev),shell=dist(shell),shell_max=max(shell,default=None),
        shell_task=dist([sum(e.get('command_seconds',0) for e in events(t) if e.get('command')) for t in tasks]),
        tool_bins=[sum(f(s) for s in shell) for f in [lambda s:s<=1,lambda s:1<s<=2,lambda s:2<s<=10,lambda s:s>10]],
        verifier=dist([t['metadata']['swe_bench_verifier']['duration_seconds'] for t in tasks if t['metadata'].get('swe_bench_verifier',{}).get('duration_seconds') is not None]),
        verifier_max=max((t['metadata'].get('swe_bench_verifier',{}).get('duration_seconds',0) or 0 for t in tasks),default=None),
        docker=dist([t['metadata']['sandbox_metrics']['container_start_seconds'] for t in tasks if 'container_start_seconds' in t['metadata'].get('sandbox_metrics',{})]),
        local_images=sum(t['metadata'].get('sandbox_metrics',{}).get('image_present_before_start') is True for t in tasks),
        stops=dict(Counter(t['metadata'].get('stop_reason','unknown') for t in tasks)),
        stop_passed=dict(Counter(t['metadata'].get('stop_reason','unknown') for t in tasks if t['metadata'].get('swe_bench_verifier',{}).get('resolved'))),
        paths=path_accounting(run,case,begin,end))


def fmt(v,unit='',digits=2):
    return '未记录' if v is None else f'{v:,.{digits}f}{unit}'


def ds(d,keys=('mean','p50','p90')):
    return ' / '.join(fmt(d.get(k) if d else None) for k in keys)


def value_for(section,label,r):
    case=r['case'];pd=case['mode']=='pd';w=r['windows'].get('middle' if section==3 else 'full')
    wall=r['end']-r['begin'];actual=r['raw']['totals']['actual'] if not r['raw']['mismatches'] and not r['raw']['unmatched'] else None
    def total(v):return '未可靠核算' if v is None else f'{v/500:,.3f} / {v:,} tokens'
    if section in (2,3):
        if w is None:return '窗口不足1500秒'
        p=w['roles']['prefill'];d=w['roles']['decode'];sec=w['seconds']
        if '收尾 /' in label:return f"{w['ended']} / {w['completed']} / {w['ended']-w['completed']}"
        if '通过题数' in label:return f"{r['resolved']} / 500 = {r['resolved']/5:.1f}%"
        if '全量完成时间' in label:return fmt(wall,' s')
        if label.startswith('T250 /'):return fmt(r['milestones']['250'])+' / '+fmt(r['milestones']['450'],' s')
        if label.startswith('T250'):return fmt(r['milestones']['250'],' s')
        if label.startswith('T450'):return fmt(r['normal450'] if '仅正常' in label else r['milestones']['450'],' s')
        if '单题完成耗时' in label:return ds(r['latency'])+' s'
        if '收尾速率' in label:return fmt(w['ended']/sec,' agent/s',4)
        if '窗口时长' in label:return fmt(sec,' s')
        if '平均活跃agent' in label:return fmt(w['active_agents'])
        if label.startswith('Prefill实际计算'):return fmt(p['p_tps'],' token/s')
        if label.startswith('Decode吞吐'):
            v=d['d_tps'];n=d['groups'];tp=case['tp']
            if '每TP组' in label:return ' / '.join(fmt(x) for x in [v,v/n if v is not None and n else None,v/n/tp if v is not None and n else None])+' token/s'
            return fmt(v/(n*tp) if '/ 卡' in label and v is not None and n else v,' token/s')
        if '完整轨迹输出' in label:return fmt(r['decode']/wall,' token/s')
        if '成功模型调用未命中' in label:return fmt(actual/wall if actual is not None else None,' token/s')
        if 'Forward时间' in label:
            v=p['p_forward'] if label.startswith(('P ','Prefill')) else d['d_forward']
            return fmt(v*sec if v is not None else None,' s')+' / '+fmt(v*100 if v is not None else None,'%')
        if '非Forward' in label:
            a=p['p_forward'];b=d['d_forward']
            if a is None or b is None:return '未记录'
            return f'P {100*(1-a):.2f}% / D {100*(1-b):.2f}%' if pd else f'{100*(1-a-b):.2f}%'
        if 'running' in label.lower():return (f"P {fmt(p['running'])} / D {fmt(d['running'])}" if pd else fmt(d['running']))+(' / TP组' if case['tp']>1 else '')
        if label.startswith('Waiting'):return f"P {fmt(p['waiting'])} / D "+' / '.join(fmt(d[k]) for k in ['waiting','prealloc','transfer'])
        if '原生计算等待' in label:return f"P {fmt(p['waiting'])} / D {fmt(d['waiting'])}" if pd else fmt(d['waiting'])
        if label.startswith(('Attention KV','Mamba')):
            pool='kv' if label.startswith('Attention') else 'mamba';keys=['used','resident'] if '总驻留' in label else ['evictable','available']
            def pair(role):return ' / '.join(fmt(role[f'{pool}_{k}']*100 if role[f'{pool}_{k}'] is not None else None,'%') for k in keys)
            return 'P '+pair(p)+'；D '+pair(d) if pd else pair(d)
        if 'Prefix' in label:
            if section==2:
                raw=r['raw']['totals'];v=raw['cached']/raw['prompt'] if actual is not None and raw['prompt'] else None
            else:v=p['cached_tps']/(p['cached_tps']+p['p_tps']) if p['cached_tps'] is not None and p['p_tps'] is not None and p['cached_tps']+p['p_tps'] else None
            return fmt(v*100 if v is not None else None,'%')
        if 'retraction' in label.lower():
            value=str(r['raw']['totals']['retractions'])+('（原始Prefill调用口径）' if pd else '')
            return value+' / 日志标记 '+str(r['error_markers']) if 'HTTP' in label else value
        if label.startswith('HTTP重试'):return str(r['error_markers'])+'（日志标记，非错误请求去重数）'
        if '最大实际采样' in label:return fmt(max((v for v in [p['gap'],d['gap']] if v is not None),default=None),' s')
    if section==4:
        if '轮数' in label or '轮次 /' in label:return ds(r['turns'])+'（均值/P50/P90）'
        if label.startswith(('Decode输出','总Decode')):return total(r['decode'])
        if label.startswith(('实际Prefill','成功模型调用实际')):return total(actual)
        if label.startswith('GPU实际Prefill'):return total(r['windows']['full']['roles']['prefill']['p_tps']*wall) if r['windows']['full']['roles']['prefill']['p_tps'] is not None else '未记录'
        if label.startswith(('理想必要','理想增量')):return total(r['ideal'])+'（反事实参考）'
        if label.startswith('理想全命中'):return total(r['aligned'])+'（反事实参考）'
        if label.lower().startswith('page'):return total(r['aligned']-r['ideal'] if r['aligned'] is not None else None)
        if label.startswith('超出'):
            base=r['aligned'] if 'page64' in label else r['ideal']
            return total(actual-base if actual is not None and base is not None else None)
        if label.startswith('额外Prefill可解释'):return r['ideal_note']
        if label.startswith('累计工具'):return total(r['observations'])
        if label.startswith('各轮完整Prompt'):return total(r['prompt'])
        if label.startswith('首轮Prompt'):return ds(r['initial'])+' tokens'
        if label.startswith('所有轮次Prompt'):return ds(r['prompt_dist'])+' tokens'
        if label.startswith('单轮Decode'):return ds(r['decode_dist'])+' tokens'
        if label.startswith('终止原因'):return '；'.join(f'{k} {v}/{r["stop_passed"].get(k,0)}' for k,v in r['stops'].items())
    if section==5:
        if label.startswith('Shell调用'):return f"{r['shell']['count']} / {r['shell']['count']/500:.3f}" if r['shell'] else '0 / 0'
        if label.startswith('单次Shell时间：'):return ds(r['shell'],('mean','p50','p90','p99'))+' s'
        if label.startswith('单次Shell最长'):return fmt(r['shell_max'],' s')
        if label.startswith('每题累计Shell'):return ds(r['shell_task'])+' s'
        if label.startswith('Verifier时间'):return ds(r['verifier'])+(' / '+fmt(r['verifier_max']) if '最长' in label else '')+' s'
        if label.startswith('Verifier最长'):return fmt(r['verifier_max'],' s')
        if label.startswith('Docker启动'):return ds(r['docker'])+' s'
        if label.startswith('已有本地'):return f"{r['local_images']} / 500"
        for i,key in enumerate(['≤1秒','>1秒且≤2秒','>2秒且≤10秒','>10秒']):
            if key in label:
                n=r['tool_bins'][i];den=r['shell']['count'] if r['shell'] else 0
                return f'{n} / {n/den*100:.2f}%' if den else '0 / 不适用'
    if section==6:
        if not pd:return '不适用（Colocated）'
        if label.startswith('D→P'):
            out=[]
            for name,p in r['paths']['windows'].items():
                den=p['direct']+p['slow']
                if ' / Slow' in label:val=f"Direct {p['direct']} / Slow {p['slow']}"
                else:
                    key='direct' if 'Direct' in label else 'slow';val=str(p[key])+f' / {100*p[key]/den:.2f}%' if den else '0 / 未形成成功恢复'
                out.append(name+': '+val)
            return '；'.join(out)+'（成功恢复口径）'
        if label.startswith('未完成snapshot'):
            o=r['paths']['ownership']
            out={k:None if v is None else len(v['outstanding']) for k,v in r['paths']['final'].items()}
            return f"D→Host queued/durable/release {o['queued']}/{o['durable']}/{o['source_release']}；Host残留 {out}；TP/取消待复核"
        if '改善' in label:return '已记录，需结合逐阶段轨迹分析；不由总吞吐推断无阻塞'
    return '未可靠核算（保留原始记录待复核）'


def update_document(case,item,result=None):
    path=Path(case['document']);status={'starting':'启动中','running':'运行中','finished':'已完成','failed':'失败','cancelled':'已取消'}.get(item['state'],item['state'])
    if item.get('report_error'):status+='，统计失败待复核'
    if case['mode']=='pd' and item['state']=='finished':status+='，协议守恒待复核'
    if item.get('ended') is not None:status+=f"（收尾{item['ended']}/500）"
    with path.with_suffix('.md.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        text=path.read_text();lines=text.splitlines();section=0;column=None;out=[]
        for line in lines:
            if line.startswith('## '):
                section=int(line.split()[1].rstrip('.'));column=None
            if line.startswith('|'):
                cells=[c.strip() for c in line.strip().strip('|').split('|')]
                if cells[0] in ['指标','项目','单次模型调用长度（平均 / P50 / P90）','单次Shell时间区间（调用数 / 占比）','完整agent轨迹（每题 / 合计）'] and case['column'] in cells:
                    column=cells.index(case['column'])
                elif case['column'] in cells and section>=2:column=cells.index(case['column'])
                elif section==1 and cells[0]==case['column']:
                    cells[-1]=status;line='| '+' | '.join(cells)+' |'
                elif section>=2 and column is not None and len(cells)>column and not cells[0].startswith('---'):
                    val=value_for(section,cells[0],result) if result else status
                    if section==6 and case['mode']=='colocated':val='不适用（Colocated）'
                    cells[column]=val.replace('|',' / ').replace('\n',' ');line='| '+' | '.join(cells)+' |'
            out.append(line)
        marker=f'<!-- matrix:{case["name"]} -->'
        out=[line for line in out if marker not in line]
        out.append(f'\n{marker} {case["column"]}：{status}；原始目录 `{item["run"]}`；统计 `matrix_metrics.json`。')
        tmp=path.with_suffix('.md.tmp');tmp.write_text('\n'.join(out)+'\n');tmp.replace(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    case=json.loads((args.run/'queue_case.json').read_text());result=summarize(args.run,case)
    (args.run/'matrix_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
