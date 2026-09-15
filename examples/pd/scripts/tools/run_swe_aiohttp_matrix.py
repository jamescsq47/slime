"""Seven finite SWE500 evaluations, serially, with failure-isolated reporting.

No server/harness changes. Launchers own normal cleanup; emergency signals use
exact inherited case markers plus PID start-time/pidfd checks, never GPU-wide
kill/reset. An unsafe cleanup blocks admission, not permission to overlap runs.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

PD = Path(__file__).resolve().parents[2]
PY = '/homes/siqic/anaconda3/envs/pd_mamba_baseline/bin/python'
ENGINE = Path('/homes/siqic/sglang-qwen35-integration')
FATAL = ('Scheduler hit an exception', 'CUDA out of memory', 'CUDA error:',
         'Fatal Python error', 'Agentic CUDA worker failed',
         'Unable to fence NIXL sender handle', 'TP P->D shard submission failed',
         'NIXL_ERR_REMOTE_DISCONNECT')


def cases():
    answer=[]
    for model,mode,concurrency in [('9b','colocated',256),('9b','pd',256),('9b','pd',500),
                                 ('27b','colocated',128),('27b','colocated',256),
                                 ('27b','pd',128),('27b','pd',256)]:
        tp=1 if model=='9b' else 2
        topology='2p6d' if model=='9b' else '4p4d'
        if mode=='colocated':
            filename=('run_qwen35_9b_tp1_swe_verified_500_colocated.sh' if tp==1 else
                      'run_qwen35_27b_tp2_swe500_mamba_baseline.sh')
            launcher=PD/'scripts/baseline'/filename
            column=f'Colocated c{concurrency}'
        else:
            filename=('run_qwen35_fused_swe500_2p6d.sh' if tp==1 else
                      'run_qwen35_fused_27b_tp2_swe500_4p4d.sh')
            launcher=PD/'scripts/new_method'/filename
            column=f'新方法{topology.replace("p", "P:").replace("d", "D")} c{concurrency}'
        env=dict(MAX_INFLIGHT=str(concurrency),REQUESTS='500',PD_MODEL_HTTP_TRANSPORT='aiohttp',
                 PD_COLOCATED_GPU_IDS='0,1,2,3,4,5,6,7',PD_COLOCATED_MAMBA_RATIO='0.9',
                 PD_FUSED_4P4D='0',PD_FUSED_2P2D='0',PD_SERVE_ONLY='0')
        if mode=='pd' and tp==1:
            env.update(PD_FUSED_CONGESTION_RECOMPUTE='false',PD_FUSED_CONGESTION_HIGH='32',
                       PD_FUSED_CONGESTION_LOW='32',PD_FUSED_FAST_TOOL_THRESHOLD='2',
                       PD_FUSED_P_H2D_MAX_INFLIGHT='4',PD_FUSED_P_H2D_DECOUPLED='true',
                       PD_FUSED_P_HOST_EVENT_PROGRESS='true',PD_FUSED_P_HOST_ASYNC_PREPARE='true')
        answer.append(dict(model=model,mode=mode,tp=tp,concurrency=concurrency,
                           name=f'{model}-{mode if mode=="colocated" else topology}-c{concurrency}',
                           column=column,launcher=str(launcher),env=env,
                           document=str(PD/f'runs-host/SWEBENCH_QWEN35_{model.upper()}_TP{tp}.md')))
    return answer


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def read_json(path, default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':
            live=False
            for task in Path(f'/proc/{pid}/task').iterdir():
                try:
                    if (task/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':live=True;break
                except OSError:continue
            if not live:return None
        return fields[19]
    except OSError:return None


def owned_processes(marker):
    result={}
    needle=f'PD_MATRIX_CASE_ID={marker}'.encode()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid!=os.getuid():continue
            environments=[path/'environ']
            # A zombie group leader can retain living native CUDA threads.
            environments.extend(p/'environ' for p in (path/'task').iterdir() if p.name!=path.name)
            owned=False
            for environment in environments:
                try:
                    if needle in environment.read_bytes().split(b'\0'):owned=True;break
                except OSError:continue
            if owned:
                ident=identity(int(path.name))
                if ident is not None:result[int(path.name)]=ident
        except OSError:continue
    return result


def signal_owned(pid, expected, sig):
    try:
        if hasattr(os,'pidfd_open'):fd=os.pidfd_open(pid)
        else:
            libc=ctypes.CDLL(None,use_errno=True)
            fd=libc.pidfd_open(ctypes.c_int(pid),ctypes.c_uint(0))
            if fd<0:
                error=ctypes.get_errno();raise OSError(error,os.strerror(error))
        try:
            if identity(pid)==expected:signal.pidfd_send_signal(fd,sig)
        finally:os.close(fd)
    except ProcessLookupError:pass


def clean_owned(marker,label):
    """Only after the owning launcher got its full cleanup grace period."""
    for sig,seconds in [(signal.SIGTERM,30),(signal.SIGKILL,30)]:
        remaining=owned_processes(marker)
        if not remaining:break
        for pid,ident in remaining.items():signal_owned(pid,ident,sig)
        until=time.monotonic()+seconds
        while owned_processes(marker) and time.monotonic()<until:time.sleep(1)
    ids=subprocess.check_output(['docker','ps','-aq','--filter',f'label=pd.swe.run_id={label}'],text=True,timeout=20).split()
    if ids:subprocess.run(['docker','rm','-f',*ids],check=True,timeout=60)
    # Do not remove Host arenas. Engine supervisor owns their release fence.
    remaining=owned_processes(marker)
    if remaining:raise RuntimeError(f'cleanup blocked by live owned PIDs: {remaining}')
    assert not subprocess.check_output(['docker','ps','-aq','--filter',f'label=pd.swe.run_id={label}'],text=True,timeout=20).strip()


def source_fingerprint():
    files=set(PD.glob('*.py'))
    for directory in [PD/'data',PD/'scripts',PD/'configs/experiments',ENGINE/'python/sglang',ENGINE/'validation']:
        files.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in {'.py','.sh','.yaml'})
    files.add(PD.parents[1]/'slime/utils/http_utils.py')
    # Include baseline admission/model code, not mutable caches or results.
    baseline=Path(PY).parent.parent/'lib/python3.12/site-packages/sglang/srt'
    files.update(baseline/'managers'/n for n in ['scheduler.py','schedule_policy.py','schedule_batch.py'])
    files.add(baseline/'models/qwen3_5.py')
    files.add(Path('/tmp/pd-data/swe-bench-verified/test.jsonl'))
    files.add(Path('/tmp/pd-runtime/nixl-132-pr1987-plugin/libplugin_UCX.so'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def check_sources(expected):
    now=source_fingerprint()
    changed=[p for p in expected.keys()|now.keys() if expected.get(p)!=now.get(p)]
    if changed:raise RuntimeError('sources changed since audit: '+', '.join(changed[:8]))


def resources_available():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=15)
    mem=[int(x) for x in output.split()]
    disk=os.statvfs('/tmp/pd-persist')
    available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    return len(mem)==8 and max(mem)<=1024 and disk.f_bavail*disk.f_frsize>=100*2**30 and available>=850*2**30


def progress(run):
    rows={}
    path=run/'episode_progress.jsonl'
    if path.exists():
        for line in path.open():
            try:r=json.loads(line)
            except ValueError:continue
            rows[r['instance_id']]=r
    return dict(ended=len(rows),resolved=sum(float(r.get('reward') or 0)>0 for r in rows.values()))


class Health:
    def __init__(self):self.offsets={};self.last_forward=None;self.http500=0
    def observe(self,run,now):
        chunks=[]
        for path in [*(run/'logs').glob('*.log'),run/'inference.log']:
            if not path.exists():continue
            with path.open(errors='replace') as f:
                offset=self.offsets.get(str(path),0)
                if path.stat().st_size<offset:offset=0
                f.seek(offset);data=f.read();self.offsets[str(path)]=f.tell()
            chunks.append(data)
        text='\n'.join(chunks)
        for m in FATAL:
            if m in text:return m
        if 'Decode batch' in text or 'Prefill batch' in text:
            self.last_forward=now;self.http500=0
        self.http500+=text.count("500 Internal Server Error")
        if self.last_forward is not None and now-self.last_forward>300 and self.http500>=3:
            return 'no_forward_300s_with_repeated_http500'
        return None


def update_report(case,item,result=None):
    from swe_matrix_report import update_document
    update_document(case,item,result)


def account_ownership(case,item,run):
    if case['mode']!='pd':return
    from swe_matrix_report import path_accounting
    try:
        result=path_accounting(run,case,item.get('started_at',0),time.time())
        atomic_json(run/'ownership_accounting.json',result)
        item['ownership_accounting']=dict(final=result.get('final'),conservation=result.get('ownership'))
    except Exception as exc:item['ownership_accounting_error']=repr(exc)
    item['protocol_acceptance']='pending ownership audit'


def run_matrix(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=root/'sequence_status.json'
        if status.exists():raise RuntimeError('Refusing reused queue root')
        expected=source_fingerprint();atomic_json(root/'source_hashes.json',expected)
        state=dict(state='running',started_at=time.time(),cases=[])
        stop=False;child=None;stop_at=None
        def cancel(sig,frame):
            nonlocal stop,stop_at
            stop=True
            if child is not None and child.poll() is None and stop_at is None:
                child.terminate();stop_at=time.monotonic()
        signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
        def save():state['checked_at']=time.time();atomic_json(status,state)
        save()
        for i,case in enumerate(cases(),1):
            if stop:break
            run=root/f'{root.name}-{i:02d}-{case["name"]}'
            run.mkdir(exist_ok=False)
            item=dict(**case,run=str(run),state='queued',index=i)
            state['cases'].append(item);save()
            marker=uuid.uuid4().hex
            label=f'{root.name}-{run.name}' if case['model']=='27b' and case['mode']=='pd' else run.name
            child=None;stop_at=None
            try:
                check_sources(expected)
                while not resources_available():
                    item.update(state='waiting_resources',checked_at=time.time());save()
                    if stop:break
                    time.sleep(30)
                if stop:break
                # Avoid inherited experimental toggles; preserve basic OS env.
                allowed={'PATH','HOME','USER','LOGNAME','SHELL','LANG','LC_ALL','TZ','TMPDIR',
                         'SSH_AUTH_SOCK','DOCKER_HOST','DOCKER_CONTEXT','DOCKER_CONFIG'}
                env={k:v for k,v in os.environ.items() if k in allowed}
                env.update(case['env'],RUN_DIR=str(run),RESULTS_DIR=str(root/'archives'/run.name),PD_MATRIX_CASE_ID=marker)
                assert not subprocess.check_output(['docker','ps','-aq','--filter',f'label=pd.swe.run_id={label}'],text=True,timeout=20).strip(), 'pre-existing run label'
                atomic_json(run/'queue_case.json',dict(case,env=case['env'],marker=marker,label=label))
                (run/'model_http_transport.py').write_bytes((PD/'model_http_transport.py').read_bytes())
                item.update(state='starting',started_at=time.time());update_report(case,item);save()
                check_sources(expected)
                with (run/'supervisor.log').open('x') as log:
                    child=subprocess.Popen(['bash',case['launcher']],cwd=PD,env=env,stdin=subprocess.DEVNULL,
                                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    item['launcher_pid']=child.pid
                    if stop:cancel(None,None)
                    health=Health()
                    while child.poll() is None:
                        error=health.observe(run,time.monotonic())
                        if error and stop_at is None:
                            item['health_failure']=error;child.terminate();stop_at=time.monotonic()
                        item.update(progress(run),state='stopping' if stop_at else 'running',checked_at=time.time())
                        save();atomic_json(run/'monitor.json',item)
                        # Keep an independent, read-only progress series, even
                        # if inference has not yet flushed its metric samples.
                        try:
                            from watch_swe_host_progress import live_metrics,host_state
                            observation=dict(ts=time.time(),progress=progress(run),metrics=live_metrics(run))
                            if case['mode']=='pd' and (run/'ready').exists():observation['host']=host_state(run,time.time())
                            with (run/'queue_observations.jsonl').open('a') as f:f.write(json.dumps(observation)+'\n')
                        except Exception as exc:item['observation_warning']=repr(exc)
                        if stop_at and time.monotonic()-stop_at>600:
                            clean_owned(marker,label)
                        time.sleep(30)
                    item['exit_code']=child.wait()
                    final_error=health.observe(run,time.monotonic())
                    if final_error:item['health_failure']=final_error
                clean_owned(marker,label)
                account_ownership(case,item,run)
                item.update(progress(run),ended_at=time.time())
                if stop:item['state']='cancelled'
                elif item.get('health_failure') or item['exit_code'] or item['ended']!=500:item['state']='failed'
                else:item['state']='finished'
                config=read_json(run/'config.json',{})
                if item['state']=='finished' and config.get('model_http_transport',{}).get('backend')!='aiohttp':
                    item.update(state='failed',health_failure='effective HTTP transport is not aiohttp')
                atomic_json(run/'monitor.json',item);save()
                # Each case is independently reported; a parser failure must
                # not stop the remaining experiments or fabricate missing data.
                report=None
                if item['state']=='finished':
                    try:
                        with (run/'report.log').open('w') as log:
                            code=subprocess.run([PY,str(PD/'scripts/tools/swe_matrix_report.py'),str(run)],
                                                stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
                        if code==0:report=read_json(run/'matrix_metrics.json')
                        else:item['report_error']=f'report exit {code}; see report.log'
                    except subprocess.TimeoutExpired:item['report_error']='report timed out; raw data retained'
                    if case['mode']=='pd':
                        item['protocol_acceptance']='pending ownership audit'
                        if report:item['ownership_accounting']=report.get('paths',{}).get('final')
                update_report(case,item,report)
            except Exception as exc:
                item.update(state='failed',error=repr(exc))
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=600)
                    except subprocess.TimeoutExpired:pass
                # Never advance while a previous run can still touch GPU KV.
                while child is not None:
                    try:clean_owned(marker,label);break
                    except Exception as cleanup_error:
                        item.update(state='cleanup_blocked',cleanup_error=repr(cleanup_error));save()
                        time.sleep(30)
                if child is not None:child.wait()
                account_ownership(case,item,run)
                item['state']='cancelled' if stop else 'failed'
                try:update_report(case,item)
                except Exception as report_error:item['report_error']=repr(report_error)
            finally:
                item['checked_at']=time.time();atomic_json(run/'monitor.json',item);save()
                child=None
        state.update(state='cancelled' if stop else 'finished',ended_at=time.time());save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    run_matrix(args.root.resolve())
