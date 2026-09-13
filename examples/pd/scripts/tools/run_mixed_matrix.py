#!/usr/bin/env python3
"""Sequential, fail-closed Mixed matrix runner; never changes serving code."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

PD = Path(__file__).resolve().parents[2]
HOME_ROOT = PD.parents[2]
ROOT = PD / 'runs-host/current/mixed-aligned-default-t0-20260910'
TABLE = PD / 'runs-host/MIXED_1TO1_QWEN3_8B.md'
PY = HOME_ROOT / 'anaconda3/envs/pd/bin/python'
CASES = [
    ('native_mooncake', 2, 384), ('native_mooncake', 2, 512),
    ('native_mooncake', 2, 640), ('native_mooncake', 4, 512),
    ('no_reverse', 2, 384), ('no_reverse', 2, 512),
    ('no_reverse', 2, 640), ('no_reverse', 4, 512),
    ('full', 2, 384), ('full', 2, 512), ('full', 2, 640),
]
LABEL = {'native_mooncake': '原生 Mooncake', 'no_reverse': 'No-reverse PD', 'full': '当前新方法'}
BEGIN, END = '<!-- mixed-current-start -->', '<!-- mixed-current-end -->'
FATAL_LOG = re.compile(
    r'Scheduler hit an exception|RuntimeError:|500 Internal Server Error|'
    r'Fatal Python error:|Segfault encountered|crashed with exit code|'
    r'error while attempting to bind'
)
POST_WINDOW_TEARDOWN_CONTINUATION = re.compile(
    r'NIXL_ERR_REMOTE_DISCONNECT|exit code -15|Triggering SIGQUIT'
)


def fatal_service_error(chunk, completed_after=None):
    ignored_post_window_fatal = False
    for line in chunk.splitlines():
        # Timestamps apply only to the line that contains them.  In particular,
        # a timestamped teardown message must never suppress a following,
        # un-timestamped serving failure from the same polling chunk.
        stamp = None
        match = re.search(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}', line)
        if match:
            stamp = datetime.fromisoformat(match[0]).replace(tzinfo=timezone.utc).timestamp()
            ignored_post_window_fatal = False
        if FATAL_LOG.search(line):
            # Only after confirmed workload completion: teardown can emit
            # SIGTERM/remote-disconnect errors. Never suppress an earlier or
            # un-timestamped serving failure from the same polling interval.
            if completed_after is not None and stamp is not None and stamp >= completed_after:
                ignored_post_window_fatal = True
                continue
            # A known teardown exception may be printed as an un-timestamped
            # traceback continuation.  Suppress only that narrow case; every
            # other un-timestamped fatal remains fail-closed.
            if (stamp is None and ignored_post_window_fatal and
                    POST_WINDOW_TEARDOWN_CONTINUATION.search(line)):
                continue
            return True
    return False


def name(case):
    mode, p, c = case
    return f'{mode}-{p}p{8-p}d-c{c}'


def dump(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def environment(case, directory):
    mode, p, c = case
    # Do not inherit another experiment's ablation, runtime, or model controls.
    allowed = {'PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TMPDIR',
               'LD_LIBRARY_PATH', 'HF_HOME', 'HF_HUB_CACHE', 'HF_TOKEN',
               'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY'}
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(dict(
        MODEL_PATH=str(HOME_ROOT / 'Qwen3-8B'), PD_DATA_ROOT=str(HOME_ROOT / 'data'),
        MATH_DATA=str(HOME_ROOT / 'data/dapo-math-17k/dapo-math-17k.jsonl'),
        QA_DATA=str(HOME_ROOT / 'data/browsecomp/bc_train.jsonl'),
        RUN_DIR=str(directory), TEMPERATURE='0', TOP_P='1', TOP_K='-1',
        MATH_RATIO='0.5', REQUESTS='8192', MAX_INFLIGHT=str(c),
        SCHEDULE_FILE=str(PD / 'configs/workloads/fixed_random_s2026_n8192.json'),
        WORKLOAD_CONFIG='', DISPATCH_POLICY='fixed', SEED='2026',
        WARMUP_SECONDS='300', MAX_WARMUP_SECONDS='420', MEASURE_SECONDS='1200',
        SEARCH_GPU='7', SEARCH_PORT='8730', SEARCH_START_AFTER_MODELS='true',
        SEARCH_SERVER_EMBEDDING_CACHE=str(PD / 'data/browsecomp/artifacts/search/corpus_embeddings.pkl'),
        PD_INFERENCE_RETURN_LOGPROB='false', MEM_FRACTION_STATIC='0.80',
        MAX_EXISTING_GPU_MEMORY_MB='1024', PD_IDLE_GPU_MAX_MIB='1024',
        CLOSED_LOOP='true', WARMUP_REQUESTS='0', SLIME_HTTP_READ_TIMEOUT_SECONDS='3600',
    ))
    if mode != 'full':
        env.update(PD_ENV_BIN=str(HOME_ROOT / 'anaconda3/envs/pd_baseline/bin'),
                   PYTHONPATH='', CASE_MODE=mode, UCX_LOG_LEVEL='info',
                   PREFILL_GPUS=' '.join(map(str, range(p))),
                   DECODE_GPUS=' '.join(map(str, range(p, 8))),
                   # Keep listeners outside Linux's ephemeral port range.
                   PREFILL_PORTS=' '.join(str(23100+i) for i in range(p)),
                   PREFILL_BOOTSTRAP_PORTS=' '.join(str(24100+i) for i in range(p)),
                   DECODE_PORTS=' '.join(str(23100+i) for i in range(p, 8)),
                   PREFILL_MEM_FRACTION_STATICS=' '.join(['0.80']*p),
                   DECODE_MEM_FRACTION_STATICS=' '.join(['0.80']*(7-p)+['0.60']),
                   ROUTER_PORT='23110', ROUTER_PROMETHEUS_PORT='23120',
                   MOONCAKE_MASTER_PORT='25151', MOONCAKE_METADATA_PORT='25180',
                   MOONCAKE_METRICS_PORT='25103', MOONCAKE_CLIENT_PORT='25152',
                   MOONCAKE_CLIENT_HTTP_PORT='25190')
        script = PD / 'scripts/baseline/run_pd_case.sh'
    else:
        env.update(PD_ENV_BIN=str(PY.parent),
                   PYTHONPATH=f'{HOME_ROOT}/sglang-h100-integration/python:{PD}:{PD.parents[1]}',
                   FAST_TOOL_THRESHOLD_SECONDS='1', DIRECT_WAIT_SECONDS='1',
                   SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE='true',
                   SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE='false',
                   P_H2D_MAX_INFLIGHT='2', SGLANG_AGENTIC_KV_D2H_INFLIGHT='2',
                   SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA='1',
                   SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA='1',
                   SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER='1',
                   SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB='8',
                   SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB='640')
        for direction in ('D2H', 'P_H2D', 'P2D_D2H', 'P2D_H2D'):
            env[f'SGLANG_AGENTIC_KV_{direction}_CHUNK_TOKENS'] = '4096'
        for direction in ('P2D_D2H', 'P2D_H2D'):
            env[f'SGLANG_AGENTIC_KV_{direction}_WORKERS'] = '2'
        # Native formula: 2 logical P * 2 H2D lanes => Q high/low=16/4.
        script = PD / 'scripts/new_method/run_2p6d_numa_case.sh'
    return env, ['bash', str(script)]


def gpu_guard():
    data = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    for pid in set(data.split()):
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
        # The sole co-running process explicitly allowed by the user.
        if pid == '1228785' and (b'ipykernel' in cmd or b'jupyter' in cmd):
            continue
        raise RuntimeError(f'GPU occupied by PID {pid}; not starting or killing it')


def summarize(directory):
    b = json.loads((directory / 'closed_loop_boundaries.json').read_text())
    if b['measurement_seconds'] < 1199 or b['warmup_seconds'] < 299:
        raise RuntimeError('incomplete 300+1200 window')
    a, z = b['state_at_measurement_start'], b['state_at_measurement_end']
    if z['failures'] != a['failures']:
        raise RuntimeError('measurement contains failed agents')
    rows = [json.loads(l) for l in (directory / 'engine_metrics.jsonl').open()]
    out = {'agents': z['successes']-a['successes'], 'agent_s': (z['successes']-a['successes'])/b['measurement_seconds']}
    for role in ('prefill', 'decode'):
        rr = [r for r in rows if r['role'] == role and r.get('metrics') and
              b['measurement_start_wall'] <= r['ts'] <= b['measurement_end_wall']]
        if len(rr) < 2:
            raise RuntimeError(f'missing {role} metrics')
        first, last = rr[0], rr[-1]
        if last['ts']-first['ts'] < 1190:
            raise RuntimeError('metrics do not span the measurement')
        mode = 'prefill_compute' if role == 'prefill' else 'decode'
        cat = 'forward_extend' if role == 'prefill' else 'forward_decode'
        for label, key, div in [('tps', 'sglang_realtime_tokens_total|mode='+mode, 1),
                                ('forward', 'sglang_gpu_execution_seconds_total|category='+cat, first['engine_count']/100)]:
            out[f'{role}_{label}'] = (last['metrics'][key]-first['metrics'][key])/(last['ts']-first['ts'])/div
        for label, key in [('kv','sglang_token_usage'), ('running','sglang_num_running_reqs'),
                           ('queue','sglang_num_queue_reqs'), ('transfer','sglang_num_decode_transfer_queue_reqs')]:
            out[f'{role}_{label}'] = statistics.mean(r['metrics'].get(key,0)/r['engine_count'] for r in rr)
    return out


def update_table(states):
    lines = [BEGIN, '## 当前对齐参数重跑（自动更新）', '',
             'temperature=0；固定n8192顺序；300+1200秒；普通卡0.80、搜索GPU7为0.60。',
             '按表格顺序串行；失败立即暂停。新方法使用当前完整方案，工具/Direct均1s，',
             'Q采用默认公式（2P×2 H2D lanes：high=16、low=4），不沿用TP2专用32/8调参。', '',
             '| 方法 | 配置 | 状态 | Decode token/s |', '|---|---|---|---:|']
    for case in CASES:
        st = states.get(name(case), {}); result = st.get('result', {})
        val = f"{result['decode_tps']:,.1f}" if result else '—'
        lines.append(f'| {LABEL[case[0]]} | {case[1]}P:{8-case[1]}D c{case[2]} | {st.get("status", "待运行")} | {val} |')
    lines += ['', '| 方法/配置 | Agent/s | P token/s | P Forward | P KV | D Forward | D KV | D running | D transfer |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        st = states.get(name(case), {}); r = st.get('result')
        if not r: continue
        link = (ROOT/name(case)).relative_to(PD/'runs-host')
        lines.append(f'| [{LABEL[case[0]]} {case[1]}P c{case[2]}]({link}/result.json) | {r["agent_s"]:.3f} | {r["prefill_tps"]:,.1f} | {r["prefill_forward"]:.1f}% | {r["prefill_kv"]:.1%} | {r["decode_forward"]:.1f}% | {r["decode_kv"]:.1%} | {r["decode_running"]:.1f} | {r["decode_transfer"]:.2f} |')
    lines += ['', END]
    text = TABLE.read_text(); block = '\n'.join(lines)
    if BEGIN in text:
        text = text[:text.index(BEGIN)] + block + text[text.index(END)+len(END):]
    else:
        first, rest = text.split('\n', 1); text = first+'\n\n'+block+'\n\n'+rest
    tmp = TABLE.with_suffix('.md.tmp'); tmp.write_text(text); tmp.replace(TABLE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        for case in CASES:
            env, cmd = environment(case, ROOT/name(case))
            print(name(case), cmd, json.dumps({k:env[k] for k in ('TEMPERATURE','PD_ENV_BIN','SCHEDULE_FILE','MAX_INFLIGHT')}))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (ROOT/'controller.log').open('a') as log:
            child = subprocess.Popen([str(PY), str(Path(__file__).resolve())],
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(f'controller PID={child.pid}, log={ROOT / "controller.log"}')
        return
    with (ROOT/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        statefile = ROOT/'state.json'
        states = json.loads(statefile.read_text()) if statefile.exists() else {}
        proc = None
        def stop(*_):
            if proc is not None and proc.poll() is None:
                # Signal launcher first; its existing trap owns service groups.
                proc.terminate()
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        for case in CASES:
            key = name(case)
            if states.get(key, {}).get('status') == '完成': continue
            directory = ROOT/key
            baseline_cleanup = False
            try:
                gpu_guard()
                directory.mkdir(exist_ok=False)
                env, cmd = environment(case, directory)
                dump(directory/'launch_config.json', {'command':cmd, 'env':{k:v for k,v in env.items() if k not in os.environ or os.environ[k]!=v}})
                states[key] = {'status':'运行中', 'started':time.time()}
                dump(statefile,states); update_table(states)
                offsets = {}; started = time.monotonic(); baseline_cleanup = False
                completed_after = None
                with (directory/'launcher.log').open('w') as log:
                    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    while proc.poll() is None:
                        # Every launcher writes the same closed-loop boundary file
                        # before normal service teardown.  Once the complete formal
                        # window is durable, SIGTERM/-15 messages belong to cleanup
                        # and must not turn a valid run into a service failure.
                        boundary_path = directory / 'closed_loop_boundaries.json'
                        if completed_after is None and boundary_path.exists():
                            try:
                                boundary = json.loads(boundary_path.read_text())
                                end_wall = float(boundary['measurement_end_wall'])
                                if time.time() >= end_wall:
                                    completed_after = end_wall
                            except (KeyError, ValueError, json.JSONDecodeError):
                                # The writer uses atomic replacement, but remain
                                # fail-closed if a foreign/incomplete file appears.
                                pass
                        if case[0] != 'full' and not baseline_cleanup:
                            with (directory/'launcher.log').open('rb') as output:
                                output.seek(max(0, os.fstat(output.fileno()).st_size-4096))
                                baseline_cleanup = (
                                    f'baseline PD case complete: {directory}'.encode()
                                    in output.read()
                                )
                            if baseline_cleanup:
                                boundary = json.loads((directory/'closed_loop_boundaries.json').read_text())
                                completed_after = boundary['measurement_end_wall']
                        for path in (directory/'logs').glob('*.log'):
                            with path.open(errors='replace') as stream:
                                stream.seek(offsets.get(path,0)); chunk=stream.read(); offsets[path]=stream.tell()
                            if fatal_service_error(chunk, completed_after):
                                raise RuntimeError(f'fatal service error in {path.name}')
                        if time.monotonic()-started > 7200:
                            raise RuntimeError('case exceeded 2 hour bound')
                        dump(ROOT/'heartbeat.json', {'case':key,'pid':proc.pid,'ts':time.time()})
                        time.sleep(15)
                if proc.returncode: raise RuntimeError(f'launcher exit={proc.returncode}')
                result = summarize(directory)
                dump(directory/'result.json',result)
                gpu_guard()  # Stop queue if launcher left GPU children behind.
                states[key].update(status='完成', result=result, ended=time.time())
                dump(statefile,states); update_table(states)
            except (Exception, KeyboardInterrupt) as exc:
                if proc is not None and proc.poll() is None:
                    # Never interrupt an EXIT trap already cleaning services.
                    if not baseline_cleanup:
                        proc.terminate()
                    try: proc.wait(timeout=240)
                    except subprocess.TimeoutExpired: pass  # Never advance with surviving services.
                states.setdefault(key,{}).update(status='暂停：'+str(exc), ended=time.time())
                dump(statefile,states); update_table(states)
                raise
        dump(ROOT/'heartbeat.json', {'status':'全部完成','ts':time.time()})


if __name__ == '__main__':
    main()
