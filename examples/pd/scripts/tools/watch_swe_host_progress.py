"""Read-only 30s Host/Forward observation. Never signals workers or changes KV.

Uses existing metrics/logs and only current ledger entries (not history scans).
Alerts are diagnostic candidates, never grounds for eviction or cancellation.
"""
import argparse
import collections
import hashlib
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from prometheus_client.parser import text_string_to_metric_families


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except OSError:
        return None


def ledger_entries(path):
    # Current engine stores normal entries in per-snapshot event documents;
    # the small root JSON is authoritative only for relay-owned entries.
    registry = read_json(path)
    if registry is None:
        return None
    entries = {}
    for f in path.with_name(path.name + '.events').glob('*.json'):
        event = read_json(f)
        if event and event.get('entry') is not None:
            entries[event['snapshot_id']] = event['entry']
    entries.update(registry.get('entries', {}))
    return entries


def latest_metrics(path):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 1024 * 1024))
            lines = f.read().splitlines()
    except OSError:
        return {}
    result = {}
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('role') not in {'prefill', 'decode'}:
            continue
        for endpoint in row.get('endpoint_metrics', []):
            key = endpoint['endpoint']
            if key not in result:
                result[key] = dict(role=row['role'], ts=row['ts'], metrics=endpoint['metrics'])
        if len(result) == 4:
            break
    return result


def live_metrics(root):
    # The unmodified evaluator writes its 2s samples only on final flush.
    # Observe the same read-only metrics endpoints while it is running.
    config = read_json(root / 'config.json', {})
    targets = [(role, port) for role in ('prefill', 'decode')
               for port in config.get(role + '_ports', [])]

    def fetch(target):
        role, port = target
        url = f'http://127.0.0.1:{int(port)}'
        try:
            with urllib.request.urlopen(url + '/metrics', timeout=3) as response:
                content = response.read().decode()
            metrics = {}
            for family in text_string_to_metric_families(content):
                for sample in family.samples:
                    name = sample.name.replace(':', '_')
                    if name.endswith(('_bucket', '_created', '_sum', '_count')):
                        continue
                    if name == 'sglang_realtime_tokens_total':
                        name += '|mode=' + sample.labels.get('mode', 'unknown')
                    elif name in {'sglang_gpu_execution_seconds_total', 'sglang_forward_execution_seconds_total'}:
                        name += '|category=' + sample.labels.get('category', 'unknown')
                    metrics[name] = metrics.get(name, 0.) + float(sample.value)
            return url, dict(role=role, ts=time.time(), metrics=metrics)
        except (OSError, ValueError):
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(item for item in pool.map(fetch, targets) if item is not None)


def host_state(root, now):
    ready = root / 'ready'
    control = ready.resolve().parent
    entries = ledger_entries(control / 'host.json')
    if entries is None:
        return dict(available=False)
    states, phases = collections.Counter(), collections.Counter()
    pending = []
    host_active_states = {'host_reserved', 'host_writing', 'host_ready', 'h2d_loading',
                          'hbm_ready', 'retry_pending', 'aborting', 'evicting'}
    no_arrival = final_marked = 0
    for sid, entry in entries.items():
        state = entry.get('state', 'unknown')
        states[state] += 1
        for claim in entry.get('recovery_claims', {}).values():
            phases[claim.get('phase', 'no_phase')] += 1
        if state not in host_active_states:
            continue  # Terminal receipts are metadata, not live Host payload.
        digest = hashlib.sha256(sid.encode()).hexdigest() + '.json'
        if (ready / 'early-claims/finals' / digest).exists():
            final_marked += 1
            continue
        arrival = read_json(ready / 'early-claims/arrivals' / digest)
        if arrival is None:
            no_arrival += 1
            continue
        if state not in {'host_ready', 'h2d_loading'}:
            continue
        if any(c.get('phase') in {'io_inflight', 'handed'}
               for c in entry.get('recovery_claims', {}).values()):
            continue
        pending.append(dict(snapshot=sid, state=state,
                            arena_domain=entry.get('arena_domain'),
                            recovery_owner=entry.get('recovery_owner'),
                            arrival_age_s=max(0, now - arrival['arrived_at']),
                            bytes=entry.get('byte_size', 0)))
    p2d = ledger_entries(control / 'p2d-host.json') or {}
    return dict(available=True, states=dict(states), claim_phases=dict(phases),
                host_active_state_gib=sum(e.get('byte_size', 0) for e in entries.values()
                    if e.get('state') in host_active_states)/2**30,
                no_arrival_marker=no_arrival, final_marked=final_marked,
                arrived_durable_before_worker=len(pending),
                oldest_before_worker=sorted(pending, key=lambda r: -r['arrival_age_s'])[:10],
                p2d_states=dict(collections.Counter(e.get('state', 'unknown')
                    for e in p2d.values())))


def engine_intervals(current, previous, now):
    result = {}
    for endpoint, row in current.items():
        m = row['metrics']
        item = dict(role=row['role'], metrics_age_s=now-row['ts'])
        for name in ('sglang_num_running_reqs', 'sglang_num_queue_reqs',
                     'sglang_full_token_usage', 'sglang_mamba_usage',
                     'sglang_num_decode_transfer_queue_reqs', 'sglang_num_decode_prealloc_queue_reqs',
                     'sglang_num_prefill_prealloc_queue_reqs', 'sglang_num_prefill_inflight_queue_reqs'):
            item[name] = m.get(name)
        old = previous.get(endpoint)
        if old and row['ts'] > old['ts']:
            dt = row['ts'] - old['ts']
            item['interval_s'] = dt
            for category in ('forward_extend', 'forward_decode'):
                key = f'sglang_gpu_execution_seconds_total|category={category}'
                if key in m and key in old['metrics']:
                    item[category + '_fraction'] = (m[key] - old['metrics'][key])/dt
            for name in ('sglang_generation_tokens_total', 'sglang_prompt_tokens_total', 'sglang_cached_tokens_total'):
                if name in m and name in old['metrics']:
                    item[name + '_per_s'] = (m[name] - old['metrics'][name])/dt
            for mode in ('decode', 'prefill_compute', 'prefill_cache'):
                name = f'sglang_realtime_tokens_total|mode={mode}'
                if name in m and name in old['metrics']:
                    item['realtime_' + mode + '_tokens_per_s'] = (m[name] - old['metrics'][name])/dt
        result[endpoint] = item
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('--monitor-pid', type=int, required=True)
    ap.add_argument('--resume', action='store_true', help='Append after the previous observer has exited')
    args = ap.parse_args()
    root = args.run.resolve()
    owner = identity(args.monitor_pid)
    assert owner is not None
    command = Path(f'/proc/{args.monitor_pid}/cmdline').read_bytes()
    assert str(root).encode() in command and b'monitor_qwen35_27b_swe500.py' in command
    previous = {}
    with (root / 'host-progress.jsonl').open('a' if args.resume else 'x') as output:
        while True:
            now = time.time()
            monitor = read_json(root / 'monitor.json', {})
            current = latest_metrics(root / 'engine_metrics.jsonl')
            if not current:
                current = live_metrics(root)
            try:
                host = host_state(root, now)
            except (ValueError, KeyError, OSError) as exc:
                host = dict(available=False, error=str(exc))
            engines = engine_intervals(current, previous, now)
            idle_p = [url for url, e in engines.items() if e['role'] == 'prefill'
                      and e['metrics_age_s'] < 15 and e.get('interval_s', 0) >= 15
                      and e.get('forward_extend_fraction', 1) < .05]
            mem = {line.split(':')[0]: int(line.split()[1]) for line in
                   Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'MemTotal:'))}
            state = dict(schema_version=2, ts=now, monitor=monitor, host=host, engines=engines,
                         engine_counter_snapshots=current,
                         mem_available_gib=mem['MemAvailable']/1024**2,
                         local_disk_free_gib=shutil.disk_usage(root).free/2**30,
                         idle_p_with_global_host_backlog_candidate=idle_p if
                         host.get('arrived_durable_before_worker', 0) else [],
                         caveat='30s snapshots; a global queue is not proof of per-P eligible work or a deadlock')
            output.write(json.dumps(state) + '\n')
            output.flush()
            temp = root / 'host-progress.tmp'
            temp.write_text(json.dumps(state, indent=2) + '\n')
            temp.replace(root / 'host-progress-latest.json')
            previous = current
            if identity(args.monitor_pid) != owner or monitor.get('state') in {'finished', 'failed', 'cancelled'}:
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
