"""Evaluate launcher configuration only; never launch Docker or CUDA workers."""
import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/new_method/run_qwen35_fused_swe500_2p6d.sh"


def config(**overrides):
    script = LAUNCHER.read_text().split('mkdir -p "${RUN_DIR}/logs"', 1)[0]
    script = script.replace('${BASH_SOURCE[0]}', str(LAUNCHER))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PD_FUSED_', 'SGLANG_'))}
    env.update(overrides, RUN_DIR='/tmp/pd-topology-config-only')
    return subprocess.run(['bash', '-c', script + '\nenv -0'], env=env,
                          capture_output=True, timeout=10)


@pytest.mark.parametrize('flag,p,d,host,p2d', [
    ({}, '0 4', '1 2 3 5 6 7', 128, 32),
    ({'PD_FUSED_2P2D': '1'}, '0 6', '1 7', 128, 32),
    ({'PD_FUSED_4P4D': '1'}, '0 1 4 5', '2 3 6 7', 64, 16),
])
def test_topology_and_capacity(flag, p, d, host, p2d):
    result = config(**flag)
    assert result.returncode == 0, result.stderr.decode()
    env = dict(item.decode().split('=', 1) for item in result.stdout.split(b'\0') if item)
    assert env['PREFILL_GPUS'] == p
    assert env['DECODE_GPUS'] == d
    assert set(p.split()).isdisjoint(d.split())
    for role, gpus in [('PREFILL', p), ('DECODE', d)]:
        assert env[role + '_GPU_GROUPS'].split(';') == gpus.split()
        assert len(env[role + '_PORTS'].split()) == len(gpus.split())
        assert env[role + '_TP_SIZE'] == '1'
    assert len(env['BOOTSTRAP_PORTS'].split()) == len(p.split())
    assert env['MEM_FRACTION_STATIC'] == '0.80'
    assert env['DECODE_MEM_FRACTION_STATICS'].split() == ['0.80'] * len(d.split())
    assert int(env['SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB']) == host
    assert int(env['SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB']) == p2d
    assert len(p.split()) * (host + p2d) == 320
    ports = [int(port) for name in ['PREFILL_PORTS', 'DECODE_PORTS', 'BOOTSTRAP_PORTS']
             for port in env[name].split()]
    ports += [int(env['PD_PREFILL_NCCL_PORT_BASE']) + i for i in range(len(p.split()))]
    ports += [int(env['PD_DECODE_NCCL_PORT_BASE']) + i for i in range(len(d.split()))]
    ports += [23750, 23751] + list(range(23900, 23900 + len(d.split())))
    assert len(ports) == len(set(ports))
    assert env['SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB'] == '320'
    assert env['SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER'] == '1'
    assert env['SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH'] == 'false'


def test_conflicting_topologies_rejected():
    assert config(PD_FUSED_4P4D='1', PD_FUSED_2P2D='1').returncode == 2
