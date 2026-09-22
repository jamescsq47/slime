import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import MagicMock, patch

import multinode as m
import qwen35_multinode as q


class QwenMultinodeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = q.config(Path('/homes/siqic/dualpd/slime/runs/dualpd/unit-qwen'))

    def test_valid_two_tp8_groups_and_hybrid_settings(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'
            path.write_text(json.dumps(self.cfg))
            cfg = m.load_config(path)
        workers = m.plan(cfg)['workers']
        for p in workers:
            argv = p['command']
            self.assertEqual(argv[argv.index('--tp-size') + 1], '8')
            self.assertEqual(argv[argv.index('--ep-size') + 1], '1')
            self.assertEqual(argv[argv.index('--mem-fraction-static') + 1], '0.8')
            self.assertEqual(p['environment']['SGLANG_AGENTIC_MULTINODE_QWEN35_HYBRID'], '1')
            self.assertEqual(p['environment']['UCX_NET_DEVICES'], 'mlx5_1:1')
            self.assertNotIn('--enable-hierarchical-cache', argv)
            self.assertEqual(p['environment']['SGLANG_AGENTIC_KV_TP_HOST_ASYNC_PREPARE'], 'true')
        self.assertEqual(workers[0]['host_capacity_gib_per_direction']['p2d_source'], 64)
        self.assertEqual(workers[1]['host_capacity_gib_per_direction']['d2p_source'], 128)
        for worker in workers:
            env = worker['environment']
            self.assertEqual(env['SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD'], '1')
            self.assertEqual(env['SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT'], '1')
            self.assertEqual(env['SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE'], 'false')
            self.assertEqual(env['SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE'], 'false')
            self.assertEqual(env['SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE'], 'false')

    def test_workload_matches_reference_and_keeps_environment(self):
        p = m.workload_plan(self.cfg)
        argv = p['command']
        for key, value in [('--temperature', '0.6'), ('--top-p', '0.95'), ('--top-k', '20'),
                           ('--max-inflight', '64'), ('--requests', '500'),
                           ('--max-response-length', '81920')]:
            self.assertEqual(argv[argv.index(key) + 1], value)
        self.assertIn('structured_tool', argv[argv.index('--workload-config') + 1])
        env = m.runtime_env(p)
        self.assertEqual(env['PD_SWE_RUN_ID'], 'unit-qwen')
        self.assertEqual(env['PD_MODEL_HTTP_TRANSPORT'], 'aiohttp')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertTrue(env['PATH'].startswith(str(Path(q.PYTHON).parent) + ':'))
        self.assertEqual(env['CUDA_HOME'], '/homes/siqic/cuda-12.8')

    def test_model_triton_cache_is_local_and_persists_across_runs(self):
        workers = m.plan(self.cfg)['workers']
        for worker in workers:
            expected = str(Path(self.cfg['local_root']) / 'compiler-cache'
                           / worker['engine_id'] / 'triton')
            self.assertEqual(worker['environment']['TRITON_CACHE_DIR'], expected)
            with patch.dict(m.os.environ, {'TRITON_CACHE_DIR': '/nfs/old-cache'}):
                self.assertEqual(m.runtime_env(worker)['TRITON_CACHE_DIR'], expected)
        other = dict(self.cfg, run_id='another-run')
        self.assertEqual(workers[0]['environment']['TRITON_CACHE_DIR'],
                         m.plan(other)['workers'][0]['environment']['TRITON_CACHE_DIR'])
        legacy = dict(self.cfg, local_triton_cache=False)
        self.assertNotIn('TRITON_CACHE_DIR', m.plan(legacy)['workers'][0]['environment'])

    def test_direct_only_disables_only_reverse_host_staging(self):
        original = m.plan(self.cfg)
        self.cfg['d2p_host_staging'] = False
        ablation = m.plan(self.cfg)
        for before, after in zip(original['workers'], ablation['workers']):
            self.assertEqual(before['command'], after['command'])
            expected = dict(before['environment'])
            expected['SGLANG_AGENTIC_KV_HOST_STAGING'] = 'false'
            self.assertEqual(after['environment'], expected)
            self.assertEqual(after['environment']['SGLANG_AGENTIC_KV_P2D_HOST_STAGING'], 'true')
            self.assertEqual(after['host_capacity_gib_per_direction']['d2p_source'], 0)
        self.assertEqual(m.workload_plan(self.cfg)['command'],
                         m.workload_plan(dict(self.cfg, d2p_host_staging=True))['command'])
        self.assertEqual(ablation['workers'][0]['host_capacity_gib_per_direction']['p2d_source'], 64)

    def test_reverse_host_flag_must_be_boolean(self):
        self.cfg['d2p_host_staging'] = 'false'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'
            path.write_text(json.dumps(self.cfg))
            with self.assertRaisesRegex(ValueError, 'd2p_host_staging'):
                m.load_config(path)

    def test_strict_direct_wait_is_opt_in_and_preserves_p2d(self):
        self.assertEqual(m.plan(self.cfg)['workers'][0]['environment']
                         ['SGLANG_AGENTIC_KV_DIRECT_WAIT_ONLY'], 'false')
        self.cfg.update(d2p_host_staging=False, d2p_direct_wait_only=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'
            path.write_text(json.dumps(self.cfg))
            m.load_config(path)
            for worker in m.plan(self.cfg)['workers']:
                env = worker['environment']
                self.assertEqual(env['SGLANG_AGENTIC_KV_DIRECT_WAIT_ONLY'], 'true')
                self.assertEqual(env['SGLANG_AGENTIC_KV_HOST_STAGING'], 'false')
                self.assertEqual(env['SGLANG_AGENTIC_KV_P2D_HOST_STAGING'], 'true')
            self.cfg['d2p_host_staging'] = True
            path.write_text(json.dumps(self.cfg))
            with self.assertRaisesRegex(ValueError, 'd2p_direct_wait_only'):
                m.load_config(path)

    def test_listener_ports_avoid_ephemeral_range_and_match_workload(self):
        ports = [self.cfg['router'][key] for key in ('port', 'metrics_port')]
        for node in self.cfg['nodes']:
            ports.extend(node[key] for key in ('port', 'bootstrap_port', 'reverse_bootstrap_port'))
        self.assertTrue(all(not 32768 <= port <= 60999 for port in ports))
        argv = self.cfg['workload_command']
        for flag, port in [('--router-port', self.cfg['router']['port']),
                           ('--prefill-port', self.cfg['nodes'][0]['port']),
                           ('--decode-port', self.cfg['nodes'][1]['port'])]:
            self.assertEqual(argv[argv.index(flag) + 1], str(port))

    def test_host_probe_only_limits_decode_capacity(self):
        for worker in m.plan(self.cfg)['workers']:
            self.assertNotIn('--max-total-tokens', worker['command'])
        self.cfg['p2d_host_probe'] = True
        p, d = m.plan(self.cfg)['workers']
        self.assertNotIn('--max-total-tokens', p['command'])
        self.assertEqual(d['command'][d['command'].index('--max-total-tokens') + 1], '12288')

    def test_concurrency_override_only_changes_workload_limit(self):
        larger = q.config(Path('/homes/siqic/dualpd/slime/runs/dualpd/unit-qwen'), concurrency=128)
        expected = list(self.cfg['workload_command'])
        expected[expected.index('--max-inflight') + 1] = '128'
        self.assertEqual(larger['workload_command'], expected)
        larger['workload_command'] = self.cfg['workload_command']
        # Each newly constructed run receives a fresh broker credential;
        # it is not a workload/engine parameter affected by concurrency.
        self.assertGreaterEqual(len(larger['control']['token']), 32)
        larger['control']['token'] = self.cfg['control']['token']
        self.assertEqual(larger, self.cfg)
        with self.assertRaises(ValueError):
            q.config(Path('/tmp/invalid-run'), concurrency=0)

    def test_workload_cannot_override_lifecycle(self):
        self.cfg['workload_environment']['SGLANG_AGENTIC_KV_LIFECYCLE'] = 'false'
        with self.assertRaises(ValueError):
            m.workload_plan(self.cfg)

    def test_ssh_arguments_are_quoted_not_shell_interpolated(self):
        argv = q.remote('a11', ['python', '-c', 'print("a b")'])
        self.assertEqual(argv[-1], 'python -c \'print("a b")\'')

    def test_cleanup_continues_after_remote_timeout(self):
        effects = ['stopped', 'not started', 'stopped', subprocess.TimeoutExpired('ssh', 40), 'stopped P']
        with patch.object(q, 'call', side_effect=effects) as call:
            result = q.stop(Path('/shared/run/config.json'))
        self.assertEqual(call.call_count, 5)
        self.assertEqual(result[-1], 'stopped P')

    def router_wait_fixture(self):
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        children = {name: MagicMock() for name in ("prefill", "decode", "router")}
        for child in children.values():
            child.poll.return_value = None
        response = MagicMock(status=200)
        response.__enter__.return_value = response
        opener = MagicMock()
        return clock, sleep, children, response, opener

    def test_slow_router_import_can_exceed_old_120_second_window(self):
        clock, sleep, children, response, opener = self.router_wait_fixture()
        opener.open.side_effect = [ConnectionRefusedError("importing")] * 150 + [response]
        with patch.object(q.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(q.time, 'sleep', side_effect=sleep), \
             patch.object(q.urllib.request, 'build_opener', return_value=opener), \
             patch('builtins.print'):
            result = q.wait_router_ready(self.cfg, children)
        self.assertTrue(result['ready'])
        self.assertEqual(result['elapsed_seconds'], 150)
        self.assertEqual(opener.open.call_count, 151)

    def test_router_wait_checks_all_processes(self):
        for failed in ("prefill", "decode", "router"):
            with self.subTest(failed=failed):
                _, _, children, _, opener = self.router_wait_fixture()
                children[failed].poll.return_value = 1
                with patch.object(q.urllib.request, 'build_opener', return_value=opener):
                    with self.assertRaisesRegex(RuntimeError, failed + ' exited'):
                        q.wait_router_ready(self.cfg, children)
                opener.open.assert_not_called()

    def test_router_never_ready_obeys_deadline_and_reports_last_error(self):
        clock, sleep, children, _, opener = self.router_wait_fixture()
        self.cfg['router_startup_timeout_seconds'] = 3
        opener.open.side_effect = ConnectionRefusedError("still importing")
        with patch.object(q.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(q.time, 'sleep', side_effect=sleep), \
             patch.object(q.urllib.request, 'build_opener', return_value=opener), \
             patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'timed out after 3s: still importing'):
                q.wait_router_ready(self.cfg, children)
        self.assertEqual(clock[0], 3)
        self.assertEqual(opener.open.call_count, 3)

    def test_router_protocol_error_and_non_200_sleep_then_retry(self):
        clock, sleep, children, response, opener = self.router_wait_fixture()
        unavailable = MagicMock(status=503)
        unavailable.__enter__.return_value = unavailable
        opener.open.side_effect = [q.http.client.BadStatusLine('partial'), unavailable, response]
        with patch.object(q.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(q.time, 'sleep', side_effect=sleep), \
             patch.object(q.urllib.request, 'build_opener', return_value=opener), \
             patch('builtins.print'):
            self.assertTrue(q.wait_router_ready(self.cfg, children)['ready'])
        self.assertEqual(clock[0], 2)

    def test_import_profile_is_router_only(self):
        self.assertIn('importtime', m.router_plan(self.cfg)['command'])
        for worker in m.plan(self.cfg)['workers']:
            self.assertNotIn('importtime', worker['command'])
        self.assertNotIn('importtime', m.workload_plan(self.cfg)['command'])
        self.cfg.pop('router_profile_imports')
        self.assertNotIn('importtime', m.router_plan(self.cfg)['command'])

    def test_router_startup_timeout_rejects_invalid_values(self):
        for value in (True, 0, -1, float('inf'), float('nan'), '1800'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                self.cfg['router_startup_timeout_seconds'] = value
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(self.cfg))
                with self.assertRaises(ValueError):
                    m.load_config(path)


if __name__ == '__main__':
    unittest.main()
