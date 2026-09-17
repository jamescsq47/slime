import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import minimax_swe as m
import multinode


def model_config():
    return dict(model_type="minimax_m2", architectures=["MiniMaxM2ForCausalLM"],
                num_hidden_layers=62, attn_type_list=[1] * 62, num_attention_heads=48,
                num_key_value_heads=8, num_local_experts=256, intermediate_size=1536,
                hidden_size=3072, quantization_config=dict(quant_method="fp8", weight_block_size=[128, 128]))


class MiniMaxTests(unittest.TestCase):
    def test_failed_cleanup_still_stops_model_and_own_containers(self):
        server, infer = Mock(), Mock()
        server.poll.return_value = infer.poll.return_value = None
        infer.wait.side_effect = m.subprocess.TimeoutExpired('inference', 45)
        with patch.object(m.subprocess, 'check_output', return_value='owned-id\n') as listing, \
             patch.object(m.subprocess, 'run') as remove:
            with self.assertRaisesRegex(RuntimeError, 'cleanup incomplete'):
                m.cleanup({'model': server, 'inference': infer}, Path('/tmp/unique'))
        server.terminate.assert_called_once()
        server.wait.assert_called_once()
        infer.terminate.assert_called_once()
        self.assertIn('label=pd.swe.run_id=unique', listing.call_args.args[0])
        self.assertEqual(remove.call_args.args[0], ['docker', 'rm', '-f', 'owned-id'])

    def test_minimax_tp8_ep8_and_native_colocated(self):
        cfg = m.read_config(m.DEFAULT_CONFIG)
        cmd = m.commands(cfg, Path('/models/m27'), Path('/tmp/run'), Path('/tmp/workload'))
        server, infer = cmd['model'], cmd['inference']
        for key, value in [('--tp-size', '8'), ('--ep-size', '8'), ('--mem-fraction-static', '0.8'),
                           ('--kv-cache-dtype', 'bfloat16')]:
            self.assertEqual(server[server.index(key) + 1], value)
        self.assertNotIn('--disaggregation-mode', server)
        self.assertNotIn('--enable-hierarchical-cache', server)
        self.assertNotIn('--speculative-algorithm', server)
        self.assertEqual(infer[infer.index('--max-inflight') + 1], '64')
        self.assertEqual(infer[infer.index('--requests') + 1], '500')
        self.assertIn('--preserve-source-order', infer)

    def test_no_inherited_pd_or_qwen_state(self):
        cfg = m.read_config(m.DEFAULT_CONFIG)
        with patch.dict(os.environ, {'SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED': 'true',
                        'PD_P_READY_DIR': '/old', 'SGLANG_OVERLAY_ROOT': '/wrong',
                        'PYTHONPATH': '/wrong', 'NCCL_IB_HCA': 'mlx5_0'}):
            env = m.environment(Path('/tmp/test'), Path('/data'), cfg)
        self.assertNotIn('SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED', env)
        self.assertNotIn('SGLANG_OVERLAY_ROOT', env)
        self.assertNotIn('PD_P_READY_DIR', env)
        self.assertEqual(env['NCCL_IB_HCA'], 'mlx5_0')
        self.assertNotIn('/wrong', env['PYTHONPATH'])

    def test_unsupported_models_and_layouts_rejected(self):
        cfg = {'model_family': 'minimax_m2', 'tp_size': 8}
        multinode.validate_launch_model(cfg, model_config())
        for key, value in [('model_type', 'qwen3_5_moe'), ('num_key_value_heads', 2),
                           ('num_hidden_layers', 0), ('attn_type_list', [0] * 62),
                           ('intermediate_size', 192)]:
            raw = model_config()
            raw[key] = value
            with self.assertRaises(ValueError):
                multinode.validate_launch_model(cfg, raw)

    def test_multinode_keeps_qwen_default_and_minimax_ep_is_explicit(self):
        cfg = json.loads(Path(multinode.__file__).with_name('multinode.example.json').read_text())
        qwen = multinode.worker_plan(cfg, cfg['nodes'][0])['command']
        self.assertNotIn('--ep-size', qwen)
        cfg['model_family'] = 'minimax_m2'
        cmd = multinode.worker_plan(cfg, cfg['nodes'][0])['command']
        self.assertEqual(cmd[cmd.index('--ep-size') + 1], '8')
        self.assertEqual(cmd[cmd.index('--reasoning-parser') + 1], 'minimax-append-think')

    def test_acceptance_settings_cannot_silently_drift(self):
        cfg = m.read_config(m.DEFAULT_CONFIG)
        for key, value in [('max_inflight', 128), ('mem_fraction_static', .85), ('ep_size', 1)]:
            changed = copy.deepcopy(cfg)
            changed[key] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'test.json'
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    m.read_config(path)


if __name__ == '__main__':
    unittest.main()
