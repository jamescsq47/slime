import copy
import json
from pathlib import Path
import unittest
import tempfile

import minimax_swe as runner
from qwen35_swe import validate_model


def model_config():
    return {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "text_config": {
        "num_hidden_layers": 48, "num_key_value_heads": 2, "num_attention_heads": 32,
        "num_experts": 256, "num_experts_per_tok": 8, "linear_num_key_heads": 16,
        "linear_num_value_heads": 64, "head_dim": 256, "dtype": "bfloat16",
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12}}


class QwenTests(unittest.TestCase):
    def test_layout_and_replicated_kv(self):
        validate_model(model_config())
        for field, value in [("num_key_value_heads", 8), ("linear_num_key_heads", 4),
                             ("dtype", "float8_e4m3fn"), ("layer_types", ["full_attention"] * 48)]:
            raw = model_config()
            raw["text_config"][field] = value
            with self.assertRaises(ValueError):
                validate_model(raw)
        raw = model_config()
        raw["quantization_config"] = {"quant_method": "fp8"}
        with self.assertRaises(ValueError):
            validate_model(raw)

    def test_native_colocated_and_unchanged_workload(self):
        cfg = runner.read_config(Path(__file__).with_name("qwen35_swe.json"))
        server, infer = (runner.commands(cfg, Path('/model'), Path('/run'), Path('/workload'))[k]
                         for k in ('model', 'inference'))
        for key, value in [("--tp-size", "8"), ("--ep-size", "1"),
                           ("--mem-fraction-static", "0.8"), ("--moe-runner-backend", "triton"),
                           ("--mamba-scheduler-strategy", "extra_buffer"),
                           ("--max-prefill-tokens", "8192"),
                           ("--reasoning-parser", "glm45"), ("--tool-call-parser", "qwen3_coder")]:
            self.assertEqual(server[server.index(key) + 1], value)
        for flag in ("--disaggregation-mode", "--enable-hierarchical-cache", "--speculative-algorithm"):
            self.assertNotIn(flag, server)
        self.assertEqual(infer[infer.index('--requests') + 1], '500')
        self.assertEqual(infer[infer.index('--max-inflight') + 1], '64')
        self.assertEqual(infer[infer.index('--max-response-length') + 1], '81920')
        self.assertIn('--log-requests', server)
        for flag, value in [('--temperature', '0.6'), ('--top-p', '0.95'), ('--top-k', '20')]:
            self.assertEqual(infer[infer.index(flag) + 1], value)
        import yaml
        workload = yaml.safe_load((runner.ROOT / cfg['workload_config']).read_text())
        opts = workload['datasets'][0]['options']
        self.assertEqual(opts['action_protocol'], 'openai_tools')
        self.assertNotIn('command_contract', opts)
        self.assertEqual(opts['max_response_tokens'], 81920)
        for key in ['docker_cpu', 'docker_memory_gb', 'docker_user', 'docker_normalize_tool_timeout']:
            self.assertNotIn(key, opts)  # Match actual 27B defaults, not its old prose table.
        self.assertEqual(opts['verifier_mode'], 'inline')
        self.assertEqual(opts['max_tokens_per_turn'], 8192)
        self.assertEqual(opts['max_turns'], 64)

    def test_alignment_drift_rejected_before_gpu_launch(self):
        cfg = runner.read_config(Path(__file__).with_name('qwen35_swe.json'))
        for key, value in [('max_prefill_tokens', 16384), ('max_response_length', 524288),
                           ('context_length', 262144), ('log_requests', False),
                           ('workload_config', 'examples/pd/configs/experiments/swe_bench_verified_miles_pr51_8k_t64.yaml')]:
            changed = copy.deepcopy(cfg)
            changed[key] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'config.json'
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.read_config(path)


if __name__ == '__main__':
    unittest.main()
