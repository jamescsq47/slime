import copy
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("multinode", ROOT / "multinode.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MultiNodeV2PlanTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "multinode.example.json").read_text())

    def validate(self, cfg=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(cfg or self.cfg))
            return m.load_config(path)

    def test_tp8_plan_is_cpu_only_and_has_tcp_relay(self):
        cfg = self.validate()
        plan = m.plan(cfg)
        self.assertEqual(plan["control"]["environment"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue(any("agentic_group_protocol" in arg for arg in plan["control"]["command"]))
        self.assertIn("--links", plan["control"]["command"])
        self.assertEqual(len(plan["workers"]), 2)
        self.assertEqual(plan["workers"][1]["environment"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
        self.assertEqual(
            plan["workers"][1]["environment"][
                "SGLANG_AGENTIC_MULTINODE_DECODE_GROWTH_TOKENS"
            ],
            "512",
        )

    def test_no_shared_control_or_runtime_ledger_environment(self):
        result = m.plan(self.validate())
        legacy_v1_keys = {
            "SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY",
            "SGLANG_AGENTIC_KV_D_HOSTLESS",
            "SGLANG_AGENTIC_KV_HOST_STAGING",
            "SGLANG_AGENTIC_KV_P2D_HOST_STAGING",
            "SGLANG_AGENTIC_KV_RELAY_ENABLED",
            "SGLANG_PD_P_READY_BACKPRESSURE_MODE",
            "SGLANG_PD_P_READY_REQUEST_CAP",
            "SGLANG_PD_LATE_BIND_GLOBAL_DECODE",
        }
        for worker in result["workers"]:
            env = worker["environment"]
            self.assertFalse(any("CONTROL_ROOT" in key for key in env))
            self.assertFalse(any("LEDGER_PATH" in key for key in env))
            self.assertFalse(any("P_READY_DIR" in key for key in env))
            self.assertFalse(any("METADATA_DIR" in key for key in env))
            self.assertTrue(legacy_v1_keys.isdisjoint(env))
            self.assertNotIn("SGLANG_AGENTIC_GROUP_RANK", env)
        self.assertEqual(
            result["router"]["command"][:3],
            [self.cfg["python"], "-m", "sglang_router.launch_router"],
        )
        self.assertIn("--mini-lb", result["router"]["command"])
        self.assertNotIn(
            "launch_late_binding_router.py", result["router"]["command"]
        )

    def test_same_parent_env_yields_eight_explicit_rank_identities(self):
        worker = m.worker_plan(self.validate(), self.cfg["nodes"][0])
        rank_envs = worker["rank_environments"]
        self.assertEqual([entry["SGLANG_AGENTIC_GROUP_RANK"] for entry in rank_envs], list(map(str, range(8))))
        for entry in rank_envs:
            self.assertEqual(entry["SGLANG_AGENTIC_GROUP_GROUP_ID"], "prefill-0--decode-0")
            self.assertEqual(entry["SGLANG_AGENTIC_GROUP_ENDPOINT_GROUP"], "prefill-0")
            self.assertEqual(entry["SGLANG_AGENTIC_GROUP_ENDPOINT_ROLE"], "prefill")

    def test_link_contains_both_endpoint_groups_atomically(self):
        links = m.control_links(self.validate())
        spec = links["prefill-0--decode-0"]
        self.assertEqual(spec["coordinator"], {"endpoint_group": "prefill-0", "rank": 0})
        self.assertEqual({e["endpoint_group"] for e in spec["endpoints"]}, {"prefill-0", "decode-0"})
        self.assertEqual(sum(e["size"] for e in spec["endpoints"]), 16)

    def test_local_root_is_only_output_location(self):
        cfg = self.validate()
        plan = m.plan(cfg)
        for component in [plan["control"], plan["router"], *plan["workers"]]:
            self.assertTrue(component["local_run_dir"].startswith(cfg["local_root"] + "/"))
            self.assertFalse(any(cfg["local_root"] in value for value in component["environment"].values()))

    def test_legacy_shared_mode_rejected_and_old_control_root_ignored(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["control_root"] = "/some/shared/path"
        loaded = self.validate(cfg)
        self.assertNotIn("control_root", m.common_env(loaded))
        cfg["legacy_shared_control"] = True
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.validate(cfg)

    def test_group_control_required_and_port_collision_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        del cfg["group_control"]
        with self.assertRaisesRegex(ValueError, "group_control"):
            self.validate(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["group_control"]["port"] = cfg["nodes"][0]["port"]
        with self.assertRaisesRegex(ValueError, "collides"):
            self.validate(cfg)

    def test_tp1_tp2_tp4_command_generation(self):
        for tp in (1, 2, 4):
            cfg = copy.deepcopy(self.cfg)
            cfg["tp_size"] = tp
            for node in cfg["nodes"]:
                node["gpus"] = list(range(tp))
            loaded = self.validate(cfg)
            self.assertEqual(len(m.plan(loaded)["workers"][0]["rank_environments"]), tp)

    def test_qwen35_plan_restores_hybrid_runtime_contract(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(
            {
                "model_family": "qwen35_moe",
                "prefill_mamba_full_memory_ratio": 0.75,
                "decode_mamba_full_memory_ratio": 0.5,
                "seed": 2026,
            }
        )
        result = m.plan(self.validate(cfg))
        for node, worker in zip(cfg["nodes"], result["workers"]):
            env = worker["environment"]
            command = worker["command"]
            self.assertEqual(env["SGLANG_AGENTIC_MULTINODE_QWEN35_HYBRID"], "1")
            self.assertEqual(env["SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED"], "true")
            self.assertIn("--mamba-full-memory-ratio", command)
            expected = "0.75" if node["role"] == "prefill" else "0.5"
            self.assertEqual(command[command.index("--mamba-full-memory-ratio") + 1], expected)
            self.assertIn("--tool-call-parser", command)
            self.assertEqual(command[command.index("--tool-call-parser") + 1], "qwen3_coder")
            self.assertEqual(
                env["PATH"].split(m.os.pathsep)[0], str(Path(cfg["python"]).parent)
            )

    def test_unknown_model_family_is_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model_family"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported multi-node model family"):
            self.validate(cfg)

    def test_runtime_scrubs_another_experiment_environment(self):
        worker = m.worker_plan(self.validate(), self.cfg["nodes"][0])
        with patch.dict(m.os.environ, {"SGLANG_AGENTIC_KV_LEDGER_PATH": "/old", "HOST_IP": "127.0.0.1"}):
            env = m.runtime_env(worker)
        self.assertNotIn("SGLANG_AGENTIC_KV_LEDGER_PATH", env)
        self.assertNotIn("HOST_IP", env)
        self.assertEqual(env["SGLANG_AGENTIC_GROUP_ENDPOINT"], "tcp://28.49.24.105:62080")

    def test_listening_port_conflict_is_fail_closed(self):
        cfg = self.validate()
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            cfg["nodes"][0]["port"] = occupied.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                m.check_listen_ports(cfg, m.worker_plan(cfg, cfg["nodes"][0]), "worker")

    def test_protocol_check_never_launches_on_import_failure(self):
        cfg = self.validate()
        result = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "missing"})()
        with patch.object(m.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                m.protocol_check(cfg, {})


if __name__ == "__main__":
    unittest.main()
