import copy
import importlib.util
import json
import socket
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("multinode", ROOT / "multinode.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MultiNodePlanTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "multinode.example.json").read_text())

    def validate(self, cfg):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps(cfg))
            return m.load_config(p)

    def test_tp8_plan_no_cuda(self):
        cfg = self.validate(self.cfg)
        p = m.plan(cfg)
        self.assertEqual(len(p["workers"]), 2)
        self.assertEqual(p["workers"][1]["environment"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
        self.assertEqual(p["workers"][1]["host_capacity_gib_per_direction"]["d2p_source"], 64)

    def test_refuse_other_experiment_listening_port(self):
        cfg = copy.deepcopy(self.cfg)
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            cfg["nodes"][0]["port"] = occupied.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                m.check_listen_ports(cfg, m.worker_plan(cfg, cfg["nodes"][0]), True)

    def test_workload_has_no_listening_port_probe(self):
        p = {"engine_id": "workload"}
        with patch.object(m.socket, "socket", side_effect=AssertionError("unexpected socket")):
            m.check_listen_ports(self.cfg, p, False)

    def test_tp1_tp2_tp4(self):
        for tp in (1, 2, 4):
            cfg = copy.deepcopy(self.cfg)
            cfg["tp_size"] = tp
            for n in cfg["nodes"]:
                n["gpus"] = list(range(tp))
            self.validate(cfg)

    def test_reject_mismatched_shards(self):
        self.cfg["nodes"][1]["gpus"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "exactly TP"):
            self.validate(self.cfg)

    def test_reject_same_host(self):
        self.cfg["nodes"][1]["host_ip"] = self.cfg["nodes"][0]["host_ip"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.validate(self.cfg)

    def test_reject_loopback_and_control_tmpfs(self):
        for field, value in (("host_ip", "127.0.0.1"), ("control_root", "/dev/shm/test")):
            cfg = copy.deepcopy(self.cfg)
            if field == "host_ip":
                cfg["nodes"][0][field] = value
            else:
                cfg[field] = value
            with self.assertRaises(ValueError):
                self.validate(cfg)

    def test_reject_nonfinite_capacity_or_timeout(self):
        for field in ("d2p_host_gib_per_rank", "direct_admission_seconds"):
            for value in (float("nan"), float("inf"), True):
                cfg = copy.deepcopy(self.cfg)
                cfg[field] = value
                with self.assertRaises(ValueError):
                    self.validate(cfg)

    def test_shared_control_but_source_local_host(self):
        plans = m.plan(self.cfg)["workers"]
        a, b = [p["environment"] for p in plans]
        self.assertEqual(a["SGLANG_PD_P_READY_DIR"], b["SGLANG_PD_P_READY_DIR"])
        self.assertNotEqual(a["SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR"], b["SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR"])
        self.assertNotIn("--enable-hierarchical-cache", plans[0]["command"])

    def test_engine_gate_fail_closed(self):
        result = type("Result", (), {"returncode": 0, "stdout": '{"integrated": false, "features": []}', "stderr": ""})()
        with patch.object(m.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "not integrated"):
                m.capability_check(self.cfg, {})

    def test_missing_feature_fail_closed(self):
        result = type("Result", (), {"returncode": 0, "stdout": '{"integrated": true, "features": []}', "stderr": ""})()
        with patch.object(m.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                m.capability_check(self.cfg, {})

    def test_atomic_preflight_publish(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = copy.deepcopy(self.cfg)
            cfg["control_root"] = d
            result = m.fs_publish(cfg, cfg["nodes"][0])
            path = m.control_dir(cfg) / "preflight/node-p.json"
            self.assertEqual(json.loads(path.read_text()), result)
            self.assertEqual((path.parent / "node-p.exclusive").read_text(), result["nonce"])

    def test_duplicate_publish_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = copy.deepcopy(self.cfg)
            cfg["control_root"] = d
            m.fs_publish(cfg, cfg["nodes"][0])
            with self.assertRaises(FileExistsError):
                m.fs_publish(cfg, cfg["nodes"][0])

    def test_shared_control_root_is_not_double_run_scoped(self):
        env = m.common_env(self.cfg)
        self.assertEqual(env["SGLANG_AGENTIC_MULTINODE_CONTROL_ROOT"], self.cfg["control_root"])
        self.assertEqual(env["SGLANG_PD_P_READY_DIR"], str(m.control_dir(self.cfg)))

    def test_plans_match_engine_multinode_contract(self):
        import sys
        engine_path = ROOT.parents[2] / "sglang/python/sglang/srt/disaggregation/agentic_multinode.py"
        if not engine_path.exists():
            self.skipTest("sibling SGLang checkout required for integration contract test")
        engine_spec = importlib.util.spec_from_file_location("_multinode_contract", engine_path)
        engine = importlib.util.module_from_spec(engine_spec)
        sys.modules[engine_spec.name] = engine
        engine_spec.loader.exec_module(engine)
        for worker in m.plan(self.cfg)["workers"]:
            config = engine.load_multinode_config(worker["environment"])
            self.assertEqual(config.tp_size, 8)
            self.assertEqual(config.control_directory, str(m.control_dir(self.cfg)))
            from types import SimpleNamespace
            args = SimpleNamespace(tp_size=8, dp_size=1, pp_size=1, nnodes=1,
                                   disaggregation_mode=config.role, disaggregation_transfer_backend="nixl",
                                   enable_dp_attention=False, enable_hierarchical_cache=False,
                                   hicache_storage_backend=None, speculative_algorithm=None,
                                   disaggregation_decode_enable_offload_kvcache=False)
            with patch.dict(m.os.environ, worker["environment"], clear=True):
                checked = engine.validate_multinode_runtime(args)
            self.assertEqual(checked, config)

    def test_same_host_preflight_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = copy.deepcopy(self.cfg)
            cfg["control_root"] = d
            for n in cfg["nodes"]:
                m.fs_publish(cfg, n)
            with self.assertRaisesRegex(ValueError, "duplicate host"):
                m.fs_verify(cfg)

    def test_crosshost_contract_simulated_records(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = copy.deepcopy(self.cfg)
            cfg["control_root"] = d
            for n in cfg["nodes"]:
                record = m.fs_publish(cfg, n)
                # Simulated independent hosts; this is not a remote FS test.
                record["boot_id"] = "simulated-" + n["node_id"]
                m.atomic_json(m.control_dir(cfg) / "preflight" / (n["node_id"] + ".json"), record)
            self.assertTrue(m.fs_verify(cfg)["metadata_visible"])
            (m.control_dir(cfg) / "preflight/node-d.hardlink").unlink()
            with self.assertRaises(FileNotFoundError):
                m.fs_verify(cfg)

    def test_no_accidental_launch_when_capability_missing(self):
        with patch.object(m, "capability_check", side_effect=RuntimeError("not integrated")), patch.object(m, "supervise") as supervisor:
            with self.assertRaisesRegex(RuntimeError, "not integrated"):
                m.start_worker(self.cfg, self.cfg["nodes"][0])
            supervisor.assert_not_called()

    def test_reject_multiple_prefill_nodes(self):
        node = copy.deepcopy(self.cfg["nodes"][0])
        node.update(node_id="node-p2", engine_id="prefill-1", host_ip="28.49.193.152")
        self.cfg["nodes"].append(node)
        with self.assertRaisesRegex(ValueError, "one logical P"):
            self.validate(self.cfg)

    def test_router_remote_urls_and_disabled_cuda(self):
        p = m.router_plan(self.cfg)
        self.assertIn("http://28.49.24.105:30000", p["command"])
        self.assertIn("http://28.59.0.110:30000", p["command"])
        self.assertEqual(p["environment"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(p["environment"]["SGLANG_AGENTIC_MULTINODE_ROLE"], "router")

    def test_workload_requires_explicit_argv(self):
        with self.assertRaisesRegex(ValueError, "workload_command"):
            m.workload_plan(self.cfg)

    def test_smoke_uses_explicit_config_and_no_local_gpu(self):
        p = m.smoke_plan(self.cfg, "/tmp/test-config.json")
        self.assertIn("--config", p["command"])
        self.assertIn("/tmp/test-config.json", p["command"])
        self.assertEqual(p["environment"]["CUDA_VISIBLE_DEVICES"], "")

    def test_runtime_clears_previous_experiment_env(self):
        with patch.dict(m.os.environ, {"SGLANG_AGENTIC_KV_LEDGER_PATH": "/old/ledger", "SGLANG_ENABLE_HICACHE": "true", "HOST_IP": "127.0.0.1"}):
            env = m.runtime_env(m.worker_plan(self.cfg, self.cfg["nodes"][0]))
        self.assertNotIn("SGLANG_ENABLE_HICACHE", env)
        self.assertNotIn("HOST_IP", env)
        self.assertNotEqual(env["SGLANG_AGENTIC_KV_LEDGER_PATH"], "/old/ledger")

    def test_full_method_congestion_defaults_and_optout(self):
        env = m.common_env(self.cfg)
        self.assertEqual(env["SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE"], "true")
        self.assertEqual(env["SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH"], "32")
        self.cfg["slow_congestion_recompute"] = False
        self.assertEqual(m.common_env(self.cfg)["SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE"], "false")


if __name__ == "__main__":
    unittest.main()
