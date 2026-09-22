import unittest
import tempfile
from pathlib import Path

from multinode_smoke import output_ids, committed_path_ranks


class SmokeHelperTests(unittest.TestCase):
    def test_path_evidence_requires_actual_commit_and_exact_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {'nodes': [{'role': 'prefill', 'node_id': 'p', 'engine_id': 'engine'}],
                   'router': {'node_id': 'p'}, 'local_root': d, 'run_id': 'run'}
            log = Path(d) / 'run/engine/service.log'
            log.parent.mkdir(parents=True)
            log.write_text('[time TP0] AgenticKV early_direct_admit snapshot=s:0 tokens=64\n'
                           '[time TP1] AgenticKV early_direct_bind snapshot=s:0 tokens=64\n'
                           '[time TP2] AgenticKV early_direct_admit snapshot=s:01 tokens=64\n'
                           '[time TP0] AgenticKV early_direct_admit snapshot=s:0 tokens=64\n'
                           '[time TP3] AgenticKV shared_host_h2d_complete snapshot=s:0 tokens=64\n')
            self.assertEqual(committed_path_ranks(cfg, 's:0', 'direct'), [0])
            self.assertEqual(committed_path_ranks(cfg, 's:0', 'slow'), [3])

    def test_exact_ids_preferred(self):
        self.assertEqual(output_ids({"output_ids": [1, 4]}), [1, 4])

    def test_output_logprob_ids(self):
        response = {"meta_info": {"output_token_logprobs": [[-1.0, 10, "a"], [-0.5, 20, "b"]]}}
        self.assertEqual(output_ids(response), [10, 20])

    def test_never_retokenize_text(self):
        with self.assertRaisesRegex(RuntimeError, "retokenization"):
            output_ids({"text": "TOOL"})

    def test_reject_invalid_ids(self):
        for raw in ([], [True], [-1], ["10"]):
            with self.assertRaises(RuntimeError):
                output_ids({"output_ids": raw})


if __name__ == "__main__":
    unittest.main()
