import unittest

from multinode_smoke import output_ids


class SmokeHelperTests(unittest.TestCase):
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
