import unittest

from deepseek_v4_preflight import assess


class DeepseekPreflightTests(unittest.TestCase):
    def setUp(self):
        self.config = {"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"],
                       "expert_dtype": "fp4", "compress_ratios": [0, 4, 128, 4]}

    def test_missing_model_and_layout_are_separate_blockers(self):
        result = assess(self.config, {"Qwen3ForCausalLM"})
        self.assertEqual(result["single_node_tp8_model_run"], "blocked")
        self.assertEqual(result["custom_bidirectional_pd"], "blocked")
        self.assertEqual(result["compression_ratios"], [0, 4, 128])

    def test_model_port_alone_does_not_enable_wrong_cache_adapter(self):
        result = assess(self.config, {"DeepseekV4ForCausalLM"})
        self.assertTrue(result["native_model_class_present"])
        self.assertEqual(result["single_node_tp8_model_run"], "not_validated")
        self.assertEqual(result["custom_bidirectional_pd"], "blocked")

    def test_other_model_not_false_hardware_pass(self):
        result = assess({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
                        {"Qwen3ForCausalLM"})
        self.assertEqual(result["custom_bidirectional_pd"], "not_validated")


if __name__ == "__main__":
    unittest.main()
