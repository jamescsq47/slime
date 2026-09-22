"""Qwen3.5-122B-A10B BF16 colocated acceptance; shares the owned SWE runner.

This validates model execution, not multi-node hybrid KV/state transport.
"""
from pathlib import Path
from minimax_swe import main


def validate_model(raw):
    if raw.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]:
        raise ValueError("expected native Qwen3.5 MoE architecture")
    text = raw["text_config"]
    if raw.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("this experiment requires the official BF16 checkpoint")
    expected = {"num_hidden_layers": 48, "num_key_value_heads": 2,
                "num_attention_heads": 32, "num_experts": 256,
                "num_experts_per_tok": 8, "linear_num_key_heads": 16,
                "linear_num_value_heads": 64, "head_dim": 256}
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"unexpected 122B layout: {key}={text.get(key)}")
    if text.get("dtype") != "bfloat16":
        raise ValueError("expected BF16 checkpoint")
    if text.get("layer_types") != (["linear_attention"] * 3 + ["full_attention"]) * 12:
        raise ValueError("unexpected hybrid layer layout")


if __name__ == "__main__":
    raise SystemExit(main(default_config=Path(__file__).with_suffix(".json"),
                         default_model="Qwen3.5-122B-A10B", run_prefix="qwen35-122b"))
