import sys
from pathlib import Path


MIXED_DIR = Path(__file__).resolve().parents[1] / "examples" / "mixed"
sys.path.insert(0, str(MIXED_DIR))

from generate_with_retool import format_conversation_with_tools  # noqa: E402


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Execute Python code.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
]


def test_format_conversation_unwraps_qwen_chat_template():
    prompt = (
        "<|im_start|>user\nSolve the problem.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    formatted = format_conversation_with_tools(prompt, tools=TOOLS)

    assert formatted.count("<|im_start|>system") == 1
    assert formatted.count("<|im_start|>user") == 1
    assert formatted.count("<|im_start|>assistant") == 1
    assert "<|im_start|>user\nSolve the problem.<|im_end|>" in formatted
    assert "<|im_start|>user<|im_start|>user" not in formatted


def test_format_conversation_keeps_plain_user_prompt():
    formatted = format_conversation_with_tools("Solve the problem.", tools=TOOLS)

    assert formatted.count("<|im_start|>user") == 1
    assert formatted.count("<|im_start|>assistant") == 1
    assert "Solve the problem." in formatted
