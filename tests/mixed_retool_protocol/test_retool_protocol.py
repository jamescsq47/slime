import asyncio
import importlib
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed_retool_protocol"
MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
for path in (str(EXPERIMENT_DIR), str(MIXED_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

protocol = importlib.import_module("generate_with_retool_protocol")


def test_default_prompt_makes_tool_optional_and_formats_actions():
    rendered = protocol.format_conversation_with_tools("What is 2 + 2?", tools=[{"name": "code_interpreter"}])

    assert "Using code_interpreter is OPTIONAL" in rendered
    assert "Do not use Markdown code fences" in rendered
    assert "exactly one <tool_call> JSON block" in rendered
    assert r"#### \boxed{answer}" in rendered


def test_protocol_is_appended_to_existing_system_prompt():
    rendered = protocol.format_conversation_with_tools(
        "What is 2 + 2?",
        tools=[{"name": "code_interpreter"}],
        system_prompt="Please reason step by step.",
    )

    assert "Please reason step by step." in rendered
    assert "Using code_interpreter is OPTIONAL" in rendered
    assert r"#### \boxed{answer}" in rendered


def test_gsm8k_marker_is_a_terminal_answer():
    assert protocol.postprocess_predictions("Reasoning here.\n#### 18") == ("answer", "18")


def test_boxed_answer_remains_supported():
    assert protocol.postprocess_predictions(r"Therefore \boxed{42}.") == ("answer", "42")


def test_invalid_action_feedback_matches_current_protocol():
    observation, done = asyncio.run(protocol.execute_predictions("unfinished response"))

    assert not done
    assert "<tool_call> JSON block" in observation
    assert r"#### \boxed{answer}" in observation
    assert "<code> and </code>" not in observation
