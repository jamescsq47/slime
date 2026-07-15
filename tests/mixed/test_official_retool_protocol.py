import asyncio
import importlib
import sys
from pathlib import Path


MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
if str(MIXED_DIR) not in sys.path:
    sys.path.insert(0, str(MIXED_DIR))

retool = importlib.import_module("generate_with_retool")


def test_default_prompt_makes_tool_optional_and_formats_actions():
    rendered = retool.format_conversation_with_tools(
        "What is 2 + 2?", tools=[{"name": "code_interpreter"}]
    )

    assert "Using code_interpreter is OPTIONAL" in rendered
    assert "Do not use Markdown code fences" in rendered
    assert "exactly one <tool_call> JSON block" in rendered
    assert r"#### \boxed{answer}" in rendered


def test_protocol_is_appended_to_existing_system_prompt():
    rendered = retool.format_conversation_with_tools(
        "What is 2 + 2?",
        tools=[{"name": "code_interpreter"}],
        system_prompt="Preserve this task-specific instruction.",
    )

    assert "Preserve this task-specific instruction." in rendered
    assert "Using code_interpreter is OPTIONAL" in rendered


def test_invalid_action_feedback_matches_tool_call_protocol():
    observation, done = asyncio.run(retool.execute_predictions("unfinished response"))

    assert not done
    assert "<tool_call> JSON block" in observation
    assert r"#### \boxed{answer}" in observation
    assert "<code> and </code>" not in observation
