import asyncio
import importlib
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed_error_advisor"
MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
for path in (str(EXPERIMENT_DIR), str(MIXED_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

advisor = importlib.import_module("generate_with_retool_advisor")


def test_tool_error_uses_advisor_feedback(monkeypatch):
    async def fake_execute_tool(_name, _arguments):
        return "Error: SyntaxError: invalid syntax at ```py"

    async def fake_advisor(system_prompt, error, failed_action):
        assert "code_interpreter" in system_prompt
        assert "SyntaxError" in error
        assert "```py" in failed_action
        return "Remove the Markdown fences and pass only raw Python code."

    monkeypatch.setattr(advisor.tool_registry, "execute_tool", fake_execute_tool)
    monkeypatch.setattr(advisor, "get_advisor_feedback", fake_advisor)

    prediction = (
        '<tool_call>{"name":"code_interpreter","arguments":'
        '{"code":"```py\\nprint(18)\\n```"}}</tool_call>'
    )
    observation, done = asyncio.run(advisor.execute_predictions(prediction, "Use code_interpreter."))

    assert not done
    assert "SyntaxError" in observation
    assert "Remove the Markdown fences" in observation
    assert "<advisor_feedback>" in observation


def test_invalid_action_replaces_fixed_feedback(monkeypatch):
    async def fake_advisor(_system_prompt, error, _failed_action):
        assert "No valid final answer" in error
        return "Return either a boxed answer or a valid tool_call JSON object."

    monkeypatch.setattr(advisor, "get_advisor_feedback", fake_advisor)
    observation, done = asyncio.run(advisor.execute_predictions("unfinished response", "system"))

    assert not done
    assert "Return either a boxed answer" in observation
    assert "My previous action is invalid" not in observation


def test_empty_tool_output_is_sent_to_advisor(monkeypatch):
    async def fake_execute_tool(_name, _arguments):
        return ""

    async def fake_advisor(_system_prompt, error, _failed_action):
        assert "produced no stdout" in error
        return "Wrap the expression in print(...) so the tool emits its result."

    monkeypatch.setattr(advisor.tool_registry, "execute_tool", fake_execute_tool)
    monkeypatch.setattr(advisor, "get_advisor_feedback", fake_advisor)
    prediction = (
        '<tool_call>{"name":"code_interpreter","arguments":'
        '{"code":"2 + 1"}}</tool_call>'
    )

    observation, done = asyncio.run(advisor.execute_predictions(prediction, "Use code_interpreter."))

    assert not done
    assert "produced no stdout" in observation
    assert "Wrap the expression in print" in observation


def test_repeated_tool_call_is_not_executed_again(monkeypatch):
    executed = False

    async def fake_execute_tool(_name, _arguments):
        nonlocal executed
        executed = True
        return "Output:\n3"

    async def fake_advisor(_system_prompt, error, _failed_action):
        assert "exact same tool call" in error
        return "Stop calling the tool and provide the final boxed answer."

    monkeypatch.setattr(advisor.tool_registry, "execute_tool", fake_execute_tool)
    monkeypatch.setattr(advisor, "get_advisor_feedback", fake_advisor)
    prediction = (
        '<tool_call>{"name":"code_interpreter","arguments":'
        '{"code":"print(2 + 1)"}}</tool_call>'
    )
    previous = advisor._extract_tool_call(prediction)

    observation, done = asyncio.run(
        advisor.execute_predictions(prediction, "Use code_interpreter.", previous)
    )

    assert not done
    assert not executed
    assert "Stop calling the tool" in observation
