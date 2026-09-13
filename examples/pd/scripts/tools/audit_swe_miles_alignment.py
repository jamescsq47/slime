#!/usr/bin/env python3
"""CPU-only upstream-contract and native SGLang prompt parity audit.

Run in pd_mamba with the repo root and examples/pd on PYTHONPATH. Never loads
model weights, starts a server, or modifies historical trace files.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--miles-reference", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from data.swe_bench_openenv import harness as h

    source = subprocess.check_output([
        "git", "-C", args.miles_reference, "show",
        "e2e516603ad6688f3ba7a5e5e8d2b5a7606fa849:examples/experimental/swe-agent-v2/eval_swebench_daytona.py",
    ], text=True)
    names = {"_BASH_FENCE_RE", "_GENERIC_FENCE_RE", "_THINK_RE", "_SYSTEM_PROMPT"}
    nodes = [n for n in ast.parse(source).body if (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    ) or (isinstance(n, ast.FunctionDef) and n.name == "_extract_command")]
    namespace = {"re": re, "Any": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned_miles_contract", "exec"), namespace)
    assert h._SYSTEM_PROMPT == namespace["_SYSTEM_PROMPT"]
    cases = ["TASK_COMPLETE", "done", "```bash\npwd\n```", "```\npwd\n```",
             "```bash\ntask_complete done\n```", "```bash\npwd\n```\n```sh\nls\n```",
             "REQUEST_REVIEW", "<think>private</think>\n```sh\nls\n```"]
    for content in cases:
        assert h.extract_miles_command({"content": content}) == namespace["_extract_command"]({"content": content})

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    service = object.__new__(OpenAIServingChat)
    service.template_manager = SimpleNamespace(jinja_template_content_format="string")
    service.tokenizer_manager = SimpleNamespace(tokenizer=tokenizer)
    service.chat_encoding_spec = None
    service._tokenizer_auto_adds_specials = False
    service._encode_messages = lambda *a, **k: None
    checked = 0
    mismatches = []
    with open(args.traces) as stream:
        for line in stream:
            record = json.loads(line)
            meta = record["metadata"]
            trajectory = meta["openenv_trajectory"]
            if (not trajectory.get("messages")
                    or trajectory["messages"][-1].get("role") != "assistant"
                    or any(e.get("context_compacted") for e in trajectory["turn_events"])):
                raise ValueError("Audit requires assistant-terminal, uncompacted traces; cannot infer the exact prompt")
            # Recorded terminal prompts, plus a fenced-shell template probe
            # constructed from the same task's executed history (not a rerun).
            fenced = [{"role": "system", "content": h._SYSTEM_PROMPT},
                      {"role": "user", "content": meta["problem_statement"]}]
            for event in trajectory["turn_events"][:-1]:
                if event.get("command") and "observation_sent_to_model" in event:
                    fenced.extend([
                        {"role": "assistant", "content": f'```bash\n{event["command"]}\n```',
                         "reasoning_content": event.get("reasoning_content", "")},
                        {"role": "user", "content": event["observation_sent_to_model"]},
                    ])
            for mode, messages, tools in [
                ("legacy_tools", trajectory["messages"][:-1], [h._SHELL_TOOL]),
                ("miles_fenced", fenced, None),
            ]:
                req = ChatCompletionRequest(model="audit", messages=messages, tools=tools,
                                            chat_template_kwargs={"enable_thinking": True})
                normalized_tools = [t.model_dump() for t in req.tools] if req.tools else None
                native = service._apply_jinja_template(req, normalized_tools, False).prompt_ids
                local = h._render_prompt(tokenizer, messages, enable_thinking=True, tools=tools)
                checked += 1
                if native != local:
                    mismatches.append([meta["instance_id"], mode, len(native), len(local)])
    if not checked:
        raise ValueError("No trajectories checked")
    print(json.dumps({"upstream_contract_cases": len(cases), "system_prompt_equal": True,
                      "prompts_checked": checked, "mismatches": mismatches}))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
