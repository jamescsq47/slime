"""Use upstream original questions and declared schemas, never solution metadata."""
import copy
import json
from pathlib import Path
from slime.utils.types import Sample

CASES = (1, 10, 13, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36)


def load_samples(context, dataset):
    selected = dataset.options.get("case_ids", list(CASES))
    if not set(selected) <= set(CASES):
        raise ValueError("Only reviewed offline cases are enabled")
    rows = json.loads(Path(dataset.path).read_text())
    samples = []
    for row in rows:
        if row["id"] not in selected:
            continue
        # The worker receives the same allowlisted information, not gold data.
        tools = copy.deepcopy(row["usage_tool_protocol"])
        clean = {"id": row["id"], "question": row["question"], "usage_tool_protocol": tools,
                 "metadata": {k: row["metadata"][k] for k in ("subject", "topic")}}
        schemas = [{"type": "function", "function": t["function"]} for t in tools]
        samples.append(Sample(prompt=[
            {"role": "system", "content": "Solve the scientific question. Use the provided scientific tools when useful. "
             "You may call tools multiple times and use their returned results. Work only with files in the current "
             "working directory. This is a text-only session: returned figures are artifacts, not visible images. "
             "When finished, give a concise final answer in \\boxed{} without a tool call."},
            {"role": "user", "content": row["question"]}],
            metadata={"science_case": clean, "science_tools": schemas, "question": row["question"]}))
    if len(samples) != len(set(selected)):
        raise ValueError("Requested case IDs missing from upstream data")
    return samples
