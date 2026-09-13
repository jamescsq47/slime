"""Colocated agent loop; scientific computation stays in upstream functions."""
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

from agentic_kv_request import lifecycle_enabled
from pd_metrics import sglang_meta_attrs
from slime.dashboard.api import span as dashboard_span
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample
from data.bfcl_web_search.harness import observation_tokens


def parse_calls(text):
    text = text.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if ("<tool_call>" in text and not blocks) or len(blocks) > 8:
        raise ValueError("Return complete tool-call JSON, at most 8 calls per turn")
    calls = [json.loads(b) for b in blocks]
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments", {}), dict):
            raise ValueError("Tool call needs a name and an arguments object")
    return calls


async def stop_worker(worker):
    if worker is not None:
        if worker.returncode is None:
            try:
                worker.kill()
            except ProcessLookupError:
                pass
        await worker.wait()


async def generate(args, sample, sampling_params):
    if lifecycle_enabled():
        raise RuntimeError("SciAgentGym is colocated-only; PD lifecycle not enabled")
    opts = getattr(args, "workload_dataset_options", {}).get(sample.metadata.get("dataset_id", "science"), {})
    tokenizer = GenerateState(args).tokenizer
    rendered = tokenizer.apply_chat_template(sample.prompt, tools=sample.metadata["science_tools"],
                                             tokenize=False, add_generation_prompt=True)
    initial = list(tokenizer.encode(rendered, add_special_tokens=False))
    response, events, token_turns = [], [], []
    sample.metadata["science_token_trace"] = {"initial_ids": initial, "turns": token_turns}
    turns = tool_tokens = 0
    status, reason = Sample.Status.TRUNCATED, "max_turns"
    started = time.monotonic()
    worker = None
    total_budget = int(opts.get("max_response_tokens", 32768))
    context = int(getattr(args, "max_seq_len", None) or 40960)
    with tempfile.TemporaryDirectory(prefix="science-agent-") as scratch:
        with open(Path(scratch) / "worker.log", "wb") as log:
            try:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                       "MPLBACKEND": "Agg", "MPLCONFIGDIR": scratch,
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                worker = await asyncio.create_subprocess_exec(
                    sys.executable, str(Path(__file__).with_name("worker.py")),
                    opts.get("upstream_dir", "/homes/siqic/data/SciAgentGYM"), cwd=scratch, env=env,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, limit=256000)

                async def rpc(value, timeout):
                    worker.stdin.write((json.dumps(value, ensure_ascii=False) + "\n").encode())
                    await worker.stdin.drain()
                    line = await asyncio.wait_for(worker.stdout.readline(), timeout)
                    if not line:
                        raise RuntimeError("Upstream tool worker exited")
                    return json.loads(line)

                ready = await rpc(sample.metadata["science_case"], 30)
                sample.metadata["tool_load_seconds"] = ready["load_seconds"]
                for turn in range(int(opts.get("max_turns", 10))):
                    remaining = min(total_budget - len(response), context - len(initial) - len(response) - 512)
                    if remaining <= 0:
                        reason = "budget"
                        break
                    params = {**sampling_params, "max_new_tokens": min(remaining, int(opts.get("max_tokens_per_turn", 8192)))}
                    with dashboard_span(args, sample, "generation_turn", attrs={"task_type": "science",
                                         "turn": turn + 1, "route_mode": "colocated"}) as span:
                        output = await post(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
                                            {"input_ids": initial + response, "sampling_params": params, "return_logprob": False})
                        span.update(sglang_meta_attrs(output["meta_info"]))
                    ids = output.get("output_ids")
                    if ids is None:
                        raise RuntimeError("Exact model output_ids required")
                    response.extend(ids)
                    token_turns.append({"output_ids": list(ids), "tool_suffix_ids": []})
                    turns += 1
                    finish = output["meta_info"]["finish_reason"]["type"]
                    if finish == "abort":
                        raise RuntimeError("Model generation aborted")
                    if finish == "length" or not ids:
                        reason = finish if ids else "empty_generation"
                        break
                    try:
                        calls = parse_calls(output["text"])
                    except (ValueError, TypeError) as exc:
                        calls = None
                        observations = [{"error": str(exc), "instruction": "Correct the tool-call JSON."}]
                    if calls == []:
                        reason, status = "final", Sample.Status.COMPLETED
                        sample.metadata["predicted_answer"] = output["text"].rsplit("</think>", 1)[-1].strip()
                        break
                    if calls:
                        observations = []
                        for call in calls:
                            t = time.monotonic()
                            try:
                                result = await rpc(call, float(opts.get("tool_timeout_seconds", 60)))
                            except BaseException as exc:
                                events.append({"tool": call["name"], "arguments": call.get("arguments", {}),
                                               "ok": False, "error": type(exc).__name__, "elapsed_seconds": time.monotonic() - t})
                                raise
                            events.append({"tool": call["name"], "arguments": call.get("arguments", {}),
                                           "elapsed_seconds": time.monotonic() - t, **result})
                            observations.append({"tool": call["name"], "observation": result["observation"],
                                                 "truncated": result.get("truncated", False)})
                    suffix = observation_tokens(tokenizer, observations)
                    if ids[-1] == getattr(tokenizer, "eos_token_id", None):
                        suffix = list(tokenizer.encode("\n", add_special_tokens=False)) + suffix
                    if len(suffix) > min(total_budget - len(response), context - len(initial) - len(response) - 512):
                        reason = "tool_context_budget"
                        break
                    response.extend(suffix)
                    token_turns[-1]["tool_suffix_ids"] = list(suffix)
                    tool_tokens += len(suffix)
            except BaseException as exc:
                reason, status = type(exc).__name__, Sample.Status.ABORTED
                sample.metadata["worker_log_tail"] = (Path(scratch) / "worker.log").read_text(errors="replace")[-4000:]
                raise
            finally:
                await stop_worker(worker)
                sample.metadata.update(num_turns=turns, stop_reason=reason, science_tool_events=events, official_score=False)
                sample.status = status
                sample.tokens = initial + response
                sample.response = tokenizer.decode(response, skip_special_tokens=False)
                sample.response_length = len(response)
                sample.sample_time = time.monotonic() - started
                sample.tool_time = sum(e["elapsed_seconds"] for e in events)
                sample.tool_token_count, sample.tool_call_count = tool_tokens, len(events)
                sample.code_call_count, sample.search_call_count = len(events), 0
    return sample
