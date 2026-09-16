"""Small remote two-turn correctness/lifecycle probe, NOT a throughput test."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid


def output_ids(response):
    raw = response.get("output_ids")
    if raw is None:
        rows = response.get("meta_info", {}).get("output_token_logprobs")
        if not rows:
            raise RuntimeError("server did not return exact generated token IDs; refusing text retokenization")
        raw = [row[1] for row in rows]
    if not raw or any(type(token) is not int or token < 0 for token in raw):
        raise RuntimeError("invalid generated token IDs")
    return raw


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict) or "error" in result:
        raise RuntimeError("generation failed: " + repr(result))
    return result


def run(cfg, output):
    sys.path.insert(0, str(Path(cfg["slime_root"]) / "examples/pd"))
    from transformers import AutoTokenizer
    from agentic_kv_request import (
        add_agentic_kv_metadata, build_agentic_extra_key,
        confirm_agentic_generation_final, confirm_agentic_generation_tool,
    )
    from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
    from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration
    from sglang.srt.disaggregation.agentic_host_staging import SharedHostStagingLedger
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=False)
    root = Path(cfg["control_root"]) / cfg["run_id"]
    store = AgenticEarlyClaimStore(str(root / "early-claims"))
    ledger = SharedHostStagingLedger(str(root / "d2p.json"))
    router_node = next(n for n in cfg["nodes"] if n["node_id"] == cfg["router"]["node_id"])
    url = "http://{}:{}/generate".format(router_node["host_ip"], cfg["router"]["port"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    def persist():
        (output / "smoke.json").write_text(json.dumps({"cases": records, "performance_test": False}, indent=2))
    def generate(ids, metadata, generation, tool_output=False):
        params = {"temperature": 0, "max_new_tokens": 32 if tool_output else 16}
        if tool_output:
            params.update(regex="TOOL", stop=["TOOL"], no_stop_trim=True)
        params, request_id = add_agentic_kv_metadata(
            params, trajectory_metadata=metadata, generation=generation, tokenizer=tokenizer,
            tool_type="synthetic-smoke", tool_suffix_markers=["TOOL"], terminal_markers=[])
        payload = {"input_ids": ids, "sampling_params": params,
                   "rid": "{}-g{}".format(request_id, generation),
                   "extra_key": build_agentic_extra_key(request_id, params),
                   # At most32 generated tokens, output IDs only; no full prompt logits.
                   "return_logprob": True, "logprob_start_len": -1, "stream": False}
        return post(url, payload)
    for name, delay in (("direct", 0.0), ("slow", cfg["fast_tool_seconds"] + 2.0)):
        metadata = {"agentic_request_id": "multinode-smoke-" + uuid.uuid4().hex}
        case = {"case": name, "tool_delay_seconds": delay, "request_id": metadata["agentic_request_id"], "passed": False}
        records.append(case)
        persist()
        last_generation = 0
        try:
            text = ("This is a synthetic cache-transfer probe. The facts remain unchanged. " * 64
                    + "\nTo request the calculation, output TOOL exactly.\n")
            first_ids = tokenizer.encode(text, add_special_tokens=False)
            first = generate(first_ids, metadata, 0, tool_output=True)
            case["generation0"] = first
            first_output = output_ids(first)
            if "TOOL" not in tokenizer.decode(first_output):
                raise RuntimeError("constrained first turn did not produce TOOL; no tool ACK issued")
            if not confirm_agentic_generation_tool(metadata, 0, p_ready_dir=str(root)):
                raise RuntimeError("could not persist tool ACK")
            time.sleep(delay)
            suffix = tokenizer.encode("\nTool result: 2 + 3 = 5.\nThe answer is", add_special_tokens=False)
            next_ids = first_ids + first_output + suffix
            last_generation = 1
            second = generate(next_ids, metadata, 1)
            case["generation1"] = second
            case["input_tokens"] = len(next_ids)
            parent = RequestGeneration(metadata["agentic_request_id"], 0)
            case["route"] = store.read_route(parent, max_age_seconds=3600)
            case["host_entry"] = ledger.get(parent.snapshot_id)
            confirm_agentic_generation_final(metadata, 1, p_ready_dir=str(root))
            reference_metadata = {"agentic_request_id": "multinode-reference-" + uuid.uuid4().hex}
            try:
                reference = generate(next_ids, reference_metadata, 0)
                case["full_recompute_reference"] = reference
            finally:
                confirm_agentic_generation_final(reference_metadata, 0, p_ready_dir=str(root))
            case["exact_output_match"] = output_ids(second) == output_ids(reference)
            if not case["exact_output_match"]:
                case["output_mismatch_note"] = "Requires investigation: batch/kernel numerical differences or KV error; this probe alone does not identify the cause."
            # Last generated token may not have a KV entry; tail page can be recomputed.
            reusable = max(0, len(first_ids) + len(first_output) - 1)
            reusable = reusable // cfg["page_size"] * cfg["page_size"]
            case["expected_page_aligned_parent_tokens"] = reusable
            case["reported_cached_tokens"] = second.get("meta_info", {}).get("cached_tokens")
            expected_route = "direct_complete" if name == "direct" else "host_ready"
            case["expected_route"] = expected_route
            case["path_match"] = (case["route"] or {}).get("route") == expected_route
            cached = case["reported_cached_tokens"]
            case["cache_counter_check"] = isinstance(cached, int) and cached >= reusable
            case["passed"] = case["exact_output_match"] and case["path_match"] and case["cache_counter_check"]
            persist()
            if not case["passed"]:
                raise RuntimeError("{} smoke failed; inspect {}".format(name, output / "smoke.json"))
        except BaseException as exc:
            case["error"] = str(exc)
            persist()
            raise
        finally:
            for generation in range(last_generation + 1):
                confirm_agentic_generation_final(metadata, generation, p_ready_dir=str(root))
    print(json.dumps({"smoke_passed": True, "report": str(output / "smoke.json"),
                      "note": "Two-turn Direct/Slow only; P2D Host backpressure, TP fault injection and 300+1200s remain separate validation."}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output)
