"""Small remote two-turn correctness/lifecycle probe, NOT a throughput test."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid
import re
from concurrent.futures import ThreadPoolExecutor


def committed_path_ranks(cfg, snapshot_id, path):
    """Single-P Direct need not publish a route; use physical commit evidence.

    This engineering launcher puts Router and P on the same node. Count unique
    TP ranks with actual admission/Host completion, never marker discovery.
    """
    prefill = next(n for n in cfg['nodes'] if n['role'] == 'prefill')
    if prefill['node_id'] != cfg['router']['node_id']:
        return []
    logfile = Path(cfg['local_root']) / cfg['run_id'] / prefill['engine_id'] / 'service.log'
    if not logfile.exists():
        return []
    event = 'early_direct_admit' if path == 'direct' else 'shared_host_h2d_complete'
    ranks = set()
    for line in logfile.read_text().splitlines():
        if f'AgenticKV {event} snapshot={snapshot_id} ' in line:
            match = re.search(r' TP(\d+)\]', line)
            if match:
                ranks.add(int(match.group(1)))
    return sorted(ranks)


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
    from sglang.srt.disaggregation.agentic_host_staging import (
        SharedHostStagingLedger, create_agentic_node_local_metadata_backend,
    )
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
    def generate(ids, metadata, generation, tool_output=False, overrides=None):
        params = {"temperature": 0, "max_new_tokens": 32 if tool_output else 16}
        params.update(overrides or {})
        if tool_output:
            params.update(regex="TOOL", stop=["TOOL"], no_stop_trim=True)
        params, request_id = add_agentic_kv_metadata(
            params, trajectory_metadata=metadata, generation=generation, tokenizer=tokenizer,
            tool_type="synthetic-smoke", tool_suffix_markers=["TOOL"], terminal_markers=[])
        payload = {"input_ids": ids, "sampling_params": params,
                   "rid": "{}-g{}".format(request_id, generation),
                   "extra_key": build_agentic_extra_key(request_id, params),
                   # P2D Host deliberately excludes logprob payloads. /generate
                   # returns exact output_ids in this engine; never retokenize.
                   "return_logprob": not cfg.get('p2d_host_probe', False),
                   "logprob_start_len": -1, "stream": False}
        return post(url, payload)
    if cfg.get('p2d_host_probe'):
        # Two 8k prompts cannot fit in the diagnostic 12k D pool. Keep the
        # first decoding while the second reaches P-ready. Real admission
        # pressure must trigger Host; do not fabricate an offer or receipt.
        prefill = next(n for n in cfg['nodes'] if n['role'] == 'prefill')
        log = Path(cfg['local_root']) / cfg['run_id'] / prefill['engine_id'] / 'service.log'
        base = tokenizer.encode('Count the numbers carefully. ' * 2000, add_special_tokens=False)
        ids = base[:8192]
        if len(ids) != 8192:
            raise RuntimeError('P2D Host probe must contain exactly8192tokens')
        metas = [{'agentic_request_id': 'p2d-host-' + uuid.uuid4().hex} for _ in range(3)]
        case = {'case': 'p2d_host', 'request_id': metas[1]['agentic_request_id'], 'passed': False}
        records.append(case)
        persist()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                blocker = pool.submit(generate, ids, metas[0], 0, overrides={
                    'max_new_tokens': 512, 'ignore_eos': True})
                blocker.add_done_callback(lambda _: confirm_agentic_generation_final(
                    metas[0], 0, p_ready_dir=str(root)))
                deadline = time.monotonic() + 90
                while True:
                    lines = log.read_text().splitlines() if log.exists() else []
                    if any('AgenticKV p_to_d_release ' in line and
                           'extra_key=agentic-v1:' + metas[0]['agentic_request_id'] + ':g0' in line
                           for line in lines):
                        break
                    if blocker.done():
                        blocker.result()
                        raise RuntimeError('blocker finished before P2D Host probe')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('blocker did not reach D')
                    time.sleep(0.1)
                staged = generate(ids, metas[1], 0)
                case['generation0'] = staged
                confirm_agentic_generation_final(metas[1], 0, p_ready_dir=str(root))
                blocker.result(timeout=180)
                confirm_agentic_generation_final(metas[0], 0, p_ready_dir=str(root))
            reference = generate(ids, metas[2], 0)
            case['full_recompute_reference'] = reference
            case['exact_output_match'] = output_ids(staged) == output_ids(reference)
            matches = re.findall(r'TP(\d+)\] AgenticKV p2d_host_d2h_complete snapshot=(\S+)', log.read_text())
            host_ranks = {}
            for rank, snapshot in matches:
                host_ranks.setdefault(snapshot, set()).add(int(rank))
            case['host_snapshot_ranks'] = {s: sorted(ranks) for s, ranks in host_ranks.items()}
            case['passed'] = case['exact_output_match'] and any(
                ranks == set(range(cfg['tp_size'])) for ranks in host_ranks.values())
            persist()
            if not case['passed']:
                raise RuntimeError('P2D Host probe failed; inspect smoke.json and both worker logs')
        except BaseException as exc:
            case['error'] = str(exc)
            persist()
            raise
        finally:
            for meta in metas:
                confirm_agentic_generation_final(meta, 0, p_ready_dir=str(root))
    fallback_case = "slow" if cfg.get("d2p_host_staging", True) else "recompute"
    if cfg.get("d2p_direct_wait_only", False):
        fallback_case = "direct"
    for name, delay in (("direct", 0.0), (fallback_case, cfg["fast_tool_seconds"] + 2.0)):
        metadata = {"agentic_request_id": "multinode-smoke-" + uuid.uuid4().hex}
        case = {"case": name, "tool_delay_seconds": delay, "request_id": metadata["agentic_request_id"], "passed": False}
        records.append(case)
        persist()
        last_generation = 0
        try:
            text = ("This is a synthetic cache-transfer probe. The facts remain unchanged. " * 64
                    + "\nTo request the calculation, output TOOL exactly.\n")
            first_ids = tokenizer.encode(text, add_special_tokens=False)
            if cfg.get('model_family') == 'qwen35_moe':
                # Make the full-reference pass use the same first Prefill
                # chunk as the saved checkpoint. Arbitrary dense-vs-incremental
                # GDN chunking can introduce BF16 numerical differences.
                chunk = cfg['chunked_prefill_size']
                first_ids = (first_ids * ((chunk + len(first_ids) - 1) // len(first_ids)))[:chunk]
                case['reference_chunk_matched'] = True
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
            parent = RequestGeneration(metadata["agentic_request_id"], 0)
            if name == "recompute":
                snapshots = create_agentic_node_local_metadata_backend().agentic_snapshot_store()
                deadline = time.monotonic() + 30
                while True:
                    manifest = snapshots.load(parent, require_ready=False)
                    if manifest is not None and manifest.state.value == "failed":
                        case['terminal_manifest'] = {'state': manifest.state.value,
                                                     'reason': manifest.failure_reason}
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Direct-only fallback did not publish FAILED')
                    time.sleep(0.1)
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
            if cfg.get("model_family") == "qwen35_moe":
                # Request-owned hybrid state freezes the prompt checkpoint,
                # not the generated tail. This synthetic prompt has no think
                # opener/template rewrite; its stable prefix is first_ids.
                reusable = len(first_ids)
            reusable = reusable // cfg["page_size"] * cfg["page_size"]
            case["expected_page_aligned_parent_tokens"] = reusable
            case["reported_cached_tokens"] = second.get("meta_info", {}).get("cached_tokens")
            expected_route = {"direct": "direct_complete", "slow": "host_ready",
                              "recompute": "recompute"}[name]
            case["expected_route"] = expected_route
            case['committed_path_ranks'] = (committed_path_ranks(cfg, parent.snapshot_id, name)
                                            if name != "recompute" else [])
            case["path_match"] = ((case["route"] or {}).get("route") == expected_route
                                  or case['committed_path_ranks'] == list(range(cfg['tp_size'])))
            cached = case["reported_cached_tokens"]
            case["cache_counter_check"] = (isinstance(cached, int) and
                                           (cached == 0 if name == "recompute" else cached >= reusable))
            if name == "recompute":
                # Single-P mode need not publish a Router route. FAILED is
                # the authoritative producer fence for explicit recompute.
                case["path_match"] = (case['terminal_manifest']['state'] == 'failed'
                                      and case['terminal_manifest']['reason'] == 'shared_host_staging_unavailable'
                                      and case["host_entry"] is None)
            case["passed"] = case["exact_output_match"] and case["path_match"] and case["cache_counter_check"]
            if cfg.get("d2p_direct_wait_only", False):
                case["passed"] = case["passed"] and case["host_entry"] is None
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
