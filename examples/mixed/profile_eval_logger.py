import json
import os
import statistics
from collections import defaultdict


def _sample_attr(sample, name, default=None):
    if isinstance(sample, dict):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _reward_value(sample, key, default=None):
    reward = _sample_attr(sample, "reward", None)
    if isinstance(reward, dict):
        return reward.get(key, default)
    return reward if reward is not None else default


def _quantile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _summary(values):
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def _safe_status(sample):
    status = _sample_attr(sample, "status", "")
    return getattr(status, "name", str(status))


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_eval_rollout_data(rollout_id, args, data, extra_metrics=None) -> bool:
    """Write per-sample and per-prompt-group timing profiles for eval rollout.

    The default eval logger still runs because this function returns False.
    """

    output_dir = os.environ.get(
        "SLIME_PROFILE_OUTPUT_DIR",
        "/workspace/slime/examples/hybrid/debug/profile",
    )
    os.makedirs(output_dir, exist_ok=True)

    all_summary = {}
    group_size_default = getattr(args, "n_samples_per_eval_prompt", None)
    if group_size_default is None:
        group_size_default = getattr(args, "n_samples_per_prompt", 1)
    eval_group_sizes = {
        cfg.name: cfg.n_samples_per_eval_prompt
        for cfg in getattr(args, "eval_datasets", []) or []
        if getattr(cfg, "n_samples_per_eval_prompt", None) is not None
    }

    for dataset_name, dataset_data in data.items():
        samples = dataset_data.get("samples") or []
        group_size = int(eval_group_sizes.get(dataset_name) or group_size_default or 1)
        group_size = max(1, group_size)

        sample_rows = []
        groups = defaultdict(list)
        for sample in sorted(samples, key=lambda s: _sample_attr(s, "index", 0)):
            sample_index = int(_sample_attr(sample, "index", 0) or 0)
            group_index = sample_index // group_size
            sample_in_group = sample_index % group_size
            sample_time = float(_sample_attr(sample, "sample_time", 0.0) or 0.0)
            tool_time = float(_sample_attr(sample, "tool_time", 0.0) or 0.0)
            row = {
                "rollout_id": rollout_id,
                "dataset": dataset_name,
                "group_index": group_index,
                "sample_index": sample_index,
                "sample_in_group": sample_in_group,
                "sample_time": sample_time,
                "tool_time": tool_time,
                "model_time": max(sample_time - tool_time, 0.0),
                "tool_time_ratio": tool_time / sample_time if sample_time > 0 else 0.0,
                "response_length": int(_sample_attr(sample, "response_length", 0) or 0),
                "tool_call_count": int(_sample_attr(sample, "tool_call_count", 0) or 0),
                "code_call_count": int(_sample_attr(sample, "code_call_count", 0) or 0),
                "search_call_count": int(_sample_attr(sample, "search_call_count", 0) or 0),
                "open_page_call_count": int(_sample_attr(sample, "open_page_call_count", 0) or 0),
                "finish_call_count": int(_sample_attr(sample, "finish_call_count", 0) or 0),
                "tool_token_count": int(_sample_attr(sample, "tool_token_count", 0) or 0),
                "acc": bool(_reward_value(sample, "acc", False)),
                "score": _reward_value(sample, "score", None),
                "pred": _reward_value(sample, "pred", None),
                "status": _safe_status(sample),
            }
            sample_rows.append(row)
            groups[group_index].append(row)

        group_rows = []
        for group_index, rows in sorted(groups.items()):
            sample_times = [row["sample_time"] for row in rows]
            tool_times = [row["tool_time"] for row in rows]
            response_lengths = [row["response_length"] for row in rows]
            tool_call_counts = [row["tool_call_count"] for row in rows]
            search_call_counts = [row["search_call_count"] for row in rows]
            open_page_call_counts = [row["open_page_call_count"] for row in rows]
            finish_call_counts = [row["finish_call_count"] for row in rows]
            tool_token_counts = [row["tool_token_count"] for row in rows]
            group_rows.append(
                {
                    "rollout_id": rollout_id,
                    "dataset": dataset_name,
                    "group_index": group_index,
                    "group_size": len(rows),
                    "group_time_max": max(sample_times) if sample_times else 0.0,
                    "group_time_mean": statistics.mean(sample_times) if sample_times else 0.0,
                    "group_time_median": statistics.median(sample_times) if sample_times else 0.0,
                    "group_time_min": min(sample_times) if sample_times else 0.0,
                    "tool_time_max": max(tool_times) if tool_times else 0.0,
                    "tool_time_mean": statistics.mean(tool_times) if tool_times else 0.0,
                    "response_length_max": max(response_lengths) if response_lengths else 0,
                    "response_length_mean": statistics.mean(response_lengths) if response_lengths else 0.0,
                    "tool_call_count_max": max(tool_call_counts) if tool_call_counts else 0,
                    "tool_call_count_mean": statistics.mean(tool_call_counts) if tool_call_counts else 0.0,
                    "search_call_count_max": max(search_call_counts) if search_call_counts else 0,
                    "search_call_count_mean": statistics.mean(search_call_counts) if search_call_counts else 0.0,
                    "open_page_call_count_max": max(open_page_call_counts) if open_page_call_counts else 0,
                    "open_page_call_count_mean": statistics.mean(open_page_call_counts) if open_page_call_counts else 0.0,
                    "finish_call_count_max": max(finish_call_counts) if finish_call_counts else 0,
                    "finish_call_count_mean": statistics.mean(finish_call_counts) if finish_call_counts else 0.0,
                    "tool_token_count_max": max(tool_token_counts) if tool_token_counts else 0,
                    "tool_token_count_mean": statistics.mean(tool_token_counts) if tool_token_counts else 0.0,
                    "acc_mean": statistics.mean([1.0 if row["acc"] else 0.0 for row in rows]) if rows else 0.0,
                    "completed": sum(row["status"].endswith("COMPLETED") or row["status"].lower() == "completed" for row in rows),
                    "truncated": sum("TRUNCATED" in row["status"].upper() for row in rows),
                }
            )

        sample_path = os.path.join(output_dir, f"{dataset_name}_samples_rollout{rollout_id}.jsonl")
        group_path = os.path.join(output_dir, f"{dataset_name}_groups_rollout{rollout_id}.jsonl")
        _write_jsonl(sample_path, sample_rows)
        _write_jsonl(group_path, group_rows)

        all_summary[dataset_name] = {
            "sample_count": len(sample_rows),
            "group_count": len(group_rows),
            "sample_time": _summary([row["sample_time"] for row in sample_rows]),
            "tool_time": _summary([row["tool_time"] for row in sample_rows]),
            "group_time_max": _summary([row["group_time_max"] for row in group_rows]),
            "group_time_mean": _summary([row["group_time_mean"] for row in group_rows]),
            "response_length": _summary([row["response_length"] for row in sample_rows]),
            "tool_call_count": _summary([row["tool_call_count"] for row in sample_rows]),
            "search_call_count": _summary([row["search_call_count"] for row in sample_rows]),
            "open_page_call_count": _summary([row["open_page_call_count"] for row in sample_rows]),
            "finish_call_count": _summary([row["finish_call_count"] for row in sample_rows]),
            "tool_token_count": _summary([row["tool_token_count"] for row in sample_rows]),
            "sample_path": sample_path,
            "group_path": group_path,
        }

    summary_path = os.path.join(output_dir, f"profile_summary_rollout{rollout_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print(f"[profile_eval_logger] wrote profile summary to {summary_path}", flush=True)
    return False
