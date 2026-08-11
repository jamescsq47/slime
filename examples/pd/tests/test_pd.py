from inference import (
    DynamicScheduler,
    _parse_engine_metrics,
    arrival_offsets,
    build_engine_timeseries,
    build_summary,
)
from pd_metrics import generation_turns, sglang_meta_attrs


def test_sglang_meta_attrs_derives_ttft_and_tpot():
    attrs = sglang_meta_attrs(
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "request_received_ts": 10.0,
            "prefill_finished_ts": 10.4,
            "decode_finished_ts": 11.2,
            "e2e_latency": 1.2,
            "inference_time": 0.8,
            "finish_reason": {"type": "stop"},
        }
    )
    assert abs(attrs["ttft_seconds"] - 0.4) < 1e-9
    assert abs(attrs["tpot_seconds"] - 0.2) < 1e-9


def test_generation_turns_keeps_only_closed_generation_spans():
    trace = {
        "events": [
            {"type": "span_start", "name": "generation_turn", "attrs": {"turn": 1}},
            {"type": "span_end", "name": "tool", "attrs": {"tool": "search"}},
            {"type": "span_end", "name": "generation_turn", "attrs": {"turn": 1, "ttft_seconds": 0.1}},
        ]
    }
    assert generation_turns(trace) == [{"turn": 1, "ttft_seconds": 0.1}]


def test_arrival_offsets_are_seeded_and_monotonic():
    first = arrival_offsets(8, 0.5, "poisson", 7)
    assert first == arrival_offsets(8, 0.5, "poisson", 7)
    assert first[0] == 0.0
    assert first == sorted(first)
    assert arrival_offsets(3, 0.5, "fixed", 7) == [0.0, 2.0, 4.0]


def test_dynamic_scheduler_routes_away_from_the_heavier_node():
    scheduler = DynamicScheduler(hysteresis=0.1, max_imbalance=8, max_consecutive=3, seed=7)
    choice, reason = scheduler.choose(
        {"math", "qa"},
        {
            "prefill_relative_activity": 1.4,
            "decode_relative_activity": 0.6,
            "relative_activity_ready": 1.0,
        },
    )
    assert (choice, reason) == ("math", "prefill_relatively_heavier")

    scheduler = DynamicScheduler(hysteresis=0.1, max_imbalance=8, max_consecutive=3, seed=7)
    choice, reason = scheduler.choose(
        {"math", "qa"},
        {
            "prefill_relative_activity": 0.5,
            "decode_relative_activity": 1.3,
            "relative_activity_ready": 1.0,
        },
    )
    assert (choice, reason) == ("qa", "decode_relatively_heavier")


def test_dynamic_scheduler_does_not_override_the_pressure_signal():
    scheduler = DynamicScheduler(hysteresis=0.1, max_imbalance=3, max_consecutive=3, seed=7)
    choices = [
        scheduler.choose(
            {"math", "qa"},
            {
                "prefill_relative_activity": 2.0,
                "decode_relative_activity": 0.5,
                "relative_activity_ready": 1.0,
            },
        )[0]
        for _ in range(20)
    ]
    assert choices == ["math"] * 20


def test_dynamic_scheduler_cold_start_is_deterministically_balanced():
    scheduler = DynamicScheduler(hysteresis=0.1, max_imbalance=20, max_consecutive=3, seed=7)
    choices = [
        scheduler.choose(
            {"math", "qa"},
            {
                "prefill_relative_activity": 0.0,
                "decode_relative_activity": 0.0,
                "relative_activity_ready": 0.0,
            },
        )[0]
        for _ in range(12)
    ]
    assert "qqqq" not in "".join("m" if choice == "math" else "q" for choice in choices)
    assert "mmmm" not in "".join("m" if choice == "math" else "q" for choice in choices)


def test_parse_engine_metrics_and_summary_throughput():
    text = """
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="qwen"} 120
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="qwen"} 30
# TYPE sglang:realtime_tokens_total counter
sglang:realtime_tokens_total{model_name="qwen",mode="prefill_compute"} 90
sglang:realtime_tokens_total{model_name="qwen",mode="decode"} 25
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="qwen"} 24
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_bucket{le="0.1"} 1
sglang:time_to_first_token_seconds_bucket{le="0.5"} 2
sglang:time_to_first_token_seconds_bucket{le="+Inf"} 2
sglang:time_to_first_token_seconds_count 2
sglang:time_to_first_token_seconds_sum 0.6
"""
    assert _parse_engine_metrics(text) == {
        "sglang_prompt_tokens_total": 120.0,
        "sglang_generation_tokens_total": 30.0,
        "sglang_realtime_tokens_total|mode=prefill_compute": 90.0,
        "sglang_realtime_tokens_total|mode=decode": 25.0,
        "sglang_gen_throughput": 24.0,
        "sglang_time_to_first_token_seconds_bucket|le=0.1": 1.0,
        "sglang_time_to_first_token_seconds_bucket|le=0.5": 2.0,
        "sglang_time_to_first_token_seconds_bucket|le=+Inf": 2.0,
        "sglang_time_to_first_token_seconds_count": 2.0,
        "sglang_time_to_first_token_seconds_sum": 0.6,
    }
    requests = [
        {
            "task_type": "math",
            "status": "completed",
            "queue_delay_seconds": 0.0,
            "agent_latency_seconds": 2.0,
            "first_turn_ttft_seconds": 0.2,
            "turn_metrics": [{"ttft_seconds": 0.2, "tpot_seconds": 0.05}],
        }
    ]
    metrics = [
        {"ts": 1.0, "role": "prefill", "metrics": {"sglang_prompt_tokens_total": 100}},
        {"ts": 3.0, "role": "prefill", "metrics": {"sglang_prompt_tokens_total": 300}},
        {
            "ts": 1.0,
            "role": "decode",
            "metrics": {
                "sglang_generation_tokens_total": 40,
                "sglang_time_to_first_token_seconds_sum": 1.0,
                "sglang_time_to_first_token_seconds_count": 4,
                "sglang_time_to_first_token_seconds_bucket|le=0.1": 1,
                "sglang_time_to_first_token_seconds_bucket|le=0.5": 4,
            },
        },
        {
            "ts": 3.0,
            "role": "decode",
            "metrics": {
                "sglang_generation_tokens_total": 100,
                "sglang_time_to_first_token_seconds_sum": 1.6,
                "sglang_time_to_first_token_seconds_count": 6,
                "sglang_time_to_first_token_seconds_bucket|le=0.1": 2,
                "sglang_time_to_first_token_seconds_bucket|le=0.5": 6,
            },
        },
    ]
    summary = build_summary(requests, metrics, 0.5)
    assert summary["prefill_prompt_tokens_per_second"] == 100.0
    assert summary["decode_generation_tokens_per_second"] == 30.0
    assert abs(summary["engine_ttft_seconds"]["mean"] - 0.3) < 1e-9
    assert summary["engine_ttft_seconds"]["p50"] == 0.1


def test_engine_timeseries_prefers_scheduler_realtime_token_counters():
    rows = build_engine_timeseries(
        [
            {
                "ts": 1.0,
                "role": "decode",
                "metrics": {
                    "sglang_generation_tokens_total": 100,
                    "sglang_realtime_tokens_total|mode=decode": 1000,
                    "sglang_gen_throughput": 450,
                },
            },
            {
                "ts": 3.0,
                "role": "decode",
                "metrics": {
                    "sglang_generation_tokens_total": 120,
                    "sglang_realtime_tokens_total|mode=decode": 1900,
                    "sglang_gen_throughput": 450,
                },
            },
        ]
    )
    assert rows[0]["generation_tokens_per_second"] == 450.0
    assert rows[0]["completion_accounted_generation_tokens_per_second"] == 10.0
    assert rows[0]["scheduler_gen_throughput"] == 450.0
