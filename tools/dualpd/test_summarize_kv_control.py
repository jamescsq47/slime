import unittest

from summarize_kv_control import summarize, summarize_group_handoff, summarize_direct_cohort
from datetime import datetime, timezone


class ProgressTest(unittest.TestCase):
    def test_direct_cohort_includes_unstarted_failures_and_pending(self):
        def log(rank, event, sid, second=1):
            return f"[2026-09-21 01:00:{second:02} TP{rank}] AgenticKV {event} snapshot={sid}\n"
        d = [log(0, "fast_arrival_seen", sid) for sid in ("ok", "no", "partial", "pending", "conflict")]
        d += [log(0, "fast_arrival_seen", "ok", 2),
              log(0, "fast_arrival_seen", "outside", 0),
              log(0, "fast_arrival_seen", "outside", 2),
              log(0, "fast_arrival_seen", "multinode-smoke-x")]
        d += [log(0, "direct_fallback", sid, 5) for sid in ("no", "partial", "conflict")]
        p = [log(7, "early_direct_start", "partial", 2),
             log(0, "early_direct_group_complete", "ok", 4),
             log(0, "early_direct_admit", "ok", 5),
             log(0, "early_direct_admit", "conflict", 4)]
        start = datetime(2026, 9, 21, 1, 0, 1, tzinfo=timezone.utc).timestamp()
        r = summarize_direct_cohort(p, d, start, start + 2)
        self.assertEqual(r["eligible"], 5)
        self.assertEqual(r["direct_complete"], 1)
        self.assertEqual(r["fallback_no_rank_start"], 1)
        self.assertEqual(r["fallback_after_rank_start"], 1)
        self.assertEqual(r["pending"], ["pending"])
        self.assertEqual(r["conflicting_outcomes"], ["conflict"])
        self.assertEqual(r["success_fraction_of_all_arrivals"], .2)

    def test_received_then_cancelled_is_fallback_not_success_conflict(self):
        def log(event, sid, second):
            return f"[2026-09-21 01:00:{second:02} TP0] AgenticKV {event} snapshot={sid}\n"
        d = [log('fast_arrival_seen', sid, 1) for sid in ('retry', 'waiting')]
        d += [log('direct_fallback', 'retry', 5)]
        p = [log(event, sid, 3) for sid in ('retry', 'waiting')
             for event in ('early_direct_start', 'early_direct_group_complete')]
        start = datetime(2026, 9, 21, 1, 0, 1, tzinfo=timezone.utc).timestamp()
        r = summarize_direct_cohort(p, d, start, start + 1)
        self.assertEqual(r['direct_complete'], 0)
        self.assertEqual(r['fallback_after_rank_start'], 1)
        self.assertEqual(r['fallback_after_group_receive'], 1)
        self.assertEqual(r['pending'], ['waiting'])
        self.assertEqual(r['received_but_not_admitted_pending'], ['waiting'])
        self.assertEqual(r['conflicting_outcomes'], [])

    def test_empty_direct_cohort_is_not_a_zero_failure_success(self):
        self.assertIsNone(summarize_direct_cohort([], [], 0, 1)["success_fraction_of_all_arrivals"])

    def test_outcome_timings_do_not_hide_unstarted_or_follower_only_failures(self):
        def log(rank, event, sid, fields=""):
            return f"[2026-09-21 01:00:01 TP{rank}] AgenticKV {event} snapshot={sid} {fields}\n"
        d = [log(0, "fast_arrival_seen", sid) for sid in ("ok", "no", "follower", "failed")]
        d += [log(0, "direct_fallback", sid) for sid in ("no", "follower", "failed")]
        p = [log(0, "early_direct_start", "ok",
                 "arrival_to_start_ms=500 intent_to_grant_ms=200 grant_to_start_ms=100"),
             log(0, "early_direct_start", "ok", "arrival_to_start_ms=9999"),
             log(0, "early_direct_admit", "ok"),
             log(7, "early_direct_start", "follower", "claim_ms=50"),
             log(0, "early_direct_start", "failed", "claim_ms=150 receipt_echo_ms=-1")]
        start = datetime(2026, 9, 21, 1, 0, 1, tzinfo=timezone.utc).timestamp()
        result = summarize_direct_cohort(p, d, start, start + 1)
        self.assertEqual(result["fallback_no_rank_start"], 1)
        stages = result["tp0_start_timings_by_outcome"]
        self.assertEqual(stages["direct_complete"]["stage_ms"]["arrival_to_intent_ms"]["mean"], 200)
        self.assertEqual(stages["direct_complete"]["stage_ms"]["arrival_to_start_ms"]["mean"], 500)
        self.assertEqual(stages["fallback_after_rank_start"]["outcome_count"], 2)
        self.assertEqual(stages["fallback_after_rank_start"]["tp0_start_records"], 1)
        self.assertEqual(stages["fallback_after_rank_start"]["stage_ms"]["claim_ms"]["mean"], 150)
        self.assertNotIn("receipt_echo_ms", stages["fallback_after_rank_start"]["stage_ms"])

    def test_shards_smoke_duplicates_and_unfinished_are_not_completions(self):
        lines = [
            "[2026-09-21 01:00:00 TP0] AgenticKV tp_host_selected snapshot=a\n",
            "[2026-09-21 01:00:01 TP0] AgenticKV shared_host_h2d_complete snapshot=a\n",
            "[2026-09-21 01:00:02 TP0] AgenticKV shared_host_group_commit_release snapshot=a\n",
            "[2026-09-21 01:00:02 TP0] AgenticKV shared_host_group_commit_release snapshot=a\n",
            "[2026-09-21 01:00:01 TP1] AgenticKV shared_host_h2d_complete snapshot=a\n",
            "[2026-09-21 01:00:03 TP0] AgenticKV shared_host_h2d_complete snapshot=b\n",
            "[2026-09-21 01:00:04 TP0] AgenticKV early_direct_start snapshot=multinode-smoke-x\n",
            "[2026-09-21 01:00:05 TP0] Prefill batch\n",
        ]
        result = summarize(lines)
        self.assertEqual(result["counts_total"]["shared_host_group_commit_release"], 1)
        self.assertEqual(result["counts_total"]["early_direct_start"], 0)
        self.assertEqual(result["copy_to_release_seconds"]["mean"], 1)
        self.assertEqual(result["copy_to_release_unmatched_age_seconds"]["mean"], 2)

    def test_empty_log_is_not_success(self):
        self.assertIn("error", summarize([]))

    def test_direct_allocator_and_receipt_stages_exclude_missing_values(self):
        result = summarize([
            "[2026-09-21 01:00:01 TP0] AgenticKV early_direct_start snapshot=a "
            "intent_to_plan_ms=100 plan_to_service_ms=2 service_to_grant_ms=1 "
            "grant_to_receipt_ms=3 receipt_echo_ms=25 receipt_to_claim_ms=4\n",
            "[2026-09-21 01:00:02 TP0] AgenticKV early_direct_start snapshot=b "
            "receipt_echo_ms=-1\n",
        ])
        stages = result['direct_stage_ms']
        self.assertEqual(stages['intent_to_plan_ms']['mean'], 100)
        self.assertEqual(stages['receipt_echo_ms']['mean'], 25)
        self.assertEqual(stages['receipt_echo_ms']['count'], 1)

    def test_host_control_timings_count_one_rank_and_window(self):
        result = summarize([
            "[2026-09-21 01:00:00 TP0] AgenticKV host_completion_timing snapshot=old copy_to_loaded_ack_ms=900\n",
            "[2026-09-21 01:00:10 TP0] AgenticKV host_completion_timing snapshot=new copy_to_loaded_ack_ms=4.5\n",
            "[2026-09-21 01:00:10 TP1] AgenticKV host_completion_timing snapshot=new copy_to_loaded_ack_ms=3\n",
        ], window=5)
        timing = result["host_completion_stage_ms"]["copy_to_loaded_ack_ms"]
        self.assertEqual(timing["count"], 1)
        self.assertEqual(timing["mean"], 4.5)

    def test_group_latency_starts_after_last_shard(self):
        lines = [
            "[2026-09-21 01:00:01 TP0] AgenticKV shared_host_h2d_complete snapshot=a\n",
            "[2026-09-21 01:00:03 TP1] AgenticKV shared_host_h2d_complete snapshot=a\n",
            "[2026-09-21 01:00:05 TP0] AgenticKV shared_host_group_commit_release snapshot=a\n",
            "[2026-09-21 01:00:06 TP1] AgenticKV shared_host_group_commit_release snapshot=a\n",
            "[2026-09-21 01:00:06 TP0] AgenticKV shared_host_group_commit_release snapshot=b\n",
        ]
        result = summarize_group_handoff(lines, 2)
        self.assertEqual(result["copy_rank_skew_seconds"]["mean"], 2)
        self.assertEqual(result["all_copy_to_first_release_seconds"]["mean"], 2)
        self.assertEqual(result["all_copy_to_all_release_seconds"]["mean"], 3)
        self.assertEqual(result["incomplete_release_cohorts"], 1)

    def test_retried_inverted_group_not_reported_as_negative_latency(self):
        lines = [
            "[2026-09-21 01:00:01 TP0] AgenticKV shared_host_group_commit_release snapshot=a\n",
            "[2026-09-21 01:00:03 TP0] AgenticKV shared_host_h2d_complete snapshot=a\n",
        ]
        result = summarize_group_handoff(lines, 1)
        self.assertEqual(result["inverted_or_retried_cohorts"], 1)
        self.assertEqual(result["all_copy_to_all_release_seconds"]["count"], 0)
