"""Read a local P service log; report logical TP0 control progress, not throughput.

One snapshot is counted once per event. Completed latency cohorts exclude still
pending work; its count and age are reported separately. Timestamps in old logs
have one-second resolution. This performs no control RPCs and changes no state.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import re
import statistics


EVENTS = (
    "early_direct_start", "early_direct_group_complete", "tp_host_selected",
    "shared_host_h2d_prepared", "shared_host_h2d_complete",
    "shared_host_group_commit_release",
)
HEADER = re.compile(r"^\[([^]]+) TP0\]")
SNAPSHOT = re.compile(r"\bsnapshot=(\S+)")
RANK_HEADER = re.compile(r"^\[([^]]+) TP(\d+)\]")


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": statistics.mean(values),
            "p90": values[min(len(values) - 1, int(.9 * len(values)))],
            "max": values[-1]}


def summarize(lines, window=300):
    events = defaultdict(dict)
    timings = defaultdict(list)
    host_timings = defaultdict(list)
    end = None
    for line in lines:
        header = HEADER.match(line)
        if not header:
            continue
        try:
            at = datetime.fromisoformat(header[1]).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
        end = at if end is None else max(end, at)
        snapshot = SNAPSHOT.search(line)
        if not snapshot or snapshot[1].startswith("multinode-smoke-"):
            continue
        if "AgenticKV host_completion_timing " in line:
            for key, value in re.findall(r"\b(\w+_ms)=([0-9.]+)", line):
                host_timings[key].append((at, float(value)))
        for event in EVENTS:
            if "AgenticKV " + event + " " in line:
                events[event][snapshot[1]] = at
                if event == "early_direct_start":
                    for key in ("arrival_to_start_ms", "intent_to_grant_ms", "grant_to_start_ms",
                                "claim_ms", "metadata_ms", "intent_to_plan_ms",
                                "plan_to_service_ms", "service_to_grant_ms",
                                "grant_to_receipt_ms", "receipt_echo_ms", "receipt_to_claim_ms"):
                        match = re.search(r"\b" + key + r"=([0-9.]+)", line)
                        if match:
                            timings[key].append((at, float(match[1])))
                break
    if end is None:
        return {"error": "no timestamped TP0 records"}
    start = end - window
    result = {
        "window_end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "window_seconds": window,
        "counts_total": {e: len(events[e]) for e in EVENTS},
        "counts_window": {e: sum(t >= start for t in events[e].values()) for e in EVENTS},
        "direct_stage_ms": {k: distribution(v for t, v in values if t >= start)
                            for k, values in timings.items()},
        "host_completion_stage_ms": {k: distribution(v for t, v in values if t >= start)
                                     for k, values in host_timings.items()},
    }
    for name, before, after in (
        ("selected_to_copy", "tp_host_selected", "shared_host_h2d_complete"),
        ("copy_to_release", "shared_host_h2d_complete", "shared_host_group_commit_release"),
    ):
        result[name + "_seconds"] = distribution(
            t - events[before][sid] for sid, t in events[after].items()
            if t >= start and sid in events[before] and t >= events[before][sid])
        result[name + "_unmatched_age_seconds"] = distribution(
            end - t for sid, t in events[before].items() if sid not in events[after])
    return result


def summarize_group_handoff(lines, tp_size, window=300):
    """Diagnostic snapshot cohorts, not an attempt-level correctness check.

    Existing logs lack a common attempt ID on these two events. Keep the latest
    event per rank, reject inverted cohorts and report incomplete groups instead
    of interpreting one shard's completion as a TP-wide completion.
    """
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    events = defaultdict(lambda: defaultdict(dict))
    end = None
    names = ("shared_host_h2d_complete", "shared_host_group_commit_release")
    for line in lines:
        header, snapshot = RANK_HEADER.match(line), SNAPSHOT.search(line)
        if not header:
            continue
        try:
            at = datetime.fromisoformat(header[1]).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
        end = at if end is None else max(end, at)
        if not snapshot or snapshot[1].startswith("multinode-smoke-"):
            continue
        rank = int(header[2])
        for event in names:
            if "AgenticKV " + event + " " in line:
                events[event][snapshot[1]][rank] = at
                break
    if end is None:
        return {"error": "no timestamped rank records"}
    copies, releases = (events[name] for name in names)
    expected = set(range(tp_size))
    skew, first_release, last_release = [], [], []
    incomplete = inverted = 0
    for sid, ranks in releases.items():
        if max(ranks.values()) < end - window:
            continue
        copied = copies.get(sid, {})
        if set(ranks) != expected or set(copied) != expected:
            incomplete += 1
            continue
        copied_at = max(copied.values())
        if min(ranks.values()) < copied_at:
            inverted += 1
            continue
        skew.append(copied_at - min(copied.values()))
        first_release.append(min(ranks.values()) - copied_at)
        last_release.append(max(ranks.values()) - copied_at)
    return {
        "tp_size": tp_size,
        "copy_rank_skew_seconds": distribution(skew),
        "all_copy_to_first_release_seconds": distribution(first_release),
        "all_copy_to_all_release_seconds": distribution(last_release),
        "incomplete_release_cohorts": incomplete,
        "inverted_or_retried_cohorts": inverted,
        "copy_without_any_release": len(set(copies) - set(releases)),
    }


def summarize_direct_cohort(p_lines, d_lines, start, end):
    """Join unique fast-arrival generations to settled outcomes, including failures.

    The cohort is selected using D's first fast-arrival timestamp; P start on
    ANY rank is evidence of a start. No cross-node latency is calculated.
    Repeated log events are deduplicated; inconsistent success/fallback
    outcomes are exposed. This snapshot-level diagnostic is not attempt-level
    correctness proof. Success requires the native Direct ADMIT, not the
    earlier reversible P_RECEIVED event. Pending is not failure.
    """
    arrivals, starts, received, complete, fallback = {}, set(), set(), set(), set()
    start_timings = {}
    for lines, side in ((d_lines, "D"), (p_lines, "P")):
        for line in lines:
            header, snapshot = RANK_HEADER.match(line), SNAPSHOT.search(line)
            if not header or not snapshot or snapshot[1].startswith("multinode-smoke-"):
                continue
            sid = snapshot[1]
            if side == "P":
                if "AgenticKV early_direct_start " in line:
                    starts.add(sid)
                    if int(header[2]) == 0 and sid not in start_timings:
                        # Keep one actual start record, never mix fields from
                        # different attempts/ranks or turn missing data into 0.
                        values = {key: float(value) for key, value in
                                  re.findall(r"\b(\w+_ms)=([0-9.]+)", line)}
                        if all(key in values for key in (
                                "arrival_to_start_ms", "intent_to_grant_ms", "grant_to_start_ms")):
                            values["arrival_to_intent_ms"] = (
                                values["arrival_to_start_ms"] - values["intent_to_grant_ms"]
                                - values["grant_to_start_ms"])
                        start_timings[sid] = values
                if int(header[2]) == 0 and "AgenticKV early_direct_group_complete " in line:
                    received.add(sid)
                if int(header[2]) == 0 and "AgenticKV early_direct_admit " in line:
                    complete.add(sid)
            elif int(header[2]) == 0:
                if "AgenticKV fast_arrival_seen " in line:
                    try:
                        at = datetime.fromisoformat(header[1]).replace(tzinfo=timezone.utc).timestamp()
                    except ValueError:
                        continue
                    arrivals[sid] = min(at, arrivals.get(sid, at))
                if "AgenticKV direct_fallback " in line:
                    fallback.add(sid)
    cohort = {sid for sid, at in arrivals.items() if start <= at < end}
    conflict = cohort & complete & fallback
    success = cohort & (complete - fallback)
    failed = cohort & (fallback - complete)
    pending = cohort - (complete | fallback)
    stages = {}
    for name, snapshots in (("direct_complete", success),
                            ("fallback_after_rank_start", failed & starts)):
        records = [start_timings[sid] for sid in snapshots if sid in start_timings]
        stages[name] = {
            "outcome_count": len(snapshots), "tp0_start_records": len(records),
            "stage_ms": {key: distribution(record[key] for record in records if key in record)
                         for key in sorted({key for record in records for key in record})},
        }
    return {
        "arrival_start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "arrival_end_utc_exclusive": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "eligible": len(cohort), "direct_complete": len(success),
        "fallback_no_rank_start": len(failed - starts),
        "fallback_after_rank_start": len(failed & starts),
        "fallback_after_group_receive": len(failed & received),
        "received_but_not_admitted_pending": sorted(pending & received),
        "success_event": "early_direct_admit (TP0 native group admission)",
        "conflicting_outcomes": sorted(conflict), "pending": sorted(pending),
        "success_fraction_of_all_arrivals": len(success) / len(cohort) if cohort else None,
        "tp0_start_timings_by_outcome": stages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--window", type=float, default=300)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--d-log", help="D log for a fast-arrival outcome cohort")
    parser.add_argument("--arrival-from", help="UTC ISO timestamp, inclusive (requires --d-log)")
    parser.add_argument("--arrival-to", help="UTC ISO timestamp, exclusive (requires --d-log)")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("window must be positive")
    if args.tp_size is not None and args.tp_size < 1:
        parser.error("tp-size must be positive")
    if any((args.d_log, args.arrival_from, args.arrival_to)):
        if not all((args.d_log, args.arrival_from, args.arrival_to)):
            parser.error("--d-log, --arrival-from and --arrival-to must be given together")
        def utc_timestamp(value):
            parsed = datetime.fromisoformat(value)
            return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
        try:
            cohort_start, cohort_end = map(utc_timestamp, (args.arrival_from, args.arrival_to))
        except ValueError:
            parser.error("invalid ISO timestamp")
        if cohort_start >= cohort_end:
            parser.error("arrival-from must precede arrival-to")
    with open(args.log, errors="replace") as source:
        result = summarize(source, args.window)
        if args.tp_size is not None:
            source.seek(0)
            result["group_handoff"] = summarize_group_handoff(source, args.tp_size, args.window)
        if args.d_log:
            source.seek(0)
            with open(args.d_log, errors="replace") as decode:
                result["direct_arrival_cohort"] = summarize_direct_cohort(
                    source, decode, cohort_start, cohort_end)
        print(json.dumps(result, indent=2))
