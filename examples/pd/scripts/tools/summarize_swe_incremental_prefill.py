"""Offline per-agent Prefill accounting for the unchanged SWE Qwen3.5 harness.

The common counterfactual uses a fully retained stable prompt checkpoint at
page64, including for colocated runs (not their observed native cache policy).
It is deliberately not the sum of all full-history prompt lengths.
"""
import argparse
import base64
import csv
import json
from collections import Counter
from pathlib import Path


def summarize(run, baseline_csv=None):
    tasks = [json.loads(line) for line in (run / "requests.jsonl").open()]
    actual = Counter()
    if baseline_csv:
        with baseline_csv.open() as f:
            for row in csv.DictReader(f):
                assert int(row['matched_turns']) == int(row['turns']), row['instance_id']
                actual[row['instance_id']] = int(row['uncached_prefill_tokens'])
    else:
        agents = {x['metadata']['agentic_request_id']: x['metadata']['instance_id'] for x in tasks}
        seen = set()
        # Full SWE runs can cross the logger's hourly rollover. Include both
        # the active log and dated siblings; rid de-duplication remains below.
        for f in sorted(run.glob('raw/prefill-*/*.log*')):
            for line in f.open():
                if '"event": "request.finished"' not in line:
                    continue
                raw = json.loads(line[line.index('{'):])
                m = raw['out']['meta_info']
                if m.get('prompt_tokens') is None or raw['rid'] in seen:
                    continue
                seen.add(raw['rid'])
                encoded = raw['obj']['extra_key'].split(':', 2)
                assert encoded[0] == 'agentic-v1e'
                payload = json.loads(base64.urlsafe_b64decode(encoded[2] + '=' * (-len(encoded[2]) % 4)))
                actual[agents[payload['agentic_request_id']]] += m['prompt_tokens'] - m.get('cached_tokens', 0)
    rows = []
    for task in tasks:
        m = task['metadata']
        events = m['openenv_trajectory']['turn_events']
        previous = 0
        full_prompts = new_tokens = aligned = boundary = 0
        for event in events:
            assert not event.get('context_compacted'), 'Compacted histories need explicit token LCP accounting'
            prompt = event.get('prompt_tokens', event.get('input_tokens'))
            # Qwen3.5's generation opener <think>\n is two tokens. The source
            # p2d_mamba_checkpoint_tokens removes it before page64 alignment.
            stable = max(0, previous - 2) if previous else 0
            assert prompt >= stable, 'Non-append history needs explicit token LCP accounting'
            new_tokens += prompt - stable
            aligned += prompt - stable // 64 * 64
            boundary += stable % 64
            full_prompts += prompt
            previous = prompt
        iid = m['instance_id']
        assert iid in actual, iid
        rows.append(dict(instance_id=iid, sample_index=task['sample_index'],
                         turns=len(events), actual_prefill_tokens=actual[iid],
                         ideal_incremental_tokens=new_tokens,
                         ideal_page64_prefill_tokens=aligned,
                         page64_boundary_recompute_tokens=boundary,
                         excess_over_ideal_page64_tokens=actual[iid]-aligned,
                         decode_tokens=sum(e.get('output_tokens', 0) for e in events),
                         cumulative_full_prompt_tokens=full_prompts))
    fields = [k for k in rows[0] if k not in {'instance_id', 'sample_index'}]
    summary = dict(run=str(run), requests=len(rows),
                   totals={k:sum(row[k] for row in rows) for k in fields},
                   per_agent_mean={k:sum(row[k] for row in rows)/len(rows) for k in fields},
                   definition='Fully retained stable prompt prefix; prior reasoning removed; common page64 counterfactual. Includes visible reply/tool/template suffix; not initial prompt+raw tool text only.')
    return rows, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline-csv', type=Path)
    args = parser.parse_args()
    rows, summary = summarize(args.run, args.baseline_csv)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'per_agent_incremental_prefill.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row:row['sample_index']))
    (args.output/'incremental_prefill_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))
