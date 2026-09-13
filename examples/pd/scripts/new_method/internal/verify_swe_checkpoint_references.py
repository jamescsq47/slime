"""Verify raw model token IDs, not only rendered Chat messages, in the smoke."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    records = json.loads((args.run_dir/'swe-stable-checkpoint-smoke.json').read_text())
    outputs = {}
    for path in sorted((args.run_dir/'raw').glob('decode-*/*.log')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('event') != 'request.finished':
                continue
            custom = (row.get('obj', {}).get('sampling_params') or {}).get('custom_params') or {}
            key = (custom.get('agentic_request_id'), custom.get('agentic_generation'))
            if not key[0]:
                continue
            tokens = row['out']['output_ids']
            assert isinstance(tokens, list) and tokens, (key, path)
            if key in outputs:
                assert outputs[key] == tokens, ('inconsistent duplicate', key)
            outputs[key] = tokens
    results = []
    for record in records:
        original = outputs[(record['request_id'], record['generation'])]
        reference = outputs[(record['reference_request_id'], 0)]
        results.append(dict(request_id=record['request_id'], generation=record['generation'],
                            delay=record['tool_delay'], original_tokens=len(original),
                            reference_tokens=len(reference), exact_token_ids=original == reference))
    result = dict(comparisons=results, all_exact=bool(results) and all(r['exact_token_ids'] for r in results))
    (args.run_dir/'raw-token-reference-comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert result['all_exact'], 'Raw output token IDs differ; preserve the mismatch for analysis'


if __name__ == '__main__':
    main()
