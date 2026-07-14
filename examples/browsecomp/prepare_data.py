"""Convert FoldAgent BrowseComp-Plus parquet data into slime jsonl format.

The FoldAgent parquet (bc_train.parquet / bc_test.parquet) rows contain:
  - prompt:     [system, user] messages with the research-agent system prompt
                (search / open_page / finish function definitions in text format)
  - answer:     ground-truth answer string
  - extra_info: {query, answer, evidence_docs, ...}
  - data_source: bc_{train,test}_{easy,meduim,hard}

slime jsonl rows produced here:
  - prompt:   the messages list, passed through as-is (no chat template applied
              at data-load time; the session server renders it per request)
  - label:    ground-truth answer string
  - metadata: {question, answer, data_source, instance_id}

Usage:
  python prepare_data.py \
      --input /path/to/FoldAgent/data/bc_train.parquet \
      --output /path/to/data/bc_train.jsonl
"""

import argparse
import json

import pandas as pd


def to_plain(obj):
    """Recursively convert numpy containers to plain Python types."""
    if hasattr(obj, "tolist"):
        return to_plain(obj.tolist())
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="FoldAgent bc_*.parquet path")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            prompt = to_plain(row["prompt"])
            extra_info = to_plain(row["extra_info"])
            answer = (row["answer"] or extra_info.get("answer") or "").strip()
            assert prompt and answer, f"row {n_written} missing prompt/answer"
            record = {
                "prompt": prompt,
                "label": answer,
                "metadata": {
                    "question": extra_info["query"],
                    "answer": answer,
                    "data_source": row["data_source"],
                    "instance_id": extra_info.get("instance_id"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} rows to {args.output}")


if __name__ == "__main__":
    main()
