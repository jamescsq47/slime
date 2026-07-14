"""Render the offline-eval accuracy/length curves from eval_results/results.csv
and push them to wandb (project browsecomp-b300, one run per cell:
offline-eval-<model>-<mode>) so they sit next to the training runs.

Usage: WANDB_API_KEY=... python plot_eval_curves.py [--no-wandb]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).parent / "eval_results" / "results.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(CSV) as f:
        for r in csv.reader(f):
            if len(r) < 6:
                continue
            rows.append(dict(model=r[0], mode=r[1], iter=int(r[2]),
                             acc=float(r[3]), resp_len=float(r[4]), trunc=float(r[5])))

    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["mode"])].append(r)

    print(f"{'cell':34s} {'iter':>5s} {'acc':>7s} {'resp_len':>9s} {'trunc':>6s}")
    for (model, mode), rs in sorted(cells.items()):
        for r in sorted(rs, key=lambda x: x["iter"]):
            print(f"{model+'/'+mode:34s} {r['iter']:5d} {r['acc']:7.3f} {r['resp_len']:9.0f} {r['trunc']:6.2f}")

    if args.no_wandb:
        return

    import wandb
    for (model, mode), rs in sorted(cells.items()):
        run_id = f"offline-eval-{model}-{mode}".replace("_", "-")
        run = wandb.init(
            project="browsecomp-b300",
            id=run_id,
            name=f"offline-eval-{model}-{mode}",
            resume="allow",
            tags=["offline-eval", f"model={model}", f"mode={mode}"],
            reinit=True,
        )
        for r in sorted(rs, key=lambda x: x["iter"]):
            wandb.log({"offline_eval/acc": r["acc"],
                       "offline_eval/response_len_mean": r["resp_len"],
                       "offline_eval/truncated_ratio": r["trunc"]}, step=r["iter"])
        run.finish()
    print("pushed curves to wandb project browsecomp-b300 (offline-eval-* runs)")


if __name__ == "__main__":
    main()
