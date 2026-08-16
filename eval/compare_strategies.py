"""
Assemble the chunking bake-off comparison table from individual strategy
result JSONs (produced by eval/retrieval_eval.py --out ...).

Run: python eval/compare_strategies.py --results eval/results/strategy_A_bakeoff.json \
    eval/results/strategy_B_bakeoff.json eval/results/strategy_DE_bakeoff.json
"""

import argparse
import json


def format_table(results: list[dict]) -> str:
    cols = ["strategy", "n_evaluated", "recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    header = " | ".join(cols)
    sep = " | ".join("---" for _ in cols)
    rows = []
    for r in results:
        cells = []
        for c in cols:
            v = r.get(c, "?")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append(" | ".join(cells))
    return "\n".join([header, sep, *rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="eval/results/bakeoff_comparison.md")
    args = ap.parse_args()

    results = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))

    table = format_table(results)
    print(table)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Chunking bake-off comparison\n\n")
        f.write(table + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
