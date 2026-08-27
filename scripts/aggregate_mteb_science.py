"""Aggregate MTEB science-subset results into a single CSV/JSON for Table 4.

Reads per-task JSON files written by `scripts/run_mteb_subset.py` for each
model, extracts the primary `main_score`, and produces a tidy table:

    paper_table_row, model, task, main_score

Usage:
    python scripts/aggregate_mteb_science.py \
        --baselines-dir output/eval_results/mteb_science/baselines \
        --sciembed-dir  output/eval_results/mteb_science/sciembed \
        --output        output/eval_results/mteb_science/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = [
    "ArxivClassification.v2",
    "BIOSSES",
    "SciDocsRR",
    "SciFact",
    "TRECCOVID",
    "ArguAna",
    "NFCorpus",
    "BiorxivClusteringS2S.v2",
    "MedrxivClusteringS2S.v2",
]

# Map paper-friendly column name -> output dir name (stem of model-output dir).
MODEL_DIRS = {
    "BASE":      "sciembed-base",
    "CTX":       "sciembed-ctx",
    "FULL":      "sciembed-full",
    "GR2":       "granite-embedding-english-r2",
    "NomicMB":   "nomic-modernbert-embed-base",
    "SciNCL":    "scincl",
    "BGE-b":     "bge-base-en-v1.5",
    "BGE-l":     "bge-large-en-v1.5",
    "E5-b":      "e5-base-v2",
    "E5-l":      "e5-large-v2",
    "Nomic":     "nomic-embed-text-v1.5",
    "SPEC2":     "specter2-base",
    "ModernB":   "modernbert-base",
}


def find_main_score(task_json: Path) -> float | None:
    with open(task_json) as f:
        d = json.load(f)
    if "scores" not in d:
        return None
    for split, lst in d["scores"].items():
        if isinstance(lst, list) and lst:
            ms = lst[0].get("main_score")
            if ms is not None:
                return ms
    return None


def find_task_json(model_root: Path, task: str) -> Path | None:
    """Search for <task>.json under model_root (possibly nested)."""
    for p in model_root.rglob(f"{task}.json"):
        return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-dir", required=True)
    ap.add_argument("--sciembed-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    baselines = Path(args.baselines_dir)
    sciembed = Path(args.sciembed_dir)

    table = {}  # task -> { col -> score }
    for task in TASKS:
        table[task] = {}

    for col, dir_name in MODEL_DIRS.items():
        # try sciembed dir first, then baselines
        for root in (sciembed, baselines):
            mroot = root / dir_name
            if mroot.exists():
                for task in TASKS:
                    f = find_task_json(mroot, task)
                    if f is not None:
                        s = find_main_score(f)
                        if s is not None:
                            table[task][col] = round(s * 100, 2)
                break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2))

    cols = list(MODEL_DIRS.keys())
    print(f"{'Task':<28}", end="")
    for c in cols:
        print(f"{c:>9}", end="")
    print()
    for task in TASKS:
        row = table[task]
        print(f"{task:<28}", end="")
        for c in cols:
            v = row.get(c)
            print(f"{v:>9.2f}" if v is not None else f"{'--':>9}", end="")
        print()


if __name__ == "__main__":
    main()
