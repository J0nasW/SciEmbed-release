"""Thin runner around the official MTEB package for the 9-task science subset.

Same conventions as scripts/run_mteb_official.py but evaluates a fixed task
list rather than the full MTEB(eng, v2) benchmark.

Tasks evaluated (matches paper Table 4):
    Classification: ArxivClassification.v2
    STS:            BIOSSES
    Reranking:      SciDocsRR
    Retrieval:      SciFact, TRECCOVID, ArguAna, NFCorpus
    Clustering:     BiorxivClusteringS2S.v2, MedrxivClusteringS2S.v2

Invocation:
    python scripts/run_mteb_subset.py \
        --model <path_or_hf_name> \
        --name <output_name> \
        --output-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer


SCIENCE_TASKS = [
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument("--name", required=True, help="Output subdir name")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Optional task override (default: SCIENCE_TASKS)")
    args = p.parse_args()

    out = Path(args.output_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)

    task_names = args.tasks or SCIENCE_TASKS
    print(f"[mteb-subset] model={args.model}")
    print(f"[mteb-subset] tasks={task_names}")
    print(f"[mteb-subset] output={out}")

    model = SentenceTransformer(args.model, trust_remote_code=True)
    tasks = mteb.get_tasks(tasks=task_names)
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, output_folder=str(out))

    print(f"[mteb-subset] DONE -> {out}")


if __name__ == "__main__":
    main()
