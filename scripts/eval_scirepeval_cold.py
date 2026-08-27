"""Cold-paper SciRepEval eval.

Wrapper around the official `scirepeval.py` runner that monkey-patches
`SupervisedEvaluator.read_dataset` to drop test papers in the Signal-A candidate
pool before training the probe.  This produces a cold-paper score for the five
high-overlap supervised tasks (mesh_descriptors, biomimicry, scidocs_mesh,
cite_count, pub_year) without re-implementing the official evaluator.

The official evaluator iterates `data["train"]`/`data["test"]` and reads
`paper["paper_id"]` and `paper["label"]`.  We replace the staticmethod on the
class with a method that consults a `cold_filter` dict (loaded from JSON via
the `--cold-filter` flag) and drops test rows whose paper_id matches.
Train data is left unmodified -- the probe still trains on the full SciRepEval
training set, so probe quality is not artificially weakened.

Usage (mirrors the official runner; passes through to it):
    python scripts/eval_scirepeval_cold.py \\
        --scirepeval-dir /path/to/scirepeval \\
        --cold-filter docs/cold_filter.json \\
        --model <ckpt> \\
        --pooling-mode mean \\
        --output cold_full.json \\
        --embeddings-save-path /tmp/embeddings_cold

Outputs the standard scirepeval JSON; cold-tasks contain cold-filtered scores,
all other tasks contain official scores.  The list of dropped paper_ids per
task is logged to stderr at run start.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


SLUG_TO_DISPLAY = {
    "biomimicry": "Biomimicry",
    "mesh_descriptors": "MeSH",
    "scidocs_mesh": "SciDocs MeSH",
    "cite_count": "Citation Count",
    "pub_year": "Publication Year",
}


def install_cold_filter(cold_filter: dict[str, list[str]]) -> None:
    """Monkey-patch SupervisedEvaluator.read_dataset to apply the cold filter.

    SciRepEval keys evaluators by their human-readable display name (`self.name`,
    e.g. "MeSH", "Citation Count"), not by the dataset slug used in cold_filter.json
    (e.g. "mesh_descriptors", "cite_count").  We translate slug -> display via
    SLUG_TO_DISPLAY before installing the lookup.
    """
    from evaluation.evaluator import SupervisedEvaluator

    raw = {k: set(str(x) for x in v) for k, v in cold_filter.items() if not k.startswith("_")}
    excluded_per_task: dict[str, set[str]] = {}
    for slug, ids in raw.items():
        display = SLUG_TO_DISPLAY.get(slug)
        if display is None:
            log.warning("Cold filter: no display-name mapping for slug %r; skipping", slug)
            continue
        excluded_per_task[display] = ids
    log.info("Cold filter installed for tasks: %s", sorted(excluded_per_task.keys()))
    for task, ids in excluded_per_task.items():
        log.info("  %s: %d excluded paper_ids", task, len(ids))

    def filtered_read_dataset(self, data, embeddings) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Replacement for the SupervisedEvaluator.read_dataset staticmethod.

        Identical semantics to the upstream method, except that test rows whose
        paper_id appears in the cold filter for `self.name` are dropped before
        the train/test arrays are built.  Training rows are unchanged.
        """
        train, test = data["train"], data["test"]
        excluded = excluded_per_task.get(self.name, set())
        if excluded:
            test_filtered = [p for p in test if str(p["paper_id"]) not in excluded]
            n_before = len(test)
            n_after = len(test_filtered)
            log.info(
                "[cold] %s: %d -> %d test papers (dropped %d in-pool)",
                self.name, n_before, n_after, n_before - n_after,
            )
        else:
            test_filtered = list(test)

        x_train = np.array([
            embeddings[str(p["paper_id"])] for p in train
            if str(p["paper_id"]) in embeddings
        ])
        x_test = np.array([
            embeddings[str(p["paper_id"])] for p in test_filtered
            if str(p["paper_id"]) in embeddings
        ])
        y_train = np.array([
            p["label"] for p in train
            if str(p["paper_id"]) in embeddings
        ])
        y_test = np.array([
            p["label"] for p in test_filtered
            if str(p["paper_id"]) in embeddings
        ])
        return x_train, x_test, y_train, y_test

    # The upstream definition is a @staticmethod, but instance access still
    # works after rebinding to a function: SupervisedEvaluator.<name>(self, ...).
    SupervisedEvaluator.read_dataset = filtered_read_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scirepeval-dir", required=True, help="Path to the official scirepeval clone")
    ap.add_argument("--cold-filter", required=True, type=Path)
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--pooling-mode", default="mean")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--embeddings-save-path", required=True)
    ap.add_argument("--task-list", nargs="+", default=None,
                    help="Optional restriction; defaults to ALL 22 SciRepEval tasks")
    args = ap.parse_args()

    sys.path.insert(0, args.scirepeval_dir)

    cold_filter = json.loads(args.cold_filter.read_text())
    install_cold_filter(cold_filter)

    # Match the official runner's setup
    from evaluation.encoders import Model
    from scirepeval import SciRepEval, TASK_IDS  # noqa: F401

    model = Model(
        variant="default",
        base_checkpoint=args.model,
        all_tasks=list(TASK_IDS.values()),
        task_id=TASK_IDS,
        pooling_mode=args.pooling_mode,
    )

    runner = SciRepEval(
        tasks_config=str(Path(args.scirepeval_dir) / "scirepeval_tasks.jsonl"),
        task_list=args.task_list,
        batch_size=args.batch_size,
        embedding_save_path=args.embeddings_save_path,
    )
    runner.evaluate(model, args.output)


if __name__ == "__main__":
    main()
