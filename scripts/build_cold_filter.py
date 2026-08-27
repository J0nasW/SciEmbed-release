"""Build cold-paper filter for SciRepEval supervised tasks.

For each high-overlap paper-level test task, output the list of test paper_ids
that lie in our Stage-2 Signal-A candidate universe.  The cold-paper SciRepEval
eval (`scripts/eval_scirepeval_cold.py`) drops these from the test split before
training/scoring, leaving only test papers SciEmbed could not have seen at
training time.

Output: docs/cold_filter.json with shape:
    {
      "<scirepeval_task_name>": [int_paper_id, ...],
      ...
      "_meta": { "datalake": "...", "snapshot": "...", "tasks": [...] }
    }

Usage:
    python scripts/build_cold_filter.py \\
        --datalake /path/to/data/science_datalake/datalake.duckdb \\
        --output docs/cold_filter.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import duckdb
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Paper-level supervised tasks where contamination > 10% (per docs/contamination_report.md).
# These are the tasks where a "cold" robustness check materially changes interpretation.
PAPER_TASKS: list[tuple[str, str]] = [
    ("mesh_descriptors", "classification"),   # 13.4% overlap
    ("biomimicry", "classification"),         # 20.9% overlap
    ("scidocs_mesh", "classification"),       # 18.0% overlap
    ("cite_count", "regression"),             # 18.7% overlap
    ("pub_year", "regression"),               # 27.5% overlap
]

# SciDocs MeSH uses SHA-1 ids; resolve via the official meta config.
SHA_META = {
    "scidocs_mesh": "scidocs_mag_mesh",
}


def load_sha_map(meta_config: str) -> dict[str, int]:
    log.info("Loading SHA->corpus_id map from %s ...", meta_config)
    ds = load_dataset("allenai/scirepeval", meta_config, split="evaluation")
    m: dict[str, int] = {}
    for sha, cid in zip(ds["doc_id"], ds["corpus_id"]):
        if sha is not None and cid is not None:
            try:
                m[sha] = int(cid)
            except (TypeError, ValueError):
                continue
    log.info("  %s: %d sha mappings", meta_config, len(m))
    return m


def collect_test_paper_ids(config: str, sha_map: dict[str, int] | None) -> list[tuple[str, int]]:
    """Returns list of (original_paper_id_str, int_corpus_id).

    The cold filter's keys must be the *original* paper_id strings the
    scirepeval evaluator sees (the SHA for SciDocs tasks, the int for others),
    because SupervisedEvaluator.read_dataset compares str(paper["paper_id"])
    against the embeddings dict.
    """
    ds = load_dataset("allenai/scirepeval_test", config, split="test")
    pairs: list[tuple[str, int]] = []
    for x in ds["paper_id"]:
        if x is None:
            continue
        try:
            v = int(x)
            pairs.append((str(x), v))
            continue
        except (ValueError, TypeError):
            pass
        if sha_map is not None and isinstance(x, str) and x in sha_map:
            pairs.append((str(x), sha_map[x]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalake", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    con = duckdb.connect(":memory:")
    import os
    threads = int(os.environ.get("DUCKDB_THREADS", os.cpu_count() or 8))
    mem_gb = int(os.environ.get("DUCKDB_MEMORY_GB", 40))
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{mem_gb}GB'")
    con.execute(f"ATTACH '{args.datalake}' AS dl (READ_ONLY)")
    con.execute("CREATE SCHEMA IF NOT EXISTS mem")

    log.info("Materialising Signal-A/B candidate universe (citationcount >= 5, abstract >= 50 chars) ...")
    con.execute(
        """
        CREATE TABLE mem.sci_train_candidates AS
        SELECT p.corpusid
        FROM dl.s2ag.papers p
        JOIN dl.s2ag.abstracts a ON a.corpusid = p.corpusid
        WHERE a.abstract IS NOT NULL
          AND LENGTH(a.abstract) >= 50
          AND p.citationcount >= 5
        """
    )
    n_sig = con.execute("SELECT COUNT(*) FROM mem.sci_train_candidates").fetchone()[0]
    log.info("  Signal-A universe: %d papers", n_sig)

    sha_maps: dict[str, dict[str, int]] = {}
    for meta in set(SHA_META.values()):
        sha_maps[meta] = load_sha_map(meta)

    cold_filter: dict[str, list[str]] = {}
    counts: list[dict] = []
    for config, kind in PAPER_TASKS:
        t0 = time.time()
        try:
            pairs = collect_test_paper_ids(config, sha_maps.get(SHA_META.get(config, ""), None))
        except Exception as e:
            log.warning("Skip %s: %s", config, e)
            continue
        if not pairs:
            log.warning("Empty test set for %s", config)
            continue
        n_test = len(pairs)
        df = pd.DataFrame({"original_id": [p[0] for p in pairs], "corpusid": [p[1] for p in pairs]}, dtype=object)
        df["corpusid"] = df["corpusid"].astype("int64")
        con.register("probe_ids", df)
        in_pool = con.execute(
            """
            SELECT p.original_id
            FROM probe_ids p
            JOIN mem.sci_train_candidates u ON u.corpusid = p.corpusid
            """
        ).df()["original_id"].astype(str).tolist()
        con.unregister("probe_ids")
        cold_filter[config] = in_pool
        counts.append({
            "task": config,
            "kind": kind,
            "n_test": n_test,
            "n_in_pool": len(in_pool),
            "pct_in_pool": round(100.0 * len(in_pool) / n_test, 2),
            "elapsed_sec": round(time.time() - t0, 1),
        })
        log.info("%s: %d/%d (%.1f%%) test papers in Signal-A pool [%.1fs]",
                 config, len(in_pool), n_test, counts[-1]["pct_in_pool"], counts[-1]["elapsed_sec"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cold_filter["_meta"] = {
        "datalake": args.datalake,
        "signal_a_universe_size": n_sig,
        "tasks": counts,
        "description": (
            "Per-task list of test paper_ids in the Stage-2 Signal-A candidate "
            "universe (citation_count >= 5, abstract >= 50 chars). The cold-paper "
            "SciRepEval eval drops these from the test split before scoring."
        ),
    }
    args.output.write_text(json.dumps(cold_filter, indent=2))
    log.info("Wrote cold filter to %s (tasks=%d)", args.output, len(counts))


if __name__ == "__main__":
    main()
