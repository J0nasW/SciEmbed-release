"""BM25 lexical baseline on the Body-Fact Retrieval (BFR) probe.

Two variants of the candidate field are evaluated to make the lexical baseline
diagnostic for both the short-context and the long-context regime:

  - bm25_short  : title + abstract     (what a 512-token model effectively sees)
  - bm25_full   : title + abstract + body  (what an 8192-token model sees)

Output schema matches the dense-retrieval eval (`eval_fullpaper_context_retrieval.py`)
so a downstream aggregator can append the BM25 rows to the same summary table.

Usage:
    python scripts/eval_bfr_bm25.py \\
        --eval-dir output/eval/fullpaper_body_retrieval \\
        --output-dir output/eval/fullpaper_body_retrieval/results
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def compute_metrics(scores: np.ndarray, gold_idx: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """scores: (Q, C) similarity scores; higher = better.

    If `mask` is provided (Q, C, bool), only positions with mask=True are
    considered as candidates for query i; this is how the same-field-pool
    diagnostic restricts each query's distractor set to its gold paper's field
    of study.  The gold candidate is always retained in the candidate pool.
    Ranks are computed against this restricted pool.
    """
    Q, C = scores.shape
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
        scores[np.arange(Q), gold_idx] = scores[np.arange(Q), gold_idx]  # ensure gold not masked
    gold_scores = scores[np.arange(Q), gold_idx]
    ranks = (scores > gold_scores[:, None]).sum(axis=1) + 1
    pool_sizes = (mask.sum(axis=1) if mask is not None else np.full(Q, C))
    return {
        "R@1": float((ranks <= 1).mean()),
        "R@10": float((ranks <= 10).mean()),
        "R@100": float((ranks <= 100).mean()),
        "nDCG@10": float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0).mean()),
        "MRR": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
        "n_queries": int(Q),
        "n_candidates": int(C),
        "median_pool_size": float(np.median(pool_sizes)),
        "min_pool_size": int(pool_sizes.min()),
        "max_pool_size": int(pool_sizes.max()),
    }


def build_samefield_mask(
    candidate_fields: list[str | None],
    gold_idx: np.ndarray,
) -> np.ndarray:
    """Per-query boolean mask: True at candidate i iff i is in the same field
    as the query's gold candidate (or has no field, in which case it is treated
    as out-of-field and masked out)."""
    cand_fields = np.array([f if f else "__UNK__" for f in candidate_fields], dtype=object)
    gold_fields = cand_fields[gold_idx]
    Q = len(gold_idx)
    C = len(cand_fields)
    mask = np.empty((Q, C), dtype=bool)
    for q in range(Q):
        mask[q] = (cand_fields == gold_fields[q]) & (cand_fields != "__UNK__")
        # Always keep the gold position in the pool, even if its field is unknown
        mask[q, gold_idx[q]] = True
    return mask


def run_bm25(name: str, candidate_texts: list[str], queries: list[str], gold_idx: np.ndarray, samefield_mask: np.ndarray | None = None) -> dict:
    log.info("[%s] tokenising %d candidates ...", name, len(candidate_texts))
    t0 = time.time()
    cand_tokens = [tokenize(t) for t in candidate_texts]
    bm25 = BM25Okapi(cand_tokens)
    t_index = time.time() - t0
    log.info("[%s] indexed in %.1fs", name, t_index)

    log.info("[%s] scoring %d queries ...", name, len(queries))
    t0 = time.time()
    scores = np.empty((len(queries), len(candidate_texts)), dtype=np.float32)
    for i, q in enumerate(queries):
        scores[i] = bm25.get_scores(tokenize(q))
        if (i + 1) % 100 == 0:
            log.info("[%s]   %d / %d queries", name, i + 1, len(queries))
    t_score = time.time() - t0

    metrics = compute_metrics(scores, gold_idx)
    metrics["pool"] = "fullpool"
    metrics.update(dict(
        model_name=name,
        model_path="bm25_okapi",
        max_seq_length=None,
        pooling="bm25",
        encode_candidates_sec=round(t_index, 2),
        encode_queries_sec=round(t_score, 2),
        embedding_dim=None,
    ))
    log.info(
        "[%s/full] R@1=%.4f R@10=%.4f nDCG@10=%.4f MRR=%.4f",
        name, metrics["R@1"], metrics["R@10"], metrics["nDCG@10"], metrics["MRR"],
    )
    if samefield_mask is not None:
        sf = compute_metrics(scores, gold_idx, mask=samefield_mask)
        sf.update(dict(
            model_name=f"{name}_samefield",
            model_path="bm25_okapi",
            max_seq_length=None,
            pooling="bm25",
            pool="samefield",
            encode_candidates_sec=round(t_index, 2),
            encode_queries_sec=round(t_score, 2),
            embedding_dim=None,
        ))
        log.info(
            "[%s/samefield median_pool=%d] R@1=%.4f R@10=%.4f nDCG@10=%.4f",
            name, int(sf["median_pool_size"]), sf["R@1"], sf["R@10"], sf["nDCG@10"],
        )
        return [metrics, sf]
    return [metrics]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=None, type=Path)
    args = ap.parse_args()

    out_dir = args.output_dir or args.eval_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = pq.read_table(args.eval_dir / "candidates.parquet").to_pandas()
    queries = pq.read_table(args.eval_dir / "queries.parquet").to_pandas()
    log.info("Loaded %d candidates, %d queries", len(candidates), len(queries))

    cand_id_to_row = {int(cid): i for i, cid in enumerate(candidates["candidate_id"])}
    gold_idx = np.array([
        cand_id_to_row[int(g)] for g in queries["gold_candidate_id"]
    ], dtype=np.int64)
    query_texts = list(queries["query_text"])

    titles = candidates["title"].fillna("").tolist()
    abstracts = candidates["abstract"].fillna("").tolist()
    bodies = candidates["body"].fillna("").tolist()
    fields = candidates["field_of_study"].fillna("").tolist() if "field_of_study" in candidates.columns else None

    samefield_mask = build_samefield_mask(fields, gold_idx) if fields is not None else None
    if samefield_mask is not None:
        log.info("Same-field mask: median pool=%d, min=%d, max=%d (full pool=%d)",
                 int(np.median(samefield_mask.sum(1))), samefield_mask.sum(1).min(),
                 samefield_mask.sum(1).max(), len(candidates))

    short_texts = [f"{t}. {a}".strip() for t, a in zip(titles, abstracts)]
    full_texts = [f"{t}. {a} {b}".strip() for t, a, b in zip(titles, abstracts, bodies)]

    all_results = []
    short_results = run_bm25("bm25_short", short_texts, query_texts, gold_idx, samefield_mask)
    all_results.extend(short_results)
    (out_dir / "metrics_bm25_short.json").write_text(json.dumps(short_results[0], indent=2))
    if len(short_results) > 1:
        (out_dir / "metrics_bm25_short_samefield.json").write_text(json.dumps(short_results[1], indent=2))

    full_results = run_bm25("bm25_full", full_texts, query_texts, gold_idx, samefield_mask)
    all_results.extend(full_results)
    (out_dir / "metrics_bm25_full.json").write_text(json.dumps(full_results[0], indent=2))
    if len(full_results) > 1:
        (out_dir / "metrics_bm25_full_samefield.json").write_text(json.dumps(full_results[1], indent=2))

    summary_path = out_dir / "summary_bm25.jsonl"
    with summary_path.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    log.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
