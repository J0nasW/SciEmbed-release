"""Evaluate SciEmbed variants and long-context baselines on the Body-Fact
Retrieval (BFR) benchmark built by `build_fullpaper_context_retrieval.py`.

Task definition:
  - queries: one short body-specific sentence per paper (no abstract overlap)
  - candidates: 9 749 scientific papers with full title + abstract + body
  - gold: the paper the query sentence was drawn from
  - metrics: Recall@1, Recall@10, nDCG@10 against the full candidate pool.

Models are configured with their native `max_seq_length`.  For a short-context
model (512), only the leading title+abstract of each candidate fits, so body
content is invisible.  For a long-context model (8192), the full body enters
the encoding.  The BFR queries are drawn from the body middle-third with
<=20% trigram overlap with the abstract, so the performance gap between
short and long contexts directly measures the value of the 8K window on
scientific text.

Output:
  - `metrics.json` per model with R@1, R@10, nDCG@10, aggregate scores, and
    the per-query best rank so error analysis can be run separately.
  - Combined `summary.jsonl` across all evaluated models.

Usage (single model):
    python scripts/eval_fullpaper_context_retrieval.py \
        --eval-dir output/eval/fullpaper_body_retrieval \
        --model <model_name_or_path> \
        --max-seq-length 8192 \
        --output-name sciembed_ctx_8192

Usage (batch):
    python scripts/eval_fullpaper_context_retrieval.py \
        --eval-dir output/eval/fullpaper_body_retrieval \
        --config scripts/configs/bfr_models.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    model_name_or_path: str
    max_seq_length: int
    pooling: str = "mean"  # mean | cls | auto (sentence-transformers default)
    trust_remote_code: bool = False


def format_candidate_text(
    title: str,
    abstract: str,
    body: str,
    *,
    max_seq_length: int,
) -> str:
    """Return the candidate representation.

    We concatenate `title. abstract. body` and let the tokenizer truncate to
    max_seq_length.  Short-context models therefore only see the leading
    title+abstract; long-context models see the body.
    """
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    body = (body or "").strip()
    return f"{title}. {abstract} {body}".strip()


def load_dataset(eval_dir: Path) -> tuple[Any, Any]:
    candidates = pq.read_table(eval_dir / "candidates.parquet").to_pandas()
    queries = pq.read_table(eval_dir / "queries.parquet").to_pandas()
    log.info("Loaded %d candidates, %d queries from %s",
             len(candidates), len(queries), eval_dir)
    return candidates, queries


def encode_st(
    spec: ModelSpec,
    texts: list[str],
    *,
    batch_size: int,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode `texts` with a sentence-transformers model.  Returns (N, D)
    L2-normalised float32."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        spec.model_name_or_path,
        trust_remote_code=spec.trust_remote_code,
    )
    model.max_seq_length = spec.max_seq_length
    log.info("Loaded %s (max_seq_length=%d, effective=%d)",
             spec.name, spec.max_seq_length, model.max_seq_length)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    del model
    torch.cuda.empty_cache()
    return emb.astype(np.float32)


def compute_ranking_metrics(
    query_emb: np.ndarray,
    cand_emb: np.ndarray,
    gold_rank_index: np.ndarray,
    samefield_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Given query × candidate cosine similarities (both L2-normalised),
    compute R@1, R@10, R@100, nDCG@10, MRR, and the per-query best rank.

    `gold_rank_index[i]` is the row in `cand_emb` that corresponds to the
    gold candidate for query i.

    If `samefield_mask` is provided (Q, C, bool), only mask=True candidates
    enter the per-query candidate pool (the gold position is always retained).
    This lets us compare full-pool retrieval (9 749 candidates, mixed field)
    to same-field retrieval (~400 candidates, all sharing the gold paper's
    field of study) on a single encode pass.
    """
    scores = query_emb @ cand_emb.T  # (Q, C)
    Q, C = scores.shape
    if samefield_mask is not None:
        scores = np.where(samefield_mask, scores, -np.inf)
        # Always keep the gold position visible (paranoid; mask should already include it).
        scores[np.arange(Q), gold_rank_index] = (query_emb * cand_emb[gold_rank_index]).sum(axis=1)
    gold_scores = scores[np.arange(Q), gold_rank_index]
    ranks = (scores > gold_scores[:, None]).sum(axis=1) + 1
    pool_sizes = samefield_mask.sum(axis=1) if samefield_mask is not None else np.full(Q, C)
    r_at_1 = float((ranks <= 1).mean())
    r_at_10 = float((ranks <= 10).mean())
    r_at_100 = float((ranks <= 100).mean())
    ndcg_10 = float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0).mean())
    mrr = float((1.0 / ranks).mean())
    return {
        "R@1": r_at_1,
        "R@10": r_at_10,
        "R@100": r_at_100,
        "nDCG@10": ndcg_10,
        "MRR": mrr,
        "median_rank": float(np.median(ranks)),
        "n_queries": int(Q),
        "n_candidates": int(C),
        "median_pool_size": float(np.median(pool_sizes)),
    }


def build_samefield_mask(
    candidate_fields: list[str | None],
    gold_idx: np.ndarray,
) -> np.ndarray:
    """Per-query boolean mask over candidates: True iff candidate i is in the
    same field of study as the query's gold candidate.  Candidates with empty
    field are excluded except at the gold position itself."""
    cand_fields = np.array([f if f else "__UNK__" for f in candidate_fields], dtype=object)
    gold_fields = cand_fields[gold_idx]
    Q = len(gold_idx)
    C = len(cand_fields)
    mask = np.empty((Q, C), dtype=bool)
    for q in range(Q):
        mask[q] = (cand_fields == gold_fields[q]) & (cand_fields != "__UNK__")
        mask[q, gold_idx[q]] = True
    return mask


def evaluate_model(spec: ModelSpec, eval_dir: Path, out_dir: Path, batch_size: int) -> dict:
    candidates, queries = load_dataset(eval_dir)
    cand_texts = [
        format_candidate_text(t, a, b, max_seq_length=spec.max_seq_length)
        for t, a, b in zip(candidates["title"], candidates["abstract"], candidates["body"])
    ]
    query_texts = list(queries["query_text"])

    t0 = time.time()
    log.info("Encoding %d candidates ...", len(cand_texts))
    cand_emb = encode_st(spec, cand_texts, batch_size=batch_size)
    t_cand = time.time() - t0

    t0 = time.time()
    log.info("Encoding %d queries ...", len(query_texts))
    q_emb = encode_st(spec, query_texts, batch_size=batch_size)
    t_q = time.time() - t0

    cand_id_to_row = {int(cid): i for i, cid in enumerate(candidates["candidate_id"])}
    gold_idx = np.array([
        cand_id_to_row[int(gcid)] for gcid in queries["gold_candidate_id"]
    ], dtype=np.int64)

    samefield_mask = None
    if "field_of_study" in candidates.columns:
        cand_fields = candidates["field_of_study"].fillna("").tolist()
        samefield_mask = build_samefield_mask(cand_fields, gold_idx)
        log.info(
            "Same-field mask: median pool=%d (full=%d)",
            int(np.median(samefield_mask.sum(1))), len(candidates),
        )

    metrics = compute_ranking_metrics(q_emb, cand_emb, gold_idx)
    metrics["pool"] = "fullpool"
    metrics.update(dict(
        model_name=spec.name,
        model_path=spec.model_name_or_path,
        max_seq_length=spec.max_seq_length,
        pooling=spec.pooling,
        encode_candidates_sec=round(t_cand, 2),
        encode_queries_sec=round(t_q, 2),
        embedding_dim=int(cand_emb.shape[1]),
    ))

    out_file = out_dir / f"metrics_{spec.name}.json"
    out_file.write_text(json.dumps(metrics, indent=2))
    log.info(
        "%s/full: R@1=%.4f R@10=%.4f nDCG@10=%.4f (cand %.1fs, q %.1fs)",
        spec.name, metrics["R@1"], metrics["R@10"], metrics["nDCG@10"], t_cand, t_q,
    )

    if samefield_mask is not None:
        sf = compute_ranking_metrics(q_emb, cand_emb, gold_idx, samefield_mask=samefield_mask)
        sf["pool"] = "samefield"
        sf.update(dict(
            model_name=f"{spec.name}_samefield",
            model_path=spec.model_name_or_path,
            max_seq_length=spec.max_seq_length,
            pooling=spec.pooling,
            encode_candidates_sec=round(t_cand, 2),
            encode_queries_sec=round(t_q, 2),
            embedding_dim=int(cand_emb.shape[1]),
        ))
        sf_file = out_dir / f"metrics_{spec.name}_samefield.json"
        sf_file.write_text(json.dumps(sf, indent=2))
        log.info(
            "%s/samefield median_pool=%d: R@1=%.4f R@10=%.4f nDCG@10=%.4f",
            spec.name, int(sf["median_pool_size"]), sf["R@1"], sf["R@10"], sf["nDCG@10"],
        )

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=None, type=Path)
    ap.add_argument("--model", default=None, help="Single-model mode: HF / local path")
    ap.add_argument("--model-name", default=None, help="Single-model mode: short name")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--config", default=None, type=Path,
                    help="Batch-mode JSON config with a list of model specs")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    out_dir = args.output_dir or args.eval_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: list[ModelSpec] = []
    if args.config:
        cfg = json.loads(args.config.read_text())
        for entry in cfg["models"]:
            specs.append(ModelSpec(
                name=entry["name"],
                model_name_or_path=entry["model_name_or_path"],
                max_seq_length=int(entry.get("max_seq_length", 8192)),
                pooling=entry.get("pooling", "mean"),
                trust_remote_code=bool(entry.get("trust_remote_code", False)),
            ))
    elif args.model:
        specs.append(ModelSpec(
            name=args.model_name or Path(args.model).name.replace("/", "_"),
            model_name_or_path=args.model,
            max_seq_length=args.max_seq_length,
            pooling=args.pooling,
            trust_remote_code=args.trust_remote_code,
        ))
    else:
        ap.error("Provide --model or --config")

    summary = []
    for spec in specs:
        log.info("=== Evaluating %s ===", spec.name)
        m = evaluate_model(spec, args.eval_dir, out_dir, args.batch_size)
        summary.append(m)
        with (out_dir / "summary.jsonl").open("w") as f:
            for s in summary:
                f.write(json.dumps(s) + "\n")

    log.info("Wrote %d model result(s) to %s", len(summary), out_dir)


if __name__ == "__main__":
    main()
