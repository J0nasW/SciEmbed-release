"""Build Signal D = OpenAlex related_works contrastive pairs.

OpenAlex's `works_related_works` table contains ~2.5B paper-paper "related to"
edges curated via OpenAlex's own embedding/clustering pipeline. These are
TOPICALLY SIMILAR papers, distinct from citation edges (which are CITED-BY).

For SciRepEval, the proximity tasks (SciDocs CoCite/CoView/CoRead, Same Author,
Highly Influential Citations) measure cited-together / similar-to patterns —
which is exactly what OpenAlex related_works encodes. Adding this as a third
supervision signal targets the proximity bucket where we're at 80.8.

Output schema (matching mixed_ctx parquet format):
  anchor:       "search_query: <title>. <abstract>"
  positive:     "search_document: <title>. <abstract>"
  negative:     NULL (use in-batch negatives via MNRL)
  signal_type:  "related_works"

Usage:
  python scripts/build_related_works_pairs.py \
      --output /path/to/data/sciembed_signal_d \
      --target-pairs 30000000 \
      --shard-size 1000000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datalake", default="/path/to/data/science_datalake/datalake.duckdb")
    p.add_argument("--output", required=True, help="Output dir for parquet shards")
    p.add_argument("--target-pairs", type=int, default=30_000_000)
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42, help="Sample seed for reproducibility")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.datalake, read_only=True)

    print(f"Sampling {args.target_pairs:,} related_works pairs (seed={args.seed})")
    # Sample via SQL — DuckDB's USING SAMPLE is reservoir-style, deterministic with seed
    sample_pct = max(0.1, min(100.0, args.target_pairs / 2.49e9 * 100 * 4))  # over-sample 4× to account for filter loss
    print(f"  raw sample %: {sample_pct:.3f}")

    sql = f"""
    WITH sampled AS (
        SELECT work_id, related_work_id
        FROM openalex.works_related_works
        USING SAMPLE {sample_pct} PERCENT (bernoulli, {args.seed})
    ),
    enriched AS (
        SELECT
            'search_query: '   || w1.title || '. ' || w1.abstract AS anchor,
            'search_document: '|| w2.title || '. ' || w2.abstract AS positive,
            CAST(NULL AS VARCHAR) AS negative,
            'related_works' AS signal_type
        FROM sampled s
        JOIN openalex.works w1 ON s.work_id = w1.id
        JOIN openalex.works w2 ON s.related_work_id = w2.id
        WHERE w1.valid_title_abstract AND w2.valid_title_abstract
          AND w1.language = 'en' AND w2.language = 'en'
          AND length(w1.abstract) > 50 AND length(w2.abstract) > 50
          AND length(w1.abstract) < 5000 AND length(w2.abstract) < 5000
    )
    SELECT * FROM enriched LIMIT {args.target_pairs}
    """

    print("Running query (this can take 5-15 minutes)...")
    # Write directly to parquet shards via COPY
    tmp_view = f"{out}/all_pairs"
    Path(tmp_view).mkdir(exist_ok=True)
    cmd = f"COPY ({sql}) TO '{out}/related_works.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE {args.shard_size}, COMPRESSION SNAPPY)"
    con.execute(cmd)

    # Verify
    final_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}/related_works.parquet')").fetchone()[0]
    print(f"Wrote {final_count:,} pairs to {out}/related_works.parquet")
    if final_count < args.target_pairs * 0.5:
        print(f"WARNING: yield {final_count:,} is below half of target {args.target_pairs:,}. "
              f"Consider increasing --target-pairs or raising sample_pct.")

    # Sample print
    sample = con.execute(f"SELECT * FROM read_parquet('{out}/related_works.parquet') LIMIT 2").fetchdf()
    print("\nSample:")
    for i, row in sample.iterrows():
        print(f"  [{i}] anchor:   {row['anchor'][:120]}...")
        print(f"      positive: {row['positive'][:120]}...")


if __name__ == "__main__":
    main()
