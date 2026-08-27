"""Signal G — TLDR ↔ abstract pairs.

70M Semantic Scholar TLDRs (auto-generated 1-sentence summaries) paired with
their source paper's title+abstract. This provides ASYMMETRIC query/document
supervision: short, summary-style "query" → full title+abstract "document".

Direct target: SciRepEval search bucket where we trail Granite R2 by 2 points.
TLDR format mimics natural search queries / question-answering pairs, which is
the supervision general retrieval models receive.

Usage:
    python scripts/build_signal_g_tldr.py \
        --output /path/to/data/sciembed_signal_g \
        --target-pairs 30000000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import duckdb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datalake", default="/path/to/data/science_datalake/datalake.duckdb")
    p.add_argument("--output", required=True)
    p.add_argument("--target-pairs", type=int, default=30_000_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.datalake, read_only=True)

    sql = f"""
    WITH sampled_tldrs AS (
        SELECT corpusid, text
        FROM s2ag.tldrs
        WHERE text IS NOT NULL AND length(text) >= 30
        USING SAMPLE {args.target_pairs * 2} ROWS (reservoir, {args.seed})
    ),
    enriched AS (
        SELECT
            'search_query: '   || t.text AS anchor,
            'search_document: '|| p.title || '. ' || a.abstract AS positive,
            CAST(NULL AS VARCHAR) AS negative,
            'tldr_abstract' AS signal_type
        FROM sampled_tldrs t
        JOIN s2ag.papers p ON t.corpusid = p.corpusid
        JOIN s2ag.abstracts a ON t.corpusid = a.corpusid
        WHERE p.title IS NOT NULL AND a.abstract IS NOT NULL
          AND length(a.abstract) BETWEEN 50 AND 5000
          AND length(t.text) BETWEEN 30 AND 600
    )
    SELECT * FROM enriched LIMIT {args.target_pairs}
    """

    print(f"Building Signal G: target {args.target_pairs:,} TLDR-abstract pairs")
    cmd = f"COPY ({sql}) TO '{out}/tldr_abstract.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE 1000000, COMPRESSION SNAPPY)"
    con.execute(cmd)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}/tldr_abstract.parquet')").fetchone()[0]
    print(f"Wrote {n:,} TLDR-abstract pairs to {out}/tldr_abstract.parquet")
    print(con.execute(f"SELECT * FROM read_parquet('{out}/tldr_abstract.parquet') LIMIT 2").fetchdf().to_string())


if __name__ == "__main__":
    main()
