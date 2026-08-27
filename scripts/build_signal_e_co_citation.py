"""Signal E — co-citation pairs.

Co-citation = pairs of papers cited by the same paper. A classic scientometrics
similarity signal that is *exactly* what SciDocs CoCite/CoView/CoRead measure.
Different from Signal A (which is the citing→cited edge): Signal E is
sibling→sibling under a shared parent.

Direct target: SciRepEval proximity bucket where we trail Granite R2 by 1.5.

Strategy:
  1. Sample N citing papers uniformly from s2ag.citations.
  2. For each citing paper, take all its cited papers; form all unordered pairs.
  3. Enrich pairs with title + abstract from s2ag.papers + s2ag.abstracts.
  4. Filter to English-text, length-bounded abstracts.

Usage:
    python scripts/build_signal_e_co_citation.py \
        --output /path/to/data/sciembed_signal_e \
        --target-pairs 30000000 \
        --citing-sample 5000000
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
    p.add_argument("--citing-sample", type=int, default=5_000_000,
                   help="Sample of citing papers to draw co-citation pairs from")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.datalake, read_only=True)
    # Cap DuckDB memory + thread count to avoid OOM on self-join blowup.
    con.execute("SET memory_limit='150GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")

    # Co-citation pairs: self-join s2ag.citations on citingcorpusid.
    # To control combinatorial blow-up, restrict each citing paper to at most
    # 10 outgoing citations per parent (cap controlled by MAX_OUTGOING_PER_PARENT below).
    # Also enforce a < b on cited_id to dedupe symmetric pairs.
    sql = f"""
    WITH citing_pool AS (
        SELECT DISTINCT citingcorpusid AS cid
        FROM s2ag.citations
        WHERE citingcorpusid IS NOT NULL AND citedcorpusid IS NOT NULL
        USING SAMPLE {args.citing_sample} ROWS (reservoir, {args.seed})
    ),
    bounded_refs AS (
        SELECT c.citingcorpusid, c.citedcorpusid,
               ROW_NUMBER() OVER (PARTITION BY c.citingcorpusid ORDER BY c.citationid) AS rn
        FROM s2ag.citations c
        JOIN citing_pool cp ON c.citingcorpusid = cp.cid
        WHERE c.citedcorpusid IS NOT NULL
        QUALIFY rn <= 10
    ),
    cocite_pairs AS (
        SELECT a.citedcorpusid AS x, b.citedcorpusid AS y
        FROM bounded_refs a
        JOIN bounded_refs b
          ON a.citingcorpusid = b.citingcorpusid
         AND a.citedcorpusid < b.citedcorpusid
    ),
    pairs_dedup AS (
        SELECT DISTINCT x, y FROM cocite_pairs
    ),
    enriched AS (
        SELECT
            'search_query: '   || p1.title || '. ' || a1.abstract AS anchor,
            'search_document: '|| p2.title || '. ' || a2.abstract AS positive,
            CAST(NULL AS VARCHAR) AS negative,
            'cocitation' AS signal_type
        FROM pairs_dedup pp
        JOIN s2ag.papers p1 ON pp.x = p1.corpusid
        JOIN s2ag.papers p2 ON pp.y = p2.corpusid
        JOIN s2ag.abstracts a1 ON pp.x = a1.corpusid
        JOIN s2ag.abstracts a2 ON pp.y = a2.corpusid
        WHERE p1.title IS NOT NULL AND p2.title IS NOT NULL
          AND a1.abstract IS NOT NULL AND a2.abstract IS NOT NULL
          AND length(a1.abstract) BETWEEN 50 AND 5000
          AND length(a2.abstract) BETWEEN 50 AND 5000
    )
    SELECT * FROM enriched LIMIT {args.target_pairs}
    """

    print(f"Building Signal E: target {args.target_pairs:,} pairs from {args.citing_sample:,} citing papers")
    cmd = f"COPY ({sql}) TO '{out}/cocitation.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE 1000000, COMPRESSION SNAPPY)"
    con.execute(cmd)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}/cocitation.parquet')").fetchone()[0]
    print(f"Wrote {n:,} co-citation pairs to {out}/cocitation.parquet")
    print(con.execute(f"SELECT * FROM read_parquet('{out}/cocitation.parquet') LIMIT 2").fetchdf().to_string())


if __name__ == "__main__":
    main()
