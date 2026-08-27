"""Build a science-specific long-context retrieval eval dataset.

Task definition: **Body-Fact Retrieval (BFR)**.  Given a sentence extracted
from the body of a scientific paper that does *not* substantively appear in
the paper's abstract, retrieve the source paper from a candidate pool of full
text documents.

Why this design:
  - The query is a body-specific sentence: short-context encoders that only
    see title+abstract (<=512 tokens) are blind to the content that must be
    matched, while long-context encoders (>=2K, >=8K) that see the body can
    recover it.  This is the strongest defensible test of whether the 8 K
    context window is actually exploited on scientific documents.
  - The candidate pool restricts citation_count to [1, 4], guaranteeing that
    no candidate paper is in the Stage 2 Signal-A/B candidate universe (which
    required citation_count >= 5).  See docs/DATA_USAGE_AUDIT.md §6.
  - Holding queries to body-only content with trigram novelty vs. abstract
    removes the trivial shortcut of matching the abstract alone.

Leakage guarantees:
  - No candidate paper was a Stage 2 anchor, positive, or hard negative
    (citation_count filter).
  - MLM may have seen a candidate paper's full text via fulltext.papers; this
    is disclosed and does not constitute retrieval-task leakage (MLM is
    self-supervised, sees no query-document pairing).

Output (written to `--output`):
  - `candidates.parquet`: candidate_id (int64, s2ag corpusid), title, abstract,
    body (first MAX_BODY_CHARS characters of body_text), field_of_study,
    citation_count.
  - `queries.parquet`: query_id (int64; a synthetic id), query_text (body
    sentence), gold_candidate_id (int64).
  - `dataset_info.json`: build params, holdout evidence, field distribution,
    body-length distribution, novelty-filter stats.

Usage:
    python scripts/build_fullpaper_context_retrieval.py \
        --datalake /path/to/data/science_datalake/datalake.duckdb \
        --output output/eval/fullpaper_body_retrieval \
        --n-candidates 10000 \
        --n-queries 1000 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Stage 2 required citation_count >= 5; HOLDOUT_MAX_CITES keeps us strictly
# below that so the candidate paper is not in the Signal A+B candidate universe.
HOLDOUT_MIN_CITES = 1
HOLDOUT_MAX_CITES = 4

# Body-length thresholds in characters.  ~5 chars/token → 8 000 chars ≈ 1 600
# tokens which is well past the 512-token window of short-context baselines,
# and 80 000 chars ≈ 16 000 tokens which caps above an 8 K window.
MIN_BODY_CHARS = 8_000
MAX_BODY_CHARS = 80_000

# Query-novelty thresholds.  A body sentence is accepted as a query only if
# (i) it is within MIN/MAX char range so it is an actual self-contained
# sentence rather than a fragment, and (ii) fewer than TRIGRAM_OVERLAP_MAX
# fraction of its trigrams also appear in the abstract (removing the trivial
# shortcut of matching a body sentence that is already in the abstract).
MIN_QUERY_CHARS = 80
MAX_QUERY_CHARS = 400
TRIGRAM_OVERLAP_MAX = 0.2

# Regex for sentence splitting.  Simple but adequate for the narrow purpose
# of extracting mid-paragraph claims from scientific prose.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
# Heuristic: drop sentences that are mostly inline citation fragments or
# numeric artefacts.  These appear frequently in S2ORC body_text after
# bibliography parsing.
_NOISE_RE = re.compile(
    r"^(\s*\[?\d+\]?\s*[,;]?\s*)+$"  # bare reference markers
    r"|\bFig(?:ure)?\.?\s*\d+"        # "Fig. 3"
    r"|\bTable\s*\d+"                   # "Table 2"
    r"|^et\s+al\b"                       # etal fragments
)


# We materialise a *small* anchor set of corpusids first (papers with
# citation_count in the target range), then drive the join from that.
# Letting the planner consider the full 12M-row s2orc table as the probe
# side caused prior runs to scan body_text for ~40 min.
ANCHOR_CORPUSIDS_QUERY = """
CREATE OR REPLACE TABLE mem.anchor_corpusids AS
SELECT p.corpusid,
       p.title,
       COALESCE(p.s2fieldsofstudy[1].category, 'Unknown') AS field_of_study,
       p.citationcount AS citation_count
FROM dl.s2ag.papers p
WHERE p.citationcount BETWEEN {min_cites} AND {max_cites}
  AND p.title IS NOT NULL
"""

ANCHOR_WITH_ABSTRACT_QUERY = """
CREATE OR REPLACE TABLE mem.anchor_with_abstract AS
SELECT ac.corpusid,
       ac.title,
       a.abstract,
       ac.field_of_study,
       ac.citation_count
FROM mem.anchor_corpusids ac
JOIN dl.s2ag.abstracts a ON a.corpusid = ac.corpusid
WHERE a.abstract IS NOT NULL AND LENGTH(a.abstract) >= 50
"""

# Precomputed text_length in dl.fulltext.s2orc lets us filter without ever
# reading body_text — the difference between "scan ~60 GB of body_text" and
# "scan a single ~200 MB int column".  We then map DOI → corpusid via
# dl.s2ag.papers, and only pull body_text for the final sampled candidates.
S2ORC_LENGTHS_QUERY = """
CREATE OR REPLACE TABLE mem.s2orc_lengths AS
SELECT p.corpusid, fs.text_length AS body_len
FROM dl.fulltext.s2orc fs
JOIN dl.s2ag.papers p ON lower(p.doi) = lower(fs.doi)
WHERE fs.has_full_text = true
  AND fs.text_length BETWEEN {min_body} AND {max_body}
  AND fs.doi IS NOT NULL
  AND p.doi IS NOT NULL
"""

# Meta-only candidate table (no body_text column — keeps it in the KB range
# rather than GB).  Body text is fetched per-id at the end.
CANDIDATE_QUERY = """
CREATE OR REPLACE TABLE mem.candidates_all AS
SELECT
    aa.corpusid AS candidate_id,
    aa.title,
    aa.abstract,
    aa.field_of_study,
    aa.citation_count,
    sl.body_len
FROM mem.anchor_with_abstract aa
JOIN mem.s2orc_lengths sl ON sl.corpusid = aa.corpusid
"""

# Used after sampling: pull body_text for the final ~10K candidate ids only.
# dl.s2ag.s2orc is keyed by corpusid with body_text; a narrow IN-list makes
# this a cheap row-group-filtered scan.
FETCH_BODY_TEXTS_QUERY = """
SELECT o.corpusid AS candidate_id, o.body_text AS body
FROM dl.s2ag.s2orc o
WHERE o.corpusid IN (SELECT candidate_id FROM mem.candidates)
"""


def trigrams(text: str) -> set[str]:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    tokens = re.findall(r"[a-z0-9]+", text)
    return {" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2)} if len(tokens) >= 3 else set()


def extract_body_query(
    abstract: str,
    body: str,
    rng: random.Random,
) -> str | None:
    """Pick a body sentence whose content is not already in the abstract.

    Returns None if no suitable sentence is found.  Picks deterministically
    using the provided RNG so the final query set is reproducible.
    """
    abstract_trigrams = trigrams(abstract)
    # Focus on the middle third of the body (skip likely intro / conclusion
    # content that tends to restate the abstract)
    n = len(body)
    body_middle = body[n // 3 : 2 * n // 3]
    sentences = [s.strip() for s in _SENTENCE_RE.split(body_middle)]
    # Shuffle so we do not always pick the earliest usable sentence — helps
    # stratify the query distribution across the middle third of the body.
    rng.shuffle(sentences)

    for sent in sentences:
        if not (MIN_QUERY_CHARS <= len(sent) <= MAX_QUERY_CHARS):
            continue
        if _NOISE_RE.search(sent):
            continue
        st = trigrams(sent)
        if not st:
            continue
        overlap = len(st & abstract_trigrams) / len(st)
        if overlap > TRIGRAM_OVERLAP_MAX:
            continue
        return sent
    return None


def sample_candidates_meta(
    con: duckdb.DuckDBPyConnection,
    n_candidates: int,
    seed: int,
) -> pd.DataFrame:
    """Sample n_candidates rows (metadata only — no body) from mem.candidates_all.

    Stratified by field, deterministic via seed.  Body text is fetched
    separately by the caller for only the sampled ids to avoid scanning the
    full body_text column of the datalake's s2orc table.
    """
    con.execute(
        f"""
        CREATE OR REPLACE TABLE mem.candidates AS
        SELECT candidate_id, title, abstract, field_of_study, citation_count, body_len
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY field_of_study
                       ORDER BY hash(candidate_id + {seed})
                   ) AS rn
            FROM mem.candidates_all
        )
        WHERE rn <= GREATEST(1, {n_candidates} / 25)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE mem.candidates AS
        SELECT candidate_id, title, abstract, field_of_study, citation_count, body_len
        FROM (
            SELECT *, hash(candidate_id + {seed + 1}) AS h
            FROM mem.candidates
        )
        ORDER BY h
        LIMIT {n_candidates}
        """
    )
    log.info("Sampled %d candidate ids (metadata only)",
             con.execute("SELECT COUNT(*) FROM mem.candidates").fetchone()[0])
    return con.execute(
        "SELECT candidate_id, title, abstract, field_of_study, citation_count FROM mem.candidates"
    ).fetchdf()


def build_queries(
    candidates_df: pd.DataFrame,
    n_queries: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Build `n_queries` (query_text, gold_candidate_id) pairs from
    body-specific sentences.

    Returns (queries_df, stats).
    """
    rng = random.Random(seed)
    sampled = candidates_df.sample(
        n=min(len(candidates_df), int(n_queries * 1.5)),
        random_state=seed,
    )
    kept: list[tuple[int, str, int]] = []
    tried = 0
    failed_nothing = 0
    for _, row in sampled.iterrows():
        tried += 1
        q = extract_body_query(row["abstract"], row["body"], rng)
        if q is None:
            failed_nothing += 1
            continue
        kept.append((len(kept), q, int(row["candidate_id"])))
        if len(kept) >= n_queries:
            break
    qdf = pd.DataFrame(kept, columns=["query_id", "query_text", "gold_candidate_id"])
    stats = {
        "tried": tried,
        "no_suitable_sentence": failed_nothing,
        "final_queries": len(kept),
    }
    log.info("Query extraction: tried=%d, found=%d, failed=%d (%.1f%%)",
             tried, len(kept), failed_nothing,
             100 * failed_nothing / max(tried, 1))
    return qdf, stats


def write_outputs(
    candidates_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    query_stats: dict,
    out_dir: Path,
    args,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Truncate body before saving; model will still see the long body within
    # the 8K context window.  Keep under MAX_BODY_CHARS to bound file size.
    candidates_df["body"] = candidates_df["body"].str.slice(0, MAX_BODY_CHARS)
    pq.write_table(
        pa.Table.from_pandas(candidates_df, preserve_index=False),
        out_dir / "candidates.parquet",
        compression="zstd",
    )
    log.info("Wrote %s (%d rows)", out_dir / "candidates.parquet", len(candidates_df))

    pq.write_table(
        pa.Table.from_pandas(queries_df, preserve_index=False),
        out_dir / "queries.parquet",
        compression="zstd",
    )
    log.info("Wrote %s (%d rows)", out_dir / "queries.parquet", len(queries_df))

    body_lens = candidates_df["body"].str.len()
    info = {
        "build_params": vars(args),
        "task": "body_fact_retrieval",
        "holdout": {
            "candidate_citation_count_range": [HOLDOUT_MIN_CITES, HOLDOUT_MAX_CITES],
            "stage2_candidate_min_citations": 5,
            "guarantee": "every candidate paper was structurally excluded from Stage 2 Signal-A/B (citation_count <= 4 < 5).",
        },
        "query_extraction": {
            "body_region": "middle third (skip intro / conclusion)",
            "min_query_chars": MIN_QUERY_CHARS,
            "max_query_chars": MAX_QUERY_CHARS,
            "abstract_trigram_overlap_max": TRIGRAM_OVERLAP_MAX,
            "stats": query_stats,
        },
        "candidate_count": int(len(candidates_df)),
        "query_count": int(len(queries_df)),
        "field_distribution": candidates_df["field_of_study"].value_counts().to_dict(),
        "body_char_length_quantiles": {
            str(q): int(body_lens.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "query_char_length_quantiles": {
            str(q): int(queries_df["query_text"].str.len().quantile(q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, default=str))
    log.info("Wrote dataset_info.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalake", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-candidates", type=int, default=10000)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--memory-gb", type=int, default=40)
    args = ap.parse_args()

    random.seed(args.seed)

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_gb}GB'")
    con.execute(f"ATTACH '{args.datalake}' AS dl (READ_ONLY)")
    con.execute("CREATE SCHEMA IF NOT EXISTS mem")
    log.info("DuckDB configured with threads=%d memory_limit=%dGB", args.threads, args.memory_gb)

    import time
    log.info("Step 1/4: anchor corpusids with citation_count in [%d, %d] ...",
             HOLDOUT_MIN_CITES, HOLDOUT_MAX_CITES)
    t = time.time()
    con.execute(ANCHOR_CORPUSIDS_QUERY.format(
        min_cites=HOLDOUT_MIN_CITES,
        max_cites=HOLDOUT_MAX_CITES,
    ))
    n1 = con.execute("SELECT COUNT(*) FROM mem.anchor_corpusids").fetchone()[0]
    log.info("  anchor_corpusids: %d papers [%.1fs]", n1, time.time() - t)

    log.info("Step 2/4: join abstracts (length >= 50) ...")
    t = time.time()
    con.execute(ANCHOR_WITH_ABSTRACT_QUERY)
    n2 = con.execute("SELECT COUNT(*) FROM mem.anchor_with_abstract").fetchone()[0]
    log.info("  anchor_with_abstract: %d papers [%.1fs]", n2, time.time() - t)

    log.info("Step 3a/4: s2orc body-length sidecar (length only, no body_text) ...")
    t = time.time()
    con.execute(S2ORC_LENGTHS_QUERY.format(min_body=MIN_BODY_CHARS, max_body=MAX_BODY_CHARS))
    nlen = con.execute("SELECT COUNT(*) FROM mem.s2orc_lengths").fetchone()[0]
    log.info("  s2orc_lengths (in-range): %d papers [%.1fs]", nlen, time.time() - t)

    log.info("Step 3b/4: candidates_all = anchor_with_abstract ∩ s2orc_lengths ...")
    t = time.time()
    con.execute(CANDIDATE_QUERY)
    n3 = con.execute("SELECT COUNT(*) FROM mem.candidates_all").fetchone()[0]
    log.info("  candidates_all: %d papers [%.1fs]", n3, time.time() - t)

    log.info("Step 4/4: sample %d candidates + build queries ...", args.n_candidates)
    # Sample metadata; body_text fetched separately for the sampled ids only.
    meta_df = sample_candidates_meta(con, args.n_candidates, args.seed)
    log.info("Fetching body_text for %d sampled candidates ...", len(meta_df))
    t = time.time()
    bodies_df = con.execute(FETCH_BODY_TEXTS_QUERY).fetchdf()
    log.info("  body_text fetched: %d rows [%.1fs]", len(bodies_df), time.time() - t)
    candidates_df = meta_df.merge(bodies_df, on="candidate_id", how="inner")
    log.info("Final candidates (with body): %d", len(candidates_df))
    queries_df, query_stats = build_queries(candidates_df, args.n_queries, args.seed)
    write_outputs(candidates_df, queries_df, query_stats, Path(args.output), args)


if __name__ == "__main__":
    main()
