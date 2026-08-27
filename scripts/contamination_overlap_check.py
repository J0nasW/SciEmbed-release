"""Contamination / overlap audit for SciEmbed against SciRepEval.

Computes, for every SciRepEval test task:
  (a) size of the test-side paper-id set,
  (b) intersection with the Stage 1 MLM fulltext universe,
  (c) intersection with the Stage 2 Signal-A/B candidate universe (abstract >=50 chars,
      citation_count >= 5) — this is the upper bound on paper ids that could have
      appeared in any Stage 2 training triplet,
  (d) for proximity / adhoc-search tasks (SciDocs Cite/Co-Cite/Co-Read, Same
      Author, Highly Influential Citations, RELISH, Search, NFCorpus, TRECCoVID),
      a second column reporting edge-level overlap between (query_id, cand_id)
      positive edges and the s2ag citation table restricted to the Signal-A
      filter (candidate papers on both sides).

Outputs one JSON row per task to <output>.jsonl and a Markdown summary to
<output>.  The datalake is opened read-only; all derived universes live in an
attached in-memory database.

Usage:
    python scripts/contamination_overlap_check.py \\
        --datalake /path/to/data/science_datalake/datalake.duckdb \\
        --output docs/contamination_report.md
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

# SciRepEval tasks whose test split is labelled by paper_id.
PAPER_TASKS: list[tuple[str, str]] = [
    ("fos", "classification"),
    ("mesh_descriptors", "classification"),
    ("biomimicry", "classification"),
    ("drsm", "classification"),
    ("scidocs_mag", "classification"),
    ("scidocs_mesh", "classification"),
    ("cite_count", "regression"),
    ("pub_year", "regression"),
    ("hIndex", "regression"),
    ("peer_review_score", "regression"),
    ("tweet_mentions", "regression"),
]

# Proximity / search / retrieval tasks whose test split is labelled by
# (query_id, cand_id, score).  Positive = score > 0.
EDGE_TASKS: list[tuple[str, str]] = [
    ("search", "adhoc_search"),
    ("nfcorpus", "adhoc_search"),
    ("trec_covid", "adhoc_search"),
    ("same_author", "proximity"),
    ("high_influence_cite", "proximity"),
    ("scidocs_cite", "proximity"),
    ("scidocs_cocite", "proximity"),
    ("scidocs_read", "proximity"),
    ("scidocs_view", "proximity"),
    ("relish", "proximity"),
]


def _as_int_id(x: object, sha_to_corpus: dict[str, int] | None = None) -> int | None:
    """Best-effort coercion to int corpus-id.

    SciRepEval tasks use two id formats: (i) S2 corpus ids as decimal
    integers, (ii) SHA-1 hex strings for tasks sourced via the `paper_sha`
    identifier.  The `sha_to_corpus` mapping (built from the relevant meta
    config) resolves the hex form.  Strings that match neither are returned
    as None so the caller can count them separately.
    """
    if x is None:
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        pass
    if sha_to_corpus is not None and isinstance(x, str):
        return sha_to_corpus.get(x)
    return None


# Meta config that supplies the (doc_id/SHA -> corpus_id) mapping for a test task.
SHA_META: dict[str, str] = {
    "scidocs_mag": "scidocs_mag_mesh",
    "scidocs_mesh": "scidocs_mag_mesh",
    "scidocs_cite": "scidocs_view_cite_read",
    "scidocs_cocite": "scidocs_view_cite_read",
    "scidocs_read": "scidocs_view_cite_read",
    "scidocs_view": "scidocs_view_cite_read",
}


def load_sha_map(meta_config: str) -> dict[str, int]:
    log.info("Loading SHA->corpus_id map from meta config %s ...", meta_config)
    ds = load_dataset("allenai/scirepeval", meta_config, split="evaluation")
    m: dict[str, int] = {}
    for sha, cid in zip(ds["doc_id"], ds["corpus_id"]):
        if sha is not None and cid is not None:
            try:
                m[sha] = int(cid)
            except (TypeError, ValueError):
                continue
    log.info("  %s: %d sha->corpusid mappings", meta_config, len(m))
    return m


def collect_paper_task(
    config: str, sha_to_corpus: dict[str, int] | None = None
) -> tuple[set[int], int]:
    ds = load_dataset("allenai/scirepeval_test", config, split="test")
    ids: set[int] = set()
    non_int = 0
    for x in ds["paper_id"]:
        v = _as_int_id(x, sha_to_corpus)
        if v is None:
            if x is not None:
                non_int += 1
            continue
        ids.add(v)
    return ids, non_int


def collect_edge_task(
    config: str, sha_to_corpus: dict[str, int] | None = None
) -> tuple[set[int], set[tuple[int, int]], int]:
    ds = load_dataset("allenai/scirepeval_test", config, split="test")
    paper_ids: set[int] = set()
    pos_edges: set[tuple[int, int]] = set()
    non_int = 0
    for q, c, s in zip(ds["query_id"], ds["cand_id"], ds["score"]):
        qi = _as_int_id(q, sha_to_corpus)
        ci = _as_int_id(c, sha_to_corpus)
        if qi is None or ci is None:
            if q is not None or c is not None:
                non_int += 1
            continue
        paper_ids.add(qi)
        paper_ids.add(ci)
        try:
            if float(s) > 0:
                pos_edges.add((qi, ci))
        except (TypeError, ValueError):
            continue
    return paper_ids, pos_edges, non_int


def materialise_universes(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Materialise the two *paper-id* universes (Signal-A pool and MLM pool).

    The edge universe is *not* materialised: it would be ~130M rows and is not
    needed because each per-task edge-overlap check can push predicates down
    to the s2ag.citations Parquet directly, scanning only edges whose endpoints
    lie in the task's small (~10K) paper-id set.
    """
    log.info("Materialising Signal-A/B candidate universe (abstract + cites>=5) ...")
    con.execute("DROP TABLE IF EXISTS mem.sci_train_candidates")
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

    log.info("Materialising Stage-1 MLM fulltext universe (via DOI) ...")
    con.execute("DROP TABLE IF EXISTS mem.mlm_dois")
    con.execute(
        """
        CREATE TABLE mem.mlm_dois AS
        SELECT DISTINCT lower(doi) AS doi
        FROM dl.fulltext.papers
        WHERE has_full_text = true
          AND text_length BETWEEN 1000 AND 500000
          AND doi IS NOT NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS mem.sci_mlm_universe")
    con.execute(
        """
        CREATE TABLE mem.sci_mlm_universe AS
        SELECT DISTINCT p.corpusid
        FROM dl.s2ag.papers p
        JOIN mem.mlm_dois m ON lower(p.doi) = m.doi
        WHERE p.doi IS NOT NULL
        """
    )
    n_mlm = con.execute("SELECT COUNT(*) FROM mem.sci_mlm_universe").fetchone()[0]

    # Keep the edge-universe cardinality as a reported number; compute via a
    # streaming count with no intermediate materialisation.
    log.info("Counting citation-edge universe (streaming, no materialisation) ...")
    n_edges = con.execute(
        """
        SELECT COUNT(*)
        FROM dl.s2ag.citations c
        WHERE c.citingcorpusid IN (SELECT corpusid FROM mem.sci_train_candidates)
          AND c.citedcorpusid  IN (SELECT corpusid FROM mem.sci_train_candidates)
        """
    ).fetchone()[0]

    return {
        "signal_a_candidate_universe": n_sig,
        "mlm_fulltext_universe": n_mlm,
        "train_edge_universe": n_edges,
    }


def paper_overlap(
    con: duckdb.DuckDBPyConnection,
    paper_ids: set[int],
    universe_table: str,
) -> int:
    df = pd.DataFrame({"corpusid": list(paper_ids)}, dtype="int64")
    con.register("probe_ids", df)
    try:
        return con.execute(
            f"SELECT COUNT(*) FROM probe_ids p JOIN {universe_table} u USING (corpusid)"
        ).fetchone()[0]
    finally:
        con.unregister("probe_ids")


def edge_overlap(
    con: duckdb.DuckDBPyConnection,
    edges: set[tuple[int, int]],
) -> int:
    """Count positive (src, dst) pairs that appear in dl.s2ag.citations
    restricted to endpoints in the Signal-A candidate universe.  The filter
    uses a ~10K-entry endpoint-ids set registered on the connection, which
    DuckDB can push into the Parquet scan for near-linear scaling with core
    count."""
    if not edges:
        return 0
    df = pd.DataFrame(list(edges), columns=["src", "dst"], dtype="int64")
    endpoint_ids = pd.DataFrame(
        {"id": pd.unique(df[["src", "dst"]].values.ravel())}, dtype="int64"
    )
    con.register("probe_edges", df)
    con.register("probe_endpoints", endpoint_ids)
    try:
        # First narrow the billion-row citations table to candidate edges
        # whose endpoints lie in the task's ~10K-id endpoint set; this scans
        # only the relevant row-groups of the Parquet file.  Then restrict
        # further to the Signal-A candidate universe and join to the task's
        # positive edges in either direction.
        return con.execute(
            """
            WITH task_citations AS (
                SELECT citingcorpusid AS src, citedcorpusid AS dst
                FROM dl.s2ag.citations c
                WHERE c.citingcorpusid IN (SELECT id FROM probe_endpoints)
                  AND c.citedcorpusid  IN (SELECT id FROM probe_endpoints)
            ),
            task_citations_in_candidates AS (
                SELECT src, dst FROM task_citations
                WHERE src IN (SELECT corpusid FROM mem.sci_train_candidates)
                  AND dst IN (SELECT corpusid FROM mem.sci_train_candidates)
            )
            SELECT COUNT(DISTINCT (p.src, p.dst))
            FROM probe_edges p
            JOIN task_citations_in_candidates t
              ON (t.src = p.src AND t.dst = p.dst)
                OR (t.src = p.dst AND t.dst = p.src)
            """
        ).fetchone()[0]
    finally:
        con.unregister("probe_edges")
        con.unregister("probe_endpoints")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalake", required=True)
    ap.add_argument("--output", default="docs/contamination_report.md")
    args = ap.parse_args()

    # Open an in-memory DB as the main connection, then ATTACH the datalake
    # in read-only mode.  Derived universes live in the in-memory DB (schema
    # `mem`) so the datalake is never mutated.
    con = duckdb.connect(":memory:")
    # Saturate local cores and raise memory budget so DuckDB's parallel
    # hash-joins actually scale.  24 cores x ~1.5 GB working-set per core is
    # well within the 64+ GB typical workstation envelope; cap at 40 GB so
    # a hot OOM does not take down the whole session.
    import os
    threads = int(os.environ.get("DUCKDB_THREADS", os.cpu_count() or 8))
    mem_gb = int(os.environ.get("DUCKDB_MEMORY_GB", 40))
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{mem_gb}GB'")
    con.execute(f"ATTACH '{args.datalake}' AS dl (READ_ONLY)")
    con.execute("CREATE SCHEMA IF NOT EXISTS mem")
    log.info("DuckDB configured with threads=%d memory_limit=%dGB", threads, mem_gb)

    universes = materialise_universes(con)
    log.info("Universes: %s", universes)

    # Pre-load SHA->corpus_id maps for any tasks that need them, deduplicated.
    sha_maps: dict[str, dict[str, int]] = {}
    for meta in set(SHA_META.values()):
        sha_maps[meta] = load_sha_map(meta)

    rows: list[dict] = []
    for config, kind in PAPER_TASKS:
        t0 = time.time()
        sha_map = sha_maps.get(SHA_META.get(config, ""), None)
        try:
            ids, non_int = collect_paper_task(config, sha_map)
        except Exception as e:
            log.warning("Failed to load %s: %s", config, e)
            continue
        n = len(ids)
        sig_ov = paper_overlap(con, ids, "mem.sci_train_candidates") if n else 0
        mlm_ov = paper_overlap(con, ids, "mem.sci_mlm_universe") if n else 0
        rows.append(
            dict(
                task=config,
                task_kind=kind,
                n_papers=n,
                non_int_ids=non_int,
                signal_ab_candidate_overlap=sig_ov,
                signal_ab_overlap_pct=round(100 * sig_ov / max(n, 1), 2),
                mlm_fulltext_overlap=mlm_ov,
                mlm_overlap_pct=round(100 * mlm_ov / max(n, 1), 2),
                n_edges=None,
                train_edge_overlap=None,
                edge_overlap_pct=None,
            )
        )
        log.info(
            "%s (%s): n=%d (non-int skipped=%d) sig_ab=%d (%.1f%%) mlm=%d (%.1f%%) [%.1fs]",
            config, kind, n, non_int, sig_ov, rows[-1]["signal_ab_overlap_pct"],
            mlm_ov, rows[-1]["mlm_overlap_pct"], time.time() - t0,
        )

    for config, kind in EDGE_TASKS:
        t0 = time.time()
        sha_map = sha_maps.get(SHA_META.get(config, ""), None)
        try:
            ids, edges, non_int = collect_edge_task(config, sha_map)
        except Exception as e:
            log.warning("Failed to load %s: %s", config, e)
            continue
        n = len(ids)
        sig_ov = paper_overlap(con, ids, "mem.sci_train_candidates") if n else 0
        mlm_ov = paper_overlap(con, ids, "mem.sci_mlm_universe") if n else 0
        e_ov = edge_overlap(con, edges)
        rows.append(
            dict(
                task=config,
                task_kind=kind,
                n_papers=n,
                non_int_ids=non_int,
                signal_ab_candidate_overlap=sig_ov,
                signal_ab_overlap_pct=round(100 * sig_ov / max(n, 1), 2),
                mlm_fulltext_overlap=mlm_ov,
                mlm_overlap_pct=round(100 * mlm_ov / max(n, 1), 2),
                n_edges=len(edges),
                train_edge_overlap=e_ov,
                edge_overlap_pct=round(100 * e_ov / max(len(edges), 1), 2),
            )
        )
        log.info(
            "%s (%s): n_papers=%d (non-int=%d) sig_ab=%d (%.1f%%) mlm=%d (%.1f%%) edges=%d ov=%d (%.1f%%) [%.1fs]",
            config, kind, n, non_int, sig_ov, rows[-1]["signal_ab_overlap_pct"],
            mlm_ov, rows[-1]["mlm_overlap_pct"], len(edges), e_ov,
            rows[-1]["edge_overlap_pct"], time.time() - t0,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# Contamination / overlap report\n\n")
        f.write(
            "Structural, *unlabeled* paper-id and edge overlap between each SciRepEval "
            "test task and two upper-bound training universes.  No downstream labels "
            "(F1 targets, MeSH / MAG codes, regression targets, reviewer identities, "
            "etc.) are part of the training signal, so these numbers capture only "
            "exposure of a paper's title + abstract (or full-text, for the MLM column) "
            "to the model.\n\n"
            "- **Signal A+B candidate universe**: s2ag papers with non-empty abstract "
            "(>=50 characters) and `citation_count >= 5`.  Maximal set of corpus ids "
            "that could appear as an anchor, positive, or hard-negative in any Stage 2 "
            "training triplet.  Actual Stage 2 training is a subsample of edges over "
            "this pool.\n"
            "- **MLM fulltext universe**: union of PubMed Central, S2ORC, arXiv, and "
            "peS2o papers seen by Stage 1 MLM pretraining.  Self-supervised exposure "
            "only; no downstream labels leak.\n"
            "- **Train-edge universe**: s2ag citation edges with both endpoints in the "
            "Signal-A/B candidate universe.  For proximity tasks, edge-overlap counts "
            "positive (query, cand) test pairs that appear (in either direction) in "
            "this universe.\n\n"
        )
        f.write("Training universes:\n\n")
        for k, v in universes.items():
            f.write(f"- `{k}`: {v:,}\n")
        f.write("\n")
        f.write(
            "| Task | Kind | n papers | non-int ids | Sig A+B ∩ | % | MLM ∩ | % | n +edges | Edge ∩ | % |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for r in rows:
            n_edges_cell = "" if r["n_edges"] is None else f"{r['n_edges']:,}"
            e_ov_cell = "" if r["train_edge_overlap"] is None else f"{r['train_edge_overlap']:,}"
            e_pct_cell = "" if r["edge_overlap_pct"] is None else str(r["edge_overlap_pct"])
            f.write(
                f"| {r['task']} | {r['task_kind']} | {r['n_papers']:,} | {r.get('non_int_ids', 0):,} | "
                f"{r['signal_ab_candidate_overlap']:,} | {r['signal_ab_overlap_pct']} | "
                f"{r['mlm_fulltext_overlap']:,} | {r['mlm_overlap_pct']} | "
                f"{n_edges_cell} | {e_ov_cell} | {e_pct_cell} |\n"
            )
        f.write(
            "\nThe `non-int ids` column counts test-split rows whose `paper_id` / "
            "`query_id` / `cand_id` is a SHA-1 hex string rather than an S2 corpus id "
            "(used by a subset of tasks, including FoS, MAG, and MeSH). These rows "
            "are excluded from the overlap counts; the overlap numbers should be "
            "read as a *lower bound* on paper-level coverage for any such task.\n\n"
        )
    log.info("Wrote Markdown report to %s", out)

    jsonl_out = out.with_suffix(".jsonl")
    with jsonl_out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log.info("Wrote JSONL to %s", jsonl_out)


if __name__ == "__main__":
    main()
