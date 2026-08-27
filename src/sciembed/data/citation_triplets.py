"""Citation triplet generator for contrastive training.

Generates (anchor, positive, hard_negative) triplets from the citation graph,
using S2AG citations, papers, abstracts, and S2AG fields of study.

Supports multi-node parallelism via partition(worker_id, num_workers):
  1. Prep job: materialize paper pool + forward citation index (shared)
  2. Worker jobs: each generates triplets from its partition of citation edges
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

from sciembed.data.datalake import DatalakeConnection

logger = logging.getLogger(__name__)

# SQL queries

PAPER_POOL_QUERY = """
SELECT
    p.corpusid,
    p.title,
    SUBSTRING(a.abstract, 1, {max_abstract_chars}) AS abstract_preview,
    p.externalids.DOI AS doi,
    p.citationcount,
    p.s2fieldsofstudy[1].category AS field_of_study
FROM s2ag.papers p
JOIN s2ag.abstracts a ON p.corpusid = a.corpusid
WHERE a.abstract IS NOT NULL
  AND LENGTH(a.abstract) >= 50
"""

CITATION_EDGES_QUERY = """
SELECT citingcorpusid, citedcorpusid
FROM s2ag.citations
WHERE {influential_filter}
"""

CITATION_EDGES_PARTITIONED_QUERY = """
SELECT citingcorpusid, citedcorpusid
FROM s2ag.citations
WHERE {influential_filter}
  AND citingcorpusid % {num_workers} = {worker_id}
"""


def _influential_filter(influential_only: bool) -> str:
    """Return the SQL WHERE clause fragment for citation quality tier."""
    if influential_only:
        return "isinfluential = true"
    return "isinfluential = false"


def format_anchor(title: str, abstract: str) -> str:
    """Format an anchor text with search_query prefix."""
    return f"search_query: {title}. {abstract}"


def format_document(title: str, abstract: str) -> str:
    """Format a document text with search_document prefix."""
    return f"search_document: {title}. {abstract}"


def materialize_paper_pool(
    dl: DatalakeConnection,
    output_path: Path,
    max_abstract_chars: int = 512,
) -> None:
    """Pre-compute paper pool with title + abstract + topic_id.

    Materializes ~30M papers into a Parquet file for efficient lookups.
    """
    logger.info("Materializing paper pool (this may take ~1h)...")
    query = PAPER_POOL_QUERY.format(max_abstract_chars=max_abstract_chars)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    for batch in tqdm(dl.stream_query(query, batch_size=500_000), desc="Paper pool"):
        table = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema, compression="zstd")
        writer.write_table(table)

    if writer:
        writer.close()
    logger.info("Paper pool materialized to %s", output_path)


def materialize_forward_citations(
    dl: DatalakeConnection,
    paper_pool_path: Path,
    output_path: Path,
    max_per_paper: int = 20,
    influential_only: bool = True,
) -> None:
    """Pre-compute forward citation index and save to Parquet.

    For each paper in the pool, stores the list of papers it cites (up to
    max_per_paper). Used by workers for two-hop hard negative sampling.

    Output schema: (corpusid: int64, cited_ids: list<int64>)
    """
    logger.info("Loading paper pool IDs for filtering...")
    pool_ids = set(
        pq.read_table(paper_pool_path, columns=["corpusid"])
        .column("corpusid")
        .to_pylist()
    )
    logger.info("Paper pool: %d papers", len(pool_ids))

    logger.info("Building forward citation index (influential_only=%s)...", influential_only)
    forward: dict[int, list[int]] = defaultdict(list)
    edge_count = 0

    query = CITATION_EDGES_QUERY.format(
        influential_filter=_influential_filter(influential_only),
    )
    for batch in tqdm(
        dl.stream_query(query, batch_size=1_000_000),
        desc="Forward citations",
    ):
        citing = batch.column("citingcorpusid").to_pylist()
        cited = batch.column("citedcorpusid").to_pylist()

        for a, b in zip(citing, cited):
            if a in pool_ids and b in pool_ids:
                if len(forward[a]) < max_per_paper:
                    forward[a].append(b)
                    edge_count += 1

    logger.info("Forward index: %d papers, %d edges", len(forward), edge_count)

    # Save to Parquet with list column
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_ids = list(forward.keys())
    cited_lists = [forward[k] for k in corpus_ids]
    table = pa.table({
        "corpusid": pa.array(corpus_ids, type=pa.int64()),
        "cited_ids": pa.array(cited_lists, type=pa.list_(pa.int64())),
    })
    pq.write_table(table, output_path, compression="zstd")
    logger.info("Forward citations saved to %s", output_path)


def load_forward_citations(path: Path) -> dict[int, list[int]]:
    """Load pre-materialized forward citation index from Parquet."""
    logger.info("Loading forward citations from %s", path)
    table = pq.read_table(path)
    corpus_ids = table.column("corpusid").to_pylist()
    cited_lists = table.column("cited_ids").to_pylist()
    result = dict(zip(corpus_ids, cited_lists))
    logger.info("Loaded forward citations for %d papers", len(result))
    return result


def load_paper_pool(pool_path: Path) -> tuple[dict[int, dict], dict[str, list[int]]]:
    """Load paper pool from Parquet and build field-of-study index.

    Returns:
        papers: {corpusid: {title, abstract_preview, field_of_study, citationcount}}
        field_index: {field_of_study: [corpusid, ...]}
    """
    logger.info("Loading paper pool from %s", pool_path)
    table = pq.read_table(
        pool_path,
        columns=["corpusid", "title", "abstract_preview", "field_of_study", "citationcount"],
    )

    papers = {}
    field_index: dict[str, list[int]] = defaultdict(list)

    for batch in table.to_batches(max_chunksize=500_000):
        corpusids = batch.column("corpusid").to_pylist()
        titles = batch.column("title").to_pylist()
        abstracts = batch.column("abstract_preview").to_pylist()
        fields = batch.column("field_of_study").to_pylist()
        citationcounts = batch.column("citationcount").to_pylist()

        for cid, title, abstract, fos, cc in zip(
            corpusids, titles, abstracts, fields, citationcounts
        ):
            papers[cid] = {
                "title": title or "",
                "abstract_preview": abstract or "",
                "field_of_study": fos,
                "citationcount": cc or 0,
            }
            if fos is not None:
                field_index[fos].append(cid)

    logger.info("Loaded %d papers, %d fields", len(papers), len(field_index))
    return papers, field_index


def sample_hard_negative(
    anchor_id: int,
    positive_id: int,
    field_of_study: str | None,
    papers: dict[int, dict],
    field_index: dict[str, list[int]],
    cited_set: set[int],
    negative_mix: dict[str, float],
    pool_keys: list[int] | None = None,
    forward_citations: dict[int, list[int]] | None = None,
) -> int | None:
    """Sample a hard negative paper ID.

    Strategy:
        same_topic fraction: same-field non-cited paper
        two_hop fraction: paper cited by the positive (2-hop neighbor)
        remaining: random paper from pool
    """
    roll = random.random()
    same_topic_thresh = negative_mix.get("same_topic", 0.5)
    two_hop_thresh = same_topic_thresh + negative_mix.get("two_hop", 0.3)

    if roll < same_topic_thresh and field_of_study is not None:
        # Same-field negative
        candidates = field_index.get(field_of_study, [])
        if len(candidates) > 100:
            for _ in range(20):
                neg = candidates[random.randint(0, len(candidates) - 1)]
                if neg != anchor_id and neg not in cited_set:
                    return neg

    elif roll < two_hop_thresh and forward_citations is not None:
        # Two-hop negative: papers cited by the positive paper
        pos_cites = forward_citations.get(positive_id)
        if pos_cites:
            # Sample without mutating the original list
            for _ in range(min(10, len(pos_cites))):
                neg = pos_cites[random.randint(0, len(pos_cites) - 1)]
                if neg != anchor_id and neg not in cited_set and neg in papers:
                    return neg

    # Random negative from pre-built key list
    if pool_keys is None:
        pool_keys = list(papers.keys())
    for _ in range(10):
        neg = pool_keys[random.randint(0, len(pool_keys) - 1)]
        if neg != anchor_id and neg != positive_id and neg not in cited_set:
            return neg

    return None


def prep_shared_data(cfg: DictConfig) -> dict[str, Any]:
    """Materialize paper pool and forward citation index (run once before workers).

    Args:
        cfg: CitationTripletsConfig (as DictConfig).

    Returns:
        Paths to materialized files.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dl = DatalakeConnection(
        db_path=cfg.datalake.db_path,
        read_only=cfg.datalake.read_only,
        threads=cfg.datalake.threads,
        memory_limit=cfg.datalake.memory_limit,
    )

    pool_path = Path(cfg.paper_pool_path) if cfg.paper_pool_path else output_dir / "paper_pool.parquet"
    fwd_path = output_dir / "forward_citations.parquet"

    try:
        if not pool_path.exists():
            materialize_paper_pool(dl, pool_path, max_abstract_chars=cfg.max_abstract_chars)
        else:
            logger.info("Paper pool already exists: %s", pool_path)

        if not fwd_path.exists():
            negative_mix = dict(cfg.negative_mix)
            if negative_mix.get("two_hop", 0) > 0:
                materialize_forward_citations(
                    dl, pool_path, fwd_path,
                    influential_only=cfg.influential_only,
                )
            else:
                logger.info("Skipping forward citations (two_hop=0 in negative_mix)")
        else:
            logger.info("Forward citations already exist: %s", fwd_path)
    finally:
        dl.close()

    return {"paper_pool": str(pool_path), "forward_citations": str(fwd_path)}


def build_triplets(
    cfg: DictConfig,
    worker_id: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Generate citation triplets and write to Parquet shards.

    Args:
        cfg: CitationTripletsConfig (as DictConfig).
        worker_id: If set, only process citation edges where
            citingcorpusid % num_workers == worker_id.
        num_workers: Total number of parallel workers.

    Returns:
        Statistics dict.
    """
    partitioned = worker_id is not None and num_workers is not None
    if partitioned:
        output_dir = Path(cfg.output_dir) / f"worker_{worker_id}"
        target = cfg.num_triplets // num_workers
    else:
        output_dir = Path(cfg.output_dir)
        target = cfg.num_triplets
    output_dir.mkdir(parents=True, exist_ok=True)

    dl = DatalakeConnection(
        db_path=cfg.datalake.db_path,
        read_only=cfg.datalake.read_only,
        threads=cfg.datalake.threads,
        memory_limit=cfg.datalake.memory_limit,
    )

    # Load shared pre-materialized data
    pool_path = Path(cfg.paper_pool_path) if cfg.paper_pool_path else Path(cfg.output_dir) / "paper_pool.parquet"
    if not pool_path.exists():
        materialize_paper_pool(dl, pool_path, max_abstract_chars=cfg.max_abstract_chars)

    papers, field_index = load_paper_pool(pool_path)

    # Load forward citations for two-hop negatives
    negative_mix = dict(cfg.negative_mix)
    forward_citations = None
    if negative_mix.get("two_hop", 0) > 0:
        fwd_path = Path(cfg.output_dir) / "forward_citations.parquet"
        if fwd_path.exists():
            forward_citations = load_forward_citations(fwd_path)
        else:
            logger.warning(
                "Forward citations not found at %s. Run 'build-triplets-prep' first "
                "for multi-node, or forward citations will be skipped.", fwd_path,
            )

    # Quality filter: minimum citation count for both papers
    min_citation_count = getattr(cfg, "min_citation_count", 0)

    # Choose edge query — partitioned or full
    influential_only = cfg.influential_only
    inf_filter = _influential_filter(influential_only)
    if partitioned:
        edge_query = CITATION_EDGES_PARTITIONED_QUERY.format(
            influential_filter=inf_filter,
            num_workers=num_workers, worker_id=worker_id,
        )
        logger.info(
            "Worker %d/%d: generating %d triplets (influential_only=%s, min_cite=%d)",
            worker_id, num_workers, target, influential_only, min_citation_count,
        )
    else:
        edge_query = CITATION_EDGES_QUERY.format(influential_filter=inf_filter)
        logger.info(
            "Generating %d triplets (single node, influential_only=%s, min_cite=%d)",
            target, influential_only, min_citation_count,
        )

    shard_size = cfg.shard_size

    # Pre-build pool keys list for fast random sampling
    pool_keys = list(papers.keys())
    logger.info("Pool keys: %d papers available for negative sampling", len(pool_keys))

    # Dedup: track seen (anchor_id, positive_id) pairs
    seen_pairs: set[tuple[int, int]] = set()

    triplet_count = 0
    shard_idx = 0
    shard_data: dict[str, list] = {
        "anchor": [],
        "positive": [],
        "negative": [],
        "anchor_id": [],
        "positive_id": [],
        "negative_id": [],
    }

    def _flush_shard() -> None:
        nonlocal shard_idx, shard_data
        if not shard_data["anchor"]:
            return
        table = pa.table(shard_data)
        shard_path = output_dir / f"triplets_{shard_idx:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        logger.info(
            "Wrote shard %d (%d triplets so far)", shard_idx, triplet_count
        )
        shard_idx += 1
        shard_data = {k: [] for k in shard_data}

    try:
        for batch in tqdm(
            dl.stream_query(edge_query, batch_size=1_000_000),
            desc=f"Citation edges (worker {worker_id})" if partitioned else "Citation edges",
        ):
            citing_ids = batch.column("citingcorpusid").to_pylist()
            cited_ids = batch.column("citedcorpusid").to_pylist()

            for citing_id, cited_id in zip(citing_ids, cited_ids):
                if triplet_count >= target:
                    break

                if citing_id not in papers or cited_id not in papers:
                    continue

                anchor_paper = papers[citing_id]
                positive_paper = papers[cited_id]

                # Quality filter: skip low-citation papers
                if min_citation_count > 0:
                    if (anchor_paper["citationcount"] < min_citation_count
                            or positive_paper["citationcount"] < min_citation_count):
                        continue

                # Dedup: skip if we've already seen this pair
                pair_key = (citing_id, cited_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if not anchor_paper["title"] or not positive_paper["title"]:
                    continue

                field_of_study = anchor_paper.get("field_of_study")
                cited_set = {cited_id}

                neg_id = sample_hard_negative(
                    citing_id,
                    cited_id,
                    field_of_study,
                    papers,
                    field_index,
                    cited_set,
                    negative_mix,
                    pool_keys=pool_keys,
                    forward_citations=forward_citations,
                )
                if neg_id is None or neg_id not in papers:
                    continue

                neg_paper = papers[neg_id]

                shard_data["anchor"].append(
                    format_anchor(anchor_paper["title"], anchor_paper["abstract_preview"])
                )
                shard_data["positive"].append(
                    format_document(positive_paper["title"], positive_paper["abstract_preview"])
                )
                shard_data["negative"].append(
                    format_document(neg_paper["title"], neg_paper["abstract_preview"])
                )
                shard_data["anchor_id"].append(citing_id)
                shard_data["positive_id"].append(cited_id)
                shard_data["negative_id"].append(neg_id)

                triplet_count += 1

                if len(shard_data["anchor"]) >= shard_size:
                    _flush_shard()

            if triplet_count >= target:
                break

        _flush_shard()

    finally:
        dl.close()

    stats = {
        "total_triplets": triplet_count,
        "num_shards": shard_idx,
        "output_dir": str(output_dir),
        "paper_pool_size": len(papers),
        "num_fields": len(field_index),
        "unique_pairs": len(seen_pairs),
        "two_hop_index_size": len(forward_citations) if forward_citations else 0,
    }
    if partitioned:
        stats["worker_id"] = worker_id
        stats["num_workers"] = num_workers
    logger.info("Generated %d triplets in %d shards", triplet_count, shard_idx)
    return stats
