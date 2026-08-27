"""Multi-task instruction-aware pair generator.

Generates training pairs with instruction prefixes for different embedding tasks:
- Search: search_query: / search_document:
- Classification: classify:
- Clustering: cluster:
- Similarity: similarity:
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

TLDR_QUERY = """
SELECT t.corpusid, t.text AS tldr, p.title
FROM s2ag.tldrs t
JOIN s2ag.papers p ON t.corpusid = p.corpusid
WHERE t.text IS NOT NULL AND LENGTH(t.text) >= 20
"""

TOPIC_CLASSIFICATION_QUERY = """
SELECT
    p.title,
    SUBSTRING(a.abstract, 1, 512) AS abstract_preview,
    wt.topic_display_name AS topic_name,
    t.field_display_name AS field_name
FROM s2ag.papers p
JOIN s2ag.abstracts a ON p.corpusid = a.corpusid
JOIN openalex.works oa ON 'https://doi.org/' || p.externalids.DOI = oa.doi
JOIN openalex.works_topics wt ON oa.id = wt.work_id
JOIN openalex.topics t ON wt.topic_id = t.id
WHERE p.externalids.DOI IS NOT NULL
  AND a.abstract IS NOT NULL
  AND LENGTH(a.abstract) >= 50
  AND wt.topic_display_name IS NOT NULL
"""

CLUSTERING_QUERY = """
SELECT
    p.title,
    SUBSTRING(a.abstract, 1, 512) AS abstract_preview,
    t.subfield_display_name AS subfield
FROM s2ag.papers p
JOIN s2ag.abstracts a ON p.corpusid = a.corpusid
JOIN openalex.works oa ON 'https://doi.org/' || p.externalids.DOI = oa.doi
JOIN openalex.works_topics wt ON oa.id = wt.work_id
JOIN openalex.topics t ON wt.topic_id = t.id
WHERE p.externalids.DOI IS NOT NULL
  AND a.abstract IS NOT NULL
  AND LENGTH(a.abstract) >= 50
  AND t.subfield_display_name IS NOT NULL
"""

RELATED_WORKS_QUERY = """
SELECT
    p1.title AS title1,
    SUBSTRING(a1.abstract, 1, 512) AS abstract1,
    p2.title AS title2,
    SUBSTRING(a2.abstract, 1, 512) AS abstract2
FROM openalex.works_related_works rw
JOIN openalex.works ow1 ON rw.work_id = ow1.id
JOIN openalex.works ow2 ON rw.related_work_id = ow2.id
JOIN s2ag.papers p1 ON 'https://doi.org/' || p1.externalids.DOI = ow1.doi
JOIN s2ag.papers p2 ON 'https://doi.org/' || p2.externalids.DOI = ow2.doi
JOIN s2ag.abstracts a1 ON p1.corpusid = a1.corpusid
JOIN s2ag.abstracts a2 ON p2.corpusid = a2.corpusid
WHERE p1.externalids.DOI IS NOT NULL AND p2.externalids.DOI IS NOT NULL
  AND a1.abstract IS NOT NULL AND a2.abstract IS NOT NULL
  AND LENGTH(a1.abstract) >= 50 AND LENGTH(a2.abstract) >= 50
"""


def _write_pairs_parquet(
    pairs: list[dict[str, str]],
    output_dir: Path,
    prefix: str,
    shard_idx: int,
) -> None:
    """Write a batch of pairs to a Parquet shard."""
    if not pairs:
        return
    table = pa.table({
        "text1": [p["text1"] for p in pairs],
        "text2": [p["text2"] for p in pairs],
        "task": [p["task"] for p in pairs],
    })
    shard_path = output_dir / f"{prefix}_{shard_idx:05d}.parquet"
    pq.write_table(table, shard_path, compression="zstd")


def _build_search_pairs(
    dl: DatalakeConnection,
    output_dir: Path,
    target: int,
    triplets_dir: str | None,
) -> int:
    """Build search pairs from TLDRs and citation triplets.

    TLDR pairs: search_query: {tldr} → search_document: {title}. {abstract}
    """
    logger.info("Building search pairs (target: %d)", target)
    count = 0
    shard_idx = 0
    batch: list[dict[str, str]] = []

    # Use TLDRs as search queries
    for record_batch in tqdm(
        dl.stream_query(TLDR_QUERY, batch_size=100_000),
        desc="Search pairs (TLDRs)",
    ):
        tldrs = record_batch.column("tldr").to_pylist()
        titles = record_batch.column("title").to_pylist()

        for tldr, title in zip(tldrs, titles):
            if count >= target:
                break
            if not tldr or not title:
                continue

            batch.append({
                "text1": f"search_query: {tldr}",
                "text2": f"search_document: {title}",
                "task": "search",
            })
            count += 1

            if len(batch) >= 500_000:
                _write_pairs_parquet(batch, output_dir, "search", shard_idx)
                shard_idx += 1
                batch = []

        if count >= target:
            break

    if batch:
        _write_pairs_parquet(batch, output_dir, "search", shard_idx)

    logger.info("Built %d search pairs", count)
    return count


def _build_classification_pairs(
    dl: DatalakeConnection,
    output_dir: Path,
    target: int,
) -> int:
    """Build classification pairs: classify: {title}. {abstract} → {topic_name}."""
    logger.info("Building classification pairs (target: %d)", target)
    count = 0
    shard_idx = 0
    batch: list[dict[str, str]] = []

    for record_batch in tqdm(
        dl.stream_query(TOPIC_CLASSIFICATION_QUERY, batch_size=100_000),
        desc="Classification pairs",
    ):
        titles = record_batch.column("title").to_pylist()
        abstracts = record_batch.column("abstract_preview").to_pylist()
        topics = record_batch.column("topic_name").to_pylist()
        fields = record_batch.column("field_name").to_pylist()

        for title, abstract, topic, field_name in zip(titles, abstracts, topics, fields):
            if count >= target:
                break
            if not title or not topic:
                continue

            text = f"{title}. {abstract}" if abstract else title
            label = f"{field_name} > {topic}" if field_name else topic

            batch.append({
                "text1": f"classify: {text}",
                "text2": label,
                "task": "classification",
            })
            count += 1

            if len(batch) >= 500_000:
                _write_pairs_parquet(batch, output_dir, "classification", shard_idx)
                shard_idx += 1
                batch = []

        if count >= target:
            break

    if batch:
        _write_pairs_parquet(batch, output_dir, "classification", shard_idx)

    logger.info("Built %d classification pairs", count)
    return count


def _build_clustering_pairs(
    dl: DatalakeConnection,
    output_dir: Path,
    target: int,
) -> int:
    """Build clustering pairs: same-subfield paper pairs.

    cluster: {title1}. {abstract1} ↔ cluster: {title2}. {abstract2}
    """
    logger.info("Building clustering pairs (target: %d)", target)

    # Collect papers by subfield
    subfield_papers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    max_per_subfield = 50_000

    for record_batch in tqdm(
        dl.stream_query(CLUSTERING_QUERY, batch_size=100_000),
        desc="Loading subfield papers",
    ):
        titles = record_batch.column("title").to_pylist()
        abstracts = record_batch.column("abstract_preview").to_pylist()
        subfields = record_batch.column("subfield").to_pylist()

        for title, abstract, subfield in zip(titles, abstracts, subfields):
            if not title or not subfield:
                continue
            if len(subfield_papers[subfield]) < max_per_subfield:
                subfield_papers[subfield].append((title, abstract or ""))

    # Generate pairs from same subfield
    count = 0
    shard_idx = 0
    batch: list[dict[str, str]] = []

    subfields = list(subfield_papers.keys())
    random.shuffle(subfields)

    for subfield in subfields:
        papers = subfield_papers[subfield]
        if len(papers) < 2:
            continue

        # Sample pairs
        pairs_per_subfield = min(target // max(len(subfields), 1), len(papers) * (len(papers) - 1) // 2)
        for _ in range(pairs_per_subfield):
            if count >= target:
                break
            p1, p2 = random.sample(papers, 2)

            batch.append({
                "text1": f"cluster: {p1[0]}. {p1[1]}",
                "text2": f"cluster: {p2[0]}. {p2[1]}",
                "task": "clustering",
            })
            count += 1

            if len(batch) >= 500_000:
                _write_pairs_parquet(batch, output_dir, "clustering", shard_idx)
                shard_idx += 1
                batch = []

        if count >= target:
            break

    if batch:
        _write_pairs_parquet(batch, output_dir, "clustering", shard_idx)

    logger.info("Built %d clustering pairs from %d subfields", count, len(subfields))
    return count


def _build_similarity_pairs(
    dl: DatalakeConnection,
    output_dir: Path,
    target: int,
) -> int:
    """Build similarity pairs from related works.

    similarity: {title1}. {abstract1} ↔ similarity: {title2}. {abstract2}
    """
    logger.info("Building similarity pairs (target: %d)", target)
    count = 0
    shard_idx = 0
    batch: list[dict[str, str]] = []

    for record_batch in tqdm(
        dl.stream_query(RELATED_WORKS_QUERY, batch_size=100_000),
        desc="Similarity pairs",
    ):
        titles1 = record_batch.column("title1").to_pylist()
        abstracts1 = record_batch.column("abstract1").to_pylist()
        titles2 = record_batch.column("title2").to_pylist()
        abstracts2 = record_batch.column("abstract2").to_pylist()

        for t1, a1, t2, a2 in zip(titles1, abstracts1, titles2, abstracts2):
            if count >= target:
                break
            if not t1 or not t2:
                continue

            batch.append({
                "text1": f"similarity: {t1}. {a1 or ''}",
                "text2": f"similarity: {t2}. {a2 or ''}",
                "task": "similarity",
            })
            count += 1

            if len(batch) >= 500_000:
                _write_pairs_parquet(batch, output_dir, "similarity", shard_idx)
                shard_idx += 1
                batch = []

        if count >= target:
            break

    if batch:
        _write_pairs_parquet(batch, output_dir, "similarity", shard_idx)

    logger.info("Built %d similarity pairs", count)
    return count


def build_instruction_pairs(cfg: DictConfig) -> dict[str, Any]:
    """Generate all instruction-aware training pairs.

    Args:
        cfg: InstructionPairsConfig (as DictConfig).

    Returns:
        Statistics dict.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dl = DatalakeConnection(
        db_path=cfg.datalake.db_path,
        read_only=cfg.datalake.read_only,
        threads=cfg.datalake.threads,
        memory_limit=cfg.datalake.memory_limit,
    )

    stats = {}

    try:
        stats["search"] = _build_search_pairs(
            dl, output_dir, cfg.search_pairs, cfg.triplets_dir
        )
        stats["classification"] = _build_classification_pairs(
            dl, output_dir, cfg.classification_pairs
        )
        stats["clustering"] = _build_clustering_pairs(
            dl, output_dir, cfg.clustering_pairs
        )
        stats["similarity"] = _build_similarity_pairs(
            dl, output_dir, cfg.similarity_pairs
        )
    finally:
        dl.close()

    stats["total"] = sum(stats.values())
    logger.info("Total instruction pairs: %d", stats["total"])
    return stats
