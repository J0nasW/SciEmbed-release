"""Signal B: Citation context → document pairs for contrastive training.

Uses citation context sentences as natural search queries for cited papers.
When a scientist writes "Smith et al. developed a novel approach to protein
folding [23]", that sentence is a high-quality query for paper [23].

Source: s2ag.citations (1.01B contexts, 140M from influential citations).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

from sciembed.data.datalake import DatalakeConnection

logger = logging.getLogger(__name__)

# SQL queries

# Count contexts available per filter criteria
COUNT_CONTEXTS_QUERY = """
SELECT COUNT(*)
FROM s2ag.citations
WHERE contexts IS NOT NULL
  AND len(contexts) > 0
  {influential_filter}
"""

# Paginated context extraction — one page at a time to stay memory-bounded
CONTEXTS_PAGE_QUERY = """
SELECT
    c.citingcorpusid,
    c.citedcorpusid,
    c.contexts,
    c.intents
FROM s2ag.citations c
WHERE c.contexts IS NOT NULL
  AND len(c.contexts) > 0
  {influential_filter}
LIMIT {limit} OFFSET {offset}
"""

# Paper lookup for formatting documents
PAPER_LOOKUP_QUERY = """
SELECT p.corpusid, p.title, SUBSTRING(a.abstract, 1, {max_chars}) AS abstract
FROM s2ag.papers p
JOIN s2ag.abstracts a ON p.corpusid = a.corpusid
WHERE p.corpusid IN ({ids})
  AND p.title IS NOT NULL
  AND a.abstract IS NOT NULL
  AND LENGTH(a.abstract) >= 50
"""


def _filter_context(text: str, min_chars: int, max_chars: int) -> str | None:
    """Filter and clean a single citation context sentence.

    Returns None if the context doesn't meet quality thresholds.
    """
    if not text:
        return None

    text = text.strip()

    # Length filter
    if len(text) < min_chars or len(text) > max_chars:
        return None

    # Basic quality: must contain at least 3 words
    if len(text.split()) < 3:
        return None

    # Remove contexts that are just reference lists or numbers
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return None

    return text


def build_context_pairs(cfg: DictConfig) -> dict[str, Any]:
    """Generate citation context → cited paper pairs.

    Each (context_sentence, cited_paper) becomes a training pair:
        search_query: {context sentence}
        search_document: {cited_title}. {cited_abstract}

    Args:
        cfg: CitationContextsConfig (as DictConfig).

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

    influential_filter = "AND isinfluential = true" if cfg.influential_only else ""

    # Count total citations with contexts
    count_q = COUNT_CONTEXTS_QUERY.format(influential_filter=influential_filter)
    total_citations = dl.query(count_q)[0][0]
    logger.info("Total citations with contexts: %d", total_citations)

    target = cfg.num_pairs
    page_size = cfg.page_size
    shard_size = cfg.shard_size
    min_context_chars = cfg.min_context_chars
    max_context_chars = cfg.max_context_chars
    max_abstract_chars = cfg.max_abstract_chars
    sample_rate = min(1.0, target / max(total_citations, 1) * 2.0)  # oversample then cap

    pair_count = 0
    shard_idx = 0
    shard_data: dict[str, list] = {
        "query": [],
        "document": [],
        "citing_id": [],
        "cited_id": [],
    }

    def _flush_shard() -> None:
        nonlocal shard_idx, shard_data
        if not shard_data["query"]:
            return
        table = pa.table(shard_data)
        shard_path = output_dir / f"context_pairs_{shard_idx:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        logger.info("  Shard %d: %d pairs", shard_idx, len(shard_data["query"]))
        shard_idx += 1
        shard_data = {k: [] for k in shard_data}

    # Cache for paper lookups to avoid repeated queries
    paper_cache: dict[int, dict[str, str]] = {}
    pending_lookup: set[int] = set()

    def _lookup_papers(ids: set[int]) -> None:
        """Batch-lookup papers and cache results."""
        ids_to_fetch = {i for i in ids if i is not None} - set(paper_cache.keys())
        if not ids_to_fetch:
            return

        # DuckDB has parameter limits, process in chunks
        id_list = list(ids_to_fetch)
        for chunk_start in range(0, len(id_list), 10_000):
            chunk = id_list[chunk_start : chunk_start + 10_000]
            ids_str = ",".join(str(i) for i in chunk)
            query = PAPER_LOOKUP_QUERY.format(max_chars=max_abstract_chars, ids=ids_str)
            rows = dl.query(query)
            for row in rows:
                paper_cache[row[0]] = {"title": row[1], "abstract": row[2]}

    try:
        offset = 0
        pbar = tqdm(total=target, desc="Context pairs", unit="pairs")

        while offset < total_citations and pair_count < target:
            query = CONTEXTS_PAGE_QUERY.format(
                influential_filter=influential_filter,
                limit=page_size,
                offset=offset,
            )
            rows = dl.query(query)
            if not rows:
                break

            # Collect all cited paper IDs for batch lookup
            cited_ids = {row[1] for row in rows if row[1] is not None}

            _lookup_papers(cited_ids)

            for row in rows:
                if pair_count >= target:
                    break

                citing_id = row[0]
                cited_id = row[1]
                contexts = row[2]  # VARCHAR[] — list of context strings
                # intents stored but not used in this module (used by intent_triplets.py)

                if cited_id not in paper_cache:
                    continue

                cited_paper = paper_cache[cited_id]

                # Subsample to hit target
                if random.random() > sample_rate:
                    continue

                # Process each context sentence
                if not contexts:
                    continue

                for ctx_text in contexts:
                    if pair_count >= target:
                        break

                    cleaned = _filter_context(ctx_text, min_context_chars, max_context_chars)
                    if cleaned is None:
                        continue

                    shard_data["query"].append(f"search_query: {cleaned}")
                    shard_data["document"].append(
                        f"search_document: {cited_paper['title']}. {cited_paper['abstract']}"
                    )
                    shard_data["citing_id"].append(citing_id)
                    shard_data["cited_id"].append(cited_id)

                    pair_count += 1
                    pbar.update(1)

                    if len(shard_data["query"]) >= shard_size:
                        _flush_shard()

            offset += page_size

        _flush_shard()
        pbar.close()

    finally:
        dl.close()

    stats = {
        "total_pairs": pair_count,
        "num_shards": shard_idx,
        "output_dir": str(output_dir),
        "total_citations_scanned": offset,
        "paper_cache_size": len(paper_cache),
    }
    logger.info("Generated %d context pairs in %d shards", pair_count, shard_idx)
    return stats
