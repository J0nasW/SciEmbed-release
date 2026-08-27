"""Signal C: Intent-conditioned citation context pairs for contrastive training.

Extends Signal B (citation contexts) with intent-type prefixes:
    cite_background: {context about theoretical grounding}  → {cited paper}
    cite_method:     {context about methodology adoption}   → {cited paper}
    cite_result:     {context about empirical findings}     → {cited paper}

This teaches the model to distinguish *why* papers are related, enabling
intent-specific retrieval (e.g., "find papers using similar methodology").

Source: s2ag.citations with intents + contexts co-populated.
Distribution: background ~80%, methodology ~18%, result ~4%.
Balancing: oversample methodology 3x and result 10x.
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

# SQL queries — streaming, no LIMIT/OFFSET

INTENT_CONTEXTS_STREAM_QUERY = """
SELECT
    c.citingcorpusid,
    c.citedcorpusid,
    c.contexts,
    c.intents
FROM s2ag.citations c
WHERE c.contexts IS NOT NULL
  AND len(c.contexts) > 0
  AND c.intents IS NOT NULL
  AND len(c.intents) > 0
  {influential_filter}
"""

PAPER_LOOKUP_QUERY = """
SELECT p.corpusid, p.title, SUBSTRING(a.abstract, 1, {max_chars}) AS abstract
FROM s2ag.papers p
JOIN s2ag.abstracts a ON p.corpusid = a.corpusid
WHERE p.corpusid IN ({ids})
  AND p.title IS NOT NULL
  AND a.abstract IS NOT NULL
  AND LENGTH(a.abstract) >= 50
"""

# Map S2AG intent labels to our prefix scheme
INTENT_PREFIX_MAP = {
    "background": "cite_background",
    "methodology": "cite_method",
    "result": "cite_result",
}

# Oversampling multipliers to balance the 80/18/4 distribution
DEFAULT_OVERSAMPLE = {
    "background": 1,
    "methodology": 3,
    "result": 10,
}


def _filter_context(text: str, min_chars: int, max_chars: int) -> str | None:
    """Filter and clean a citation context sentence."""
    if not text:
        return None
    text = text.strip()
    if len(text) < min_chars or len(text) > max_chars:
        return None
    if len(text.split()) < 3:
        return None
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return None
    return text


def _parse_intents(intents_raw: list[str] | str | None) -> list[str]:
    """Parse intent labels from the citation record.

    S2AG stores intents as a VARCHAR[] column. Each citation may have
    multiple intents (one per context sentence, roughly aligned by index).
    DuckDB may return nested lists depending on schema.
    """
    if intents_raw is None:
        return []
    if isinstance(intents_raw, str):
        return [i.strip().lower() for i in intents_raw.split(",") if i.strip()]
    # Flatten if nested (DuckDB may return list of lists)
    flat: list[str] = []
    for item in intents_raw:
        if isinstance(item, list):
            flat.extend(s for s in item if isinstance(s, str))
        elif isinstance(item, str):
            flat.append(item)
    return [i.strip().lower() for i in flat if i and i.strip()]


def _parse_contexts(contexts_raw) -> list[str]:
    """Parse contexts, handling potential nesting from DuckDB."""
    if contexts_raw is None:
        return []
    if isinstance(contexts_raw, str):
        return [contexts_raw]
    flat: list[str] = []
    for item in contexts_raw:
        if isinstance(item, list):
            flat.extend(s for s in item if isinstance(s, str))
        elif isinstance(item, str):
            flat.append(item)
    return flat


def build_intent_pairs(cfg: DictConfig) -> dict[str, Any]:
    """Generate intent-conditioned citation context pairs.

    Uses streaming (Arrow batches) instead of LIMIT/OFFSET for efficiency.
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
    oversample = dict(cfg.get("oversample", DEFAULT_OVERSAMPLE))

    target = cfg.num_pairs
    shard_size = cfg.shard_size
    min_context_chars = cfg.min_context_chars
    max_context_chars = cfg.max_context_chars
    max_abstract_chars = cfg.max_abstract_chars

    pair_count = 0
    intent_counts: dict[str, int] = defaultdict(int)
    shard_idx = 0
    shard_data: dict[str, list] = {
        "query": [],
        "document": [],
        "intent": [],
        "citing_id": [],
        "cited_id": [],
    }

    def _flush_shard() -> None:
        nonlocal shard_idx, shard_data
        if not shard_data["query"]:
            return
        table = pa.table(shard_data)
        shard_path = output_dir / f"intent_pairs_{shard_idx:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        logger.info(
            "Wrote shard %d (%d pairs so far)", shard_idx, pair_count
        )
        shard_idx += 1
        shard_data = {k: [] for k in shard_data}

    # Paper cache for cited paper lookups
    paper_cache: dict[int, dict[str, str]] = {}

    def _lookup_papers(ids: set[int]) -> None:
        ids_to_fetch = {i for i in ids if i is not None} - set(paper_cache.keys())
        if not ids_to_fetch:
            return
        id_list = list(ids_to_fetch)
        for chunk_start in range(0, len(id_list), 10_000):
            chunk = id_list[chunk_start : chunk_start + 10_000]
            ids_str = ",".join(str(i) for i in chunk)
            query = PAPER_LOOKUP_QUERY.format(max_chars=max_abstract_chars, ids=ids_str)
            rows = dl.query(query)
            for row in rows:
                paper_cache[row[0]] = {"title": row[1], "abstract": row[2]}

    try:
        query = INTENT_CONTEXTS_STREAM_QUERY.format(
            influential_filter=influential_filter
        )

        pbar = tqdm(desc="Intent pairs", unit="pairs", total=target)

        for batch in dl.stream_query(query, batch_size=500_000):
            if pair_count >= target:
                break

            citing_ids = batch.column("citingcorpusid").to_pylist()
            cited_ids = batch.column("citedcorpusid").to_pylist()
            contexts_col = batch.column("contexts").to_pylist()
            intents_col = batch.column("intents").to_pylist()

            # Batch lookup all cited papers in this batch
            _lookup_papers(set(cited_ids))

            for citing_id, cited_id, contexts_raw, intents_raw in zip(
                citing_ids, cited_ids, contexts_col, intents_col
            ):
                if pair_count >= target:
                    break

                if cited_id not in paper_cache:
                    continue

                cited_paper = paper_cache[cited_id]
                intents = _parse_intents(intents_raw)
                contexts = _parse_contexts(contexts_raw)

                if not contexts or not intents:
                    continue

                n_pairs = min(len(contexts), len(intents))

                for i in range(n_pairs):
                    if pair_count >= target:
                        break

                    intent = intents[i].lower()
                    if intent not in INTENT_PREFIX_MAP:
                        continue

                    prefix = INTENT_PREFIX_MAP[intent]
                    ctx_text = _filter_context(
                        contexts[i], min_context_chars, max_context_chars
                    )
                    if ctx_text is None:
                        continue

                    repeats = oversample.get(intent, 1)
                    for _ in range(repeats):
                        if pair_count >= target:
                            break

                        shard_data["query"].append(f"{prefix}: {ctx_text}")
                        shard_data["document"].append(
                            f"search_document: {cited_paper['title']}. {cited_paper['abstract']}"
                        )
                        shard_data["intent"].append(intent)
                        shard_data["citing_id"].append(citing_id)
                        shard_data["cited_id"].append(cited_id)

                        pair_count += 1
                        intent_counts[intent] += 1
                        pbar.update(1)

                        if len(shard_data["query"]) >= shard_size:
                            _flush_shard()

        _flush_shard()
        pbar.close()

    finally:
        dl.close()

    stats = {
        "total_pairs": pair_count,
        "num_shards": shard_idx,
        "output_dir": str(output_dir),
        "intent_distribution": dict(intent_counts),
        "paper_cache_size": len(paper_cache),
    }
    logger.info(
        "Generated %d intent pairs (%s) in %d shards",
        pair_count,
        ", ".join(f"{k}={v}" for k, v in sorted(intent_counts.items())),
        shard_idx,
    )
    return stats
