"""Data mixer for combining all training signals with configurable ratios.

Supports the ablation study design:
    SciEmbed-BASE: Signal A only (citation edge triplets)
    SciEmbed-CTX:  A + B (+ citation context queries)
    SciEmbed-INT:  A + B + C (+ intent-conditioned)
    SciEmbed-HN:   A + B + C + D (+ cross-DB hard negatives)
    SciEmbed-SEC:  A + B + C + D + E (+ section-aware)
    SciEmbed-FULL: All signals + classification/clustering/similarity

Reads Parquet shards from each signal directory, samples according to
mix ratios, and writes a unified training dataset.
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _load_parquet_shards(
    directory: Path,
    pattern: str = "*.parquet",
    columns: list[str] | None = None,
) -> pa.Table:
    """Load all Parquet shards from a directory into a single table."""
    shard_paths = sorted(directory.glob(pattern))
    if not shard_paths:
        logger.warning("No Parquet files matching %s in %s", pattern, directory)
        return pa.table({})

    tables = []
    for path in shard_paths:
        t = pq.read_table(path, columns=columns)
        tables.append(t)

    combined = pa.concat_tables(tables, promote_options="permissive")
    # Cast string columns to large_string to avoid 2GB offset overflow
    # during downstream operations (take, concat, etc.)
    new_schema = []
    for field in combined.schema:
        if field.type == pa.string():
            new_schema.append(field.with_type(pa.large_string()))
        else:
            new_schema.append(field)
    return combined.cast(pa.schema(new_schema))


def _sample_rows(table: pa.Table, n: int) -> pa.Table:
    """Randomly sample n rows from a PyArrow table."""
    total = len(table)
    if total <= n:
        return table
    indices = random.sample(range(total), n)
    return table.take(indices)


def _normalize_columns(
    table: pa.Table,
    source_name: str,
) -> pa.Table:
    """Normalize a signal's table to unified schema: (anchor, positive, negative, signal).

    Different signals have different column names:
    - Citation triplets: anchor, positive, negative
    - Context pairs: query, document
    - Intent pairs: query, document, intent
    - Section pairs: anchor, positive, pair_type
    - Instruction pairs: various

    All get normalized to: anchor, positive, negative (nullable), signal_type.
    """
    col_names = set(table.column_names)

    n = len(table)
    signal_col = pa.array([source_name] * n, type=pa.large_string())

    if {"anchor", "positive", "negative"} <= col_names:
        # Citation triplets or section pairs with negatives
        return pa.table({
            "anchor": table.column("anchor"),
            "positive": table.column("positive"),
            "negative": table.column("negative"),
            "signal_type": signal_col,
        })

    if {"query", "document"} <= col_names:
        # Context pairs or intent pairs
        null_col = pa.array([None] * n, type=pa.large_string())
        return pa.table({
            "anchor": table.column("query"),
            "positive": table.column("document"),
            "negative": null_col,
            "signal_type": signal_col,
        })

    if {"anchor", "positive"} <= col_names:
        # Section pairs (no explicit negative)
        null_col = pa.array([None] * n, type=pa.large_string())
        return pa.table({
            "anchor": table.column("anchor"),
            "positive": table.column("positive"),
            "negative": null_col,
            "signal_type": signal_col,
        })

    if {"text1", "text2"} <= col_names:
        # Instruction pairs (search, classification, clustering, similarity)
        null_col = pa.array([None] * n, type=pa.large_string())
        return pa.table({
            "anchor": table.column("text1"),
            "positive": table.column("text2"),
            "negative": null_col,
            "signal_type": signal_col,
        })

    if {"text", "label"} <= col_names:
        # Classification pairs
        null_col = pa.array([None] * n, type=pa.large_string())
        return pa.table({
            "anchor": table.column("text"),
            "positive": table.column("label"),
            "negative": null_col,
            "signal_type": signal_col,
        })

    logger.warning(
        "Unknown column schema for signal '%s': %s. Skipping.",
        source_name, col_names,
    )
    return pa.table({
        "anchor": pa.array([], type=pa.large_string()),
        "positive": pa.array([], type=pa.large_string()),
        "negative": pa.array([], type=pa.large_string()),
        "signal_type": pa.array([], type=pa.large_string()),
    })


def mix_signals(cfg: DictConfig) -> dict[str, Any]:
    """Mix all training signals according to configured ratios.

    Reads from each signal directory, samples the configured number of
    pairs, normalizes to a common schema, shuffles, and writes output.

    Args:
        cfg: DataMixerConfig (as DictConfig).

    Returns:
        Statistics dict with per-signal counts.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = getattr(cfg, "seed", 42)
    random.seed(seed)
    logger.info("Data mixer seed: %d", seed)

    signal_configs = dict(cfg.signals)
    shard_size = cfg.shard_size

    all_tables = []
    signal_stats = {}

    for signal_name, signal_cfg in signal_configs.items():
        signal_dir = Path(signal_cfg["dir"])
        target_pairs = signal_cfg["pairs"]
        pattern = signal_cfg.get("pattern", "*.parquet")

        if not signal_dir.exists():
            logger.warning("Signal dir does not exist: %s (skipping %s)", signal_dir, signal_name)
            continue

        logger.info("Loading signal '%s' from %s (target: %d pairs)", signal_name, signal_dir, target_pairs)

        # Determine which columns to load (keep it lightweight)
        raw_table = _load_parquet_shards(signal_dir, pattern=pattern)
        if len(raw_table) == 0:
            logger.warning("No data for signal '%s'", signal_name)
            continue

        # Filter out known-bad subtasks from instruction_pairs
        if signal_name == "instruction_pairs":
            col_names = set(raw_table.column_names)
            text_col = "text1" if "text1" in col_names else "anchor" if "anchor" in col_names else None
            if text_col:
                texts = raw_table.column(text_col).to_pylist()
                # Drop cluster pairs (random unrelated papers) and self-pairs
                keep_mask = []
                for i, t in enumerate(texts):
                    if t is None:
                        keep_mask.append(False)
                    elif t.startswith("cluster:"):
                        keep_mask.append(False)
                    else:
                        keep_mask.append(True)
                import pyarrow.compute as pc
                raw_table = raw_table.filter(pc.is_in(
                    pa.array(range(len(raw_table))),
                    pa.array([i for i, k in enumerate(keep_mask) if k]),
                ))
                # Also filter self-pairs
                if text_col and ("text2" in col_names or "positive" in col_names):
                    pair_col = "text2" if "text2" in col_names else "positive"
                    a_col = raw_table.column(text_col).to_pylist()
                    p_col = raw_table.column(pair_col).to_pylist()
                    non_self = [i for i, (a, p) in enumerate(zip(a_col, p_col)) if a != p]
                    raw_table = raw_table.take(non_self)
                pre_filter = len(texts)
                logger.info(
                    "  Filtered instruction_pairs: %d → %d (dropped cluster + self-pairs)",
                    pre_filter, len(raw_table),
                )

        # Sample to target size
        sampled = _sample_rows(raw_table, target_pairs)

        # Normalize columns
        normalized = _normalize_columns(sampled, signal_name)

        all_tables.append(normalized)
        signal_stats[signal_name] = {
            "available": len(raw_table),
            "sampled": len(sampled),
            "after_normalize": len(normalized),
        }
        logger.info(
            "  %s: %d available → %d sampled",
            signal_name, len(raw_table), len(normalized),
        )

    if not all_tables:
        logger.error("No signals loaded — cannot create mixed dataset")
        return {"total_pairs": 0, "signal_stats": signal_stats}

    # Concatenate all signals
    logger.info("Concatenating %d signal tables...", len(all_tables))
    combined = pa.concat_tables(all_tables)
    pre_dedup = len(combined)
    logger.info("Total combined pairs (pre-dedup): %d", pre_dedup)

    # Deduplicate on (anchor, positive) text hash
    logger.info("Deduplicating on (anchor, positive) pairs...")
    anchors = combined.column("anchor").to_pylist()
    positives = combined.column("positive").to_pylist()
    seen: set[bytes] = set()
    keep_indices = []
    for i, (a, p) in enumerate(zip(anchors, positives)):
        h = hashlib.md5(f"{a}\0{p}".encode(), usedforsecurity=False).digest()
        if h not in seen:
            seen.add(h)
            keep_indices.append(i)
    del seen, anchors, positives

    combined = combined.take(keep_indices)
    total_pairs = len(combined)
    duplicates_removed = pre_dedup - total_pairs
    logger.info(
        "Dedup: %d duplicates removed, %d unique pairs remain",
        duplicates_removed, total_pairs,
    )

    # Shuffle
    logger.info("Shuffling combined dataset...")
    indices = list(range(total_pairs))
    random.shuffle(indices)
    combined = combined.take(indices)

    # Write output shards
    logger.info("Writing output shards to %s...", output_dir)
    shard_idx = 0
    for start in tqdm(range(0, total_pairs, shard_size), desc="Writing shards"):
        end = min(start + shard_size, total_pairs)
        shard = combined.slice(start, end - start)
        shard_path = output_dir / f"mixed_{shard_idx:05d}.parquet"
        pq.write_table(shard, shard_path, compression="zstd")
        shard_idx += 1

    stats = {
        "total_pairs": total_pairs,
        "duplicates_removed": duplicates_removed,
        "num_shards": shard_idx,
        "output_dir": str(output_dir),
        "signal_stats": signal_stats,
    }
    logger.info("Mixed dataset: %d pairs in %d shards", total_pairs, shard_idx)
    return stats
