"""Full-text extraction, cleaning, tokenization → MDS shards for MLM pretraining."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from sciembed.data.datalake import DatalakeConnection
from sciembed.data.text_cleaning import clean_text

logger = logging.getLogger(__name__)

# Streaming query — DuckDB streams results via Arrow batches
FULLTEXT_STREAM_QUERY = """
SELECT doi, source, text
FROM fulltext.papers
WHERE has_full_text = true
  AND text_length BETWEEN {min_len} AND {max_len}
  AND source = '{source}'
"""

COUNT_QUERY = """
SELECT COUNT(*) FROM fulltext.papers
WHERE has_full_text = true
  AND text_length BETWEEN {min_len} AND {max_len}
  AND source = '{source}'
"""


@dataclass
class ChunkedSequence:
    """A tokenized chunk of a full-text paper."""

    input_ids: np.ndarray
    attention_mask: np.ndarray


def _init_worker(tokenizer_name: str) -> None:
    """Initialize tokenizer once per worker process (ProcessPoolExecutor initializer).

    Avoids reloading the tokenizer for every batch (~134K times → 48 times).
    """
    import warnings

    from transformers import AutoTokenizer

    # Suppress "Token indices sequence length is longer than..." warnings.
    # We intentionally tokenize full documents without truncation, then chunk
    # into max_seq_length windows ourselves in chunk_tokens().
    warnings.filterwarnings("ignore", message="Token indices sequence length")

    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


def chunk_tokens(
    input_ids: list[int],
    max_seq_length: int,
    stride: int,
    cls_token_id: int,
    sep_token_id: int,
    pad_token_id: int,
) -> list[ChunkedSequence]:
    """Split a long token sequence into overlapping chunks.

    Args:
        input_ids: Full token IDs for a document.
        max_seq_length: Maximum tokens per chunk (including special tokens).
        stride: Overlap between consecutive chunks.
        cls_token_id: Tokenizer's CLS token ID.
        sep_token_id: Tokenizer's SEP token ID.
        pad_token_id: Tokenizer's PAD token ID.

    Returns:
        List of ChunkedSequence objects.
    """
    # Reserve space for [CLS] and [SEP]
    content_length = max_seq_length - 2
    chunks = []

    for start in range(0, len(input_ids), content_length - stride):
        chunk_ids = input_ids[start : start + content_length]
        if len(chunk_ids) < 64:  # skip very short trailing chunks
            break

        # Pad if needed
        pad_length = content_length - len(chunk_ids)
        attention = [1] * (len(chunk_ids) + 2) + [0] * pad_length
        padded_ids = [cls_token_id] + chunk_ids + [sep_token_id] + [pad_token_id] * pad_length

        chunks.append(
            ChunkedSequence(
                input_ids=np.array(padded_ids, dtype=np.uint16),
                attention_mask=np.array(attention, dtype=np.uint8),
            )
        )

    return chunks


def process_paper(
    text: str,
    source: str,
    tokenizer: Any,
    max_seq_length: int,
    stride: int,
) -> list[ChunkedSequence]:
    """Clean and tokenize a single paper into chunks."""
    cleaned = clean_text(text, source=source)
    if not cleaned or len(cleaned) < 100:
        return []

    input_ids = tokenizer.encode(cleaned, add_special_tokens=False)

    return chunk_tokens(
        input_ids,
        max_seq_length,
        stride,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def _process_batch(
    batch: list[tuple[str, str]],
    max_seq_length: int,
    stride: int,
) -> list[ChunkedSequence]:
    """Process a batch of papers in a worker process.

    Uses the global _WORKER_TOKENIZER set by _init_worker() (ProcessPoolExecutor
    initializer), so the tokenizer is loaded once per worker, not per batch.

    Args:
        batch: List of (text, source) tuples.
        max_seq_length: Max tokens per chunk.
        stride: Overlap between chunks.

    Returns:
        List of ChunkedSequence from all papers in the batch.
    """
    results = []
    for text, source in batch:
        chunks = process_paper(text, source, _WORKER_TOKENIZER, max_seq_length, stride)
        results.extend(chunks)
    return results


def build_corpus(cfg: DictConfig, single_source: str | None = None) -> dict[str, Any]:
    """Build the MLM corpus from full-text papers in the datalake.

    Streams results via DuckDB Arrow batches — no LIMIT/OFFSET overhead.
    Each source is processed sequentially; within each source, worker
    processes handle cleaning + tokenization in parallel.

    Args:
        cfg: MLMCorpusConfig (as DictConfig).
        single_source: If set, process only this source (for multi-node parallelism).
            Output goes to output_dir/{source}/ subdirectory.

    Returns:
        Corpus statistics dict.
    """
    from sciembed.data.streaming import MDSWriter, write_stats

    if single_source:
        output_dir = Path(cfg.output_dir) / single_source
        sources = [single_source]
    else:
        output_dir = Path(cfg.output_dir)
        sources = list(cfg.sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_batch_size = 500  # papers per worker process
    stream_batch_size = 10_000  # rows per Arrow batch from DuckDB (full-text papers hit Arrow 2GB string buffer limit at larger sizes)

    logger.info("Connecting to datalake...")
    dl = DatalakeConnection(
        db_path=cfg.datalake.db_path,
        read_only=cfg.datalake.read_only,
        threads=cfg.datalake.threads,
        memory_limit=cfg.datalake.memory_limit,
    )

    # Count papers per source
    source_counts = {}
    total_papers = 0
    for source in sources:
        count = dl.query(COUNT_QUERY.format(
            min_len=cfg.min_text_length,
            max_len=cfg.max_text_length,
            source=source,
        ))[0][0]
        source_counts[source] = count
        total_papers += count
        logger.info("  %s: %d papers", source, count)
    logger.info("Total papers to process: %d", total_papers)

    stats = {
        "total_papers": total_papers,
        "total_sequences": 0,
        "total_tokens": 0,
        "sources": {},
    }

    writer = MDSWriter(output_dir, shard_size_mb=cfg.shard_size_mb)
    executor = ProcessPoolExecutor(
        max_workers=cfg.num_workers,
        initializer=_init_worker,
        initargs=(cfg.tokenizer,),
    )

    try:
        for source in sources:
            source_total = source_counts[source]
            source_seqs = 0
            source_tokens = 0
            papers_seen = 0
            logger.info("Processing source: %s (%d papers)", source, source_total)

            pbar = tqdm(total=source_total, desc=f"  {source}", unit="papers")

            query = FULLTEXT_STREAM_QUERY.format(
                min_len=cfg.min_text_length,
                max_len=cfg.max_text_length,
                source=source,
            )

            for batch in dl.stream_query(query, batch_size=stream_batch_size):
                texts = batch.column("text").to_pylist()
                sources_col = batch.column("source").to_pylist()
                batch_size = len(texts)

                # Build worker batches: (text, source) tuples
                items = [(t, s) for t, s in zip(texts, sources_col) if t]
                worker_batches = [
                    items[i : i + worker_batch_size]
                    for i in range(0, len(items), worker_batch_size)
                ]

                # Submit to process pool
                futures = [
                    executor.submit(
                        _process_batch,
                        wb,
                        cfg.max_seq_length,
                        cfg.stride,
                    )
                    for wb in worker_batches
                ]

                for future in as_completed(futures):
                    chunks = future.result()
                    for chunk in chunks:
                        writer.write(chunk.input_ids, chunk.attention_mask)
                        source_seqs += 1
                        source_tokens += int(chunk.attention_mask.sum())

                papers_seen += batch_size
                pbar.update(batch_size)

            pbar.close()
            stats["sources"][source] = {
                "papers": papers_seen,
                "sequences": source_seqs,
                "tokens": source_tokens,
            }
            stats["total_sequences"] += source_seqs
            stats["total_tokens"] += source_tokens
            logger.info(
                "  %s done: %d sequences, %d tokens",
                source, source_seqs, source_tokens,
            )

    finally:
        executor.shutdown(wait=True)
        writer_stats = writer.finish()
        dl.close()

    stats["mds_output"] = writer_stats
    write_stats(output_dir, stats)
    logger.info(
        "Corpus built: %d sequences, %d tokens",
        stats["total_sequences"],
        stats["total_tokens"],
    )
    return stats
