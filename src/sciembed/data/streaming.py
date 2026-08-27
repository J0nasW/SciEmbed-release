"""MDS (Mosaic Data Shard) writer and iterable dataset utilities for Composer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def get_mds_columns() -> dict[str, str]:
    """Return the MDS column schema for tokenized sequences."""
    return {
        "input_ids": "ndarray:uint16",
        "attention_mask": "ndarray:uint8",
    }


class MDSWriter:
    """Write tokenized sequences as MDS shards for MosaicML Streaming.

    This wraps streaming.MDSWriter with our column schema and shard config.
    Requires `mosaicml-streaming` (optional `train` dependency).
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size_mb: int = 256,
        compression: str | None = "zstd",
    ) -> None:
        from streaming.base.format.mds.writer import MDSWriter as _MDSWriter

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._writer = _MDSWriter(
            out=str(self.output_dir),
            columns=get_mds_columns(),
            compression=compression,
            size_limit=shard_size_mb * 1024 * 1024,
        )
        self._count = 0

    def write(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
        """Write a single tokenized sequence."""
        self._writer.write(
            {
                "input_ids": input_ids.astype(np.uint16),
                "attention_mask": attention_mask.astype(np.uint8),
            }
        )
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def finish(self) -> dict[str, Any]:
        """Finalize writing and return stats."""
        self._writer.finish()
        return {
            "output_dir": str(self.output_dir),
            "num_sequences": self._count,
        }

    def __enter__(self) -> MDSWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.finish()


def write_stats(output_dir: str | Path, stats: dict[str, Any]) -> None:
    """Write corpus statistics to a JSON file alongside MDS shards."""
    output_dir = Path(output_dir)
    stats_path = output_dir / "corpus_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
