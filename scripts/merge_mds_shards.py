#!/usr/bin/env python3
"""Merge per-source MDS shard directories into a single corpus directory.

After the 4-node parallel corpus build, each source writes to its own
subdirectory (e.g., mlm_corpus/pmc/, mlm_corpus/s2orc/, ...). This script
creates a unified directory with renamed shards and a combined index.json
so that StreamingDataset can read the entire corpus from one path.

Usage:
    python scripts/merge_mds_shards.py /path/to/mlm_corpus

Expects subdirectories: pmc/, s2orc/, arxiv/, pes2o/ each containing
shard.*.mds.zstd files and optionally index.json.
"""

import json
import shutil
import struct
import sys
from pathlib import Path

import zstandard


SOURCES = ["pmc", "s2orc", "arxiv", "pes2o"]


def read_shard_samples(shard_path: Path) -> int:
    """Read num_samples from the MDS shard header."""
    dctx = zstandard.ZstdDecompressor()
    with open(shard_path, "rb") as f:
        reader = dctx.stream_reader(f)
        header = reader.read(4)
        reader.close()
    if len(header) < 4:
        raise ValueError(f"Shard {shard_path} too short to read header")
    return struct.unpack("<I", header)[0]


def merge_shards(corpus_dir: str) -> None:
    corpus_path = Path(corpus_dir)
    merged_dir = corpus_path / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    shard_idx = 0
    total_samples = 0

    for source in SOURCES:
        source_dir = corpus_path / source
        if not source_dir.exists():
            print(f"  Warning: {source_dir} not found, skipping")
            continue

        shard_files = sorted(source_dir.glob("shard.*.mds.zstd"))
        if not shard_files:
            print(f"  Warning: No shards in {source_dir}, skipping")
            continue

        print(f"  {source}: {len(shard_files)} shards")

        for shard_file in shard_files:
            new_name = f"shard.{shard_idx:05d}.mds.zstd"
            dest = merged_dir / new_name

            # Copy shard file (hard link if same filesystem, else copy)
            if dest.exists():
                dest.unlink()
            try:
                dest.hardlink_to(shard_file)
            except OSError:
                shutil.copy2(shard_file, dest)

            num_samples = read_shard_samples(dest)
            total_samples += num_samples

            shards.append({
                "column_encodings": ["ndarray:uint8", "ndarray:uint16"],
                "column_names": ["attention_mask", "input_ids"],
                "column_sizes": [None, None],
                "compression": "zstd",
                "format": "mds",
                "hashes": [],
                "raw_data": {"basename": f"shard.{shard_idx:05d}.mds", "bytes": 0, "hashes": {}},
                "samples": num_samples,
                "size_limit": 256 * 1024 * 1024,  # 256 MB
                "version": 2,
                "zip_data": {"basename": new_name, "bytes": dest.stat().st_size, "hashes": {}},
            })
            shard_idx += 1

    # Write combined index
    index = {"version": 2, "shards": shards}
    index_path = merged_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nMerged: {shard_idx} shards, {total_samples:,} samples → {merged_dir}")

    # Also collect per-source stats
    stats = {"total_samples": total_samples, "total_shards": shard_idx, "sources": {}}
    for source in SOURCES:
        stats_file = corpus_path / source / "corpus_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats["sources"][source] = json.load(f)
    stats_path = merged_dir / "corpus_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <corpus_dir>")
        sys.exit(1)
    merge_shards(sys.argv[1])
