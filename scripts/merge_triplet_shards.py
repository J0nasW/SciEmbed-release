#!/usr/bin/env python3
"""Merge per-worker triplet shards into a single output directory.

After the parallel triplet build, each worker writes to its own
subdirectory (e.g., citation_triplets/worker_0/, worker_1/, ...).
This script moves all shards into the parent directory with unique names.

Usage:
    python scripts/merge_triplet_shards.py /path/to/citation_triplets 8
"""

import json
import shutil
import sys
from pathlib import Path


def merge_triplets(base_dir: str, num_workers: int) -> None:
    base = Path(base_dir)
    shard_idx = 0
    total_triplets = 0

    for worker_id in range(num_workers):
        worker_dir = base / f"worker_{worker_id}"
        if not worker_dir.exists():
            print(f"  Warning: {worker_dir} not found, skipping")
            continue

        shards = sorted(worker_dir.glob("triplets_*.parquet"))
        print(f"  worker_{worker_id}: {len(shards)} shards")

        for shard in shards:
            new_name = f"triplets_{shard_idx:05d}.parquet"
            dest = base / new_name
            shutil.move(str(shard), str(dest))
            shard_idx += 1

        # Clean up empty worker dir
        remaining = list(worker_dir.iterdir())
        if not any(f.suffix == ".parquet" for f in remaining):
            shutil.rmtree(worker_dir)

    print(f"\nMerged: {shard_idx} shards into {base}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <base_dir> <num_workers>")
        sys.exit(1)
    merge_triplets(sys.argv[1], int(sys.argv[2]))
