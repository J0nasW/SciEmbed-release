#!/usr/bin/env python3
"""Generate MDS index.json from existing shard files.

Useful when the corpus pipeline is still running but we want to start
training on the shards already written. Re-run to pick up new shards.
"""

import json
import struct
import sys
from pathlib import Path

import zstandard


def generate_index(corpus_dir: str) -> None:
    corpus_path = Path(corpus_dir)
    shard_files = sorted(corpus_path.glob("shard.*.mds.zstd"))

    if not shard_files:
        print(f"No shard files found in {corpus_dir}")
        sys.exit(1)

    dctx = zstandard.ZstdDecompressor()
    shards = []

    for shard_file in shard_files:
        raw_name = shard_file.name.replace(".zstd", "")
        zip_bytes = shard_file.stat().st_size

        # Read just the header: first 4 bytes (decompressed) = num_samples
        try:
            with open(shard_file, "rb") as f:
                # Stream-decompress just enough to read header
                reader = dctx.stream_reader(f)
                header = reader.read(4)
                if len(header) < 4:
                    print(f"  Warning: {shard_file.name} too short, skipping")
                    continue
                num_samples = struct.unpack("<I", header)[0]
                reader.close()
        except Exception as e:
            print(f"  Warning: Error reading {shard_file.name}: {e}, skipping")
            continue

        shards.append({
            "column_encodings": ["ndarray:uint8", "ndarray:uint16"],
            "column_names": ["attention_mask", "input_ids"],
            "column_sizes": [None, None],
            "compression": "zstd",
            "format": "mds",
            "hashes": [],
            "raw_data": {"basename": raw_name, "bytes": 0, "hashes": {}},
            "samples": num_samples,
            "size_limit": 256 * 1024 * 1024,  # 256 MB
            "version": 2,
            "zip_data": {"basename": shard_file.name, "bytes": zip_bytes, "hashes": {}},
        })

    index = {
        "version": 2,
        "shards": shards,
    }

    index_path = corpus_path / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    total_samples = sum(s["samples"] for s in shards)
    print(f"Generated index.json: {len(shards)} shards, {total_samples:,} samples")


if __name__ == "__main__":
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else "/path/to/data/sciembed_output/mlm_corpus"
    generate_index(corpus_dir)
