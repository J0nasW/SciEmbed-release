#!/usr/bin/env python3
"""Build citation triplets for contrastive training."""

import logging
import sys

from sciembed.config import load_typed_config, CitationTripletsConfig
from sciembed.data.citation_triplets import build_triplets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/data/citation_triplets.yaml"
    overrides = sys.argv[2:] if len(sys.argv) > 2 else None
    cfg = load_typed_config(config_path, CitationTripletsConfig, overrides)
    stats = build_triplets(cfg)
    print(f"Done. {stats['total_triplets']} triplets in {stats['num_shards']} shards.")


if __name__ == "__main__":
    main()
