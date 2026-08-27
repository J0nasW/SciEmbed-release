#!/usr/bin/env python3
"""Build MLM corpus from full-text papers in the science datalake."""

import logging
import sys
from pathlib import Path

from sciembed.config import load_typed_config, MLMCorpusConfig
from sciembed.data.mlm_corpus import build_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/data/mlm_corpus.yaml"
    overrides = sys.argv[2:] if len(sys.argv) > 2 else None
    cfg = load_typed_config(config_path, MLMCorpusConfig, overrides)
    stats = build_corpus(cfg)
    print(f"Done. {stats['total_sequences']} sequences, {stats['total_tokens']} tokens.")


if __name__ == "__main__":
    main()
