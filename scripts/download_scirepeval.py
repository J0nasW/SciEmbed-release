#!/usr/bin/env python3
"""Pre-download ALL SciRepEval datasets to local cache.

Run this once before evaluation to avoid HuggingFace API calls.
After downloading, set HF_DATASETS_OFFLINE=1 to run fully offline.

Usage:
    python scripts/download_scirepeval.py
"""

import time
from datasets import load_dataset

# Every config referenced in scirepeval_tasks.jsonl — meta datasets
SCIREPEVAL_META_CONFIGS = [
    "biomimicry",
    "drsm",
    "relish",
    "nfcorpus",
    "trec_covid",
    "paper_reviewer_matching",
    "peer_review_score_hIndex",
    "tweet_mentions",
    "scidocs_mag_mesh",
    "scidocs_view_cite_read",
    "same_author",
    "high_influence_cite",
    "search",
    "cite_count",
    "pub_year",
    "fos",
    "mesh_descriptors",
]

# Every config referenced in scirepeval_tasks.jsonl — test datasets
SCIREPEVAL_TEST_CONFIGS = [
    "biomimicry",
    "drsm",
    "relish",
    "nfcorpus",
    "trec_covid",
    "paper_reviewer_matching",
    "reviewers",          # reviewer metadata for Paper-Reviewer Matching
    "peer_review_score",
    "hIndex",
    "tweet_mentions",
    "scidocs_mag",
    "scidocs_mesh",
    "scidocs_cite",
    "scidocs_view",
    "scidocs_cocite",
    "scidocs_read",
    "same_author",
    "high_influence_cite",
    "search",
    "cite_count",
    "pub_year",
    "fos",
    "mesh_descriptors",
]

# Also download model tokenizers/configs (no weights — those are separate)
MODELS_TO_CACHE = [
    "allenai/specter2_base",
    "intfloat/e5-large-v2",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/modernbert-embed-base",
]


def download_datasets():
    """Download all dataset configs with appropriate splits."""
    print("=== Downloading allenai/scirepeval (meta) configs ===")
    for config in SCIREPEVAL_META_CONFIGS:
        try:
            print(f"  scirepeval/{config} ...", end=" ", flush=True)
            load_dataset("allenai/scirepeval", config, split="evaluation")
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(2)  # be gentle with rate limits

    print("\n=== Downloading allenai/scirepeval_test configs ===")
    for config in SCIREPEVAL_TEST_CONFIGS:
        try:
            print(f"  scirepeval_test/{config} ...", end=" ", flush=True)
            # Test datasets use different split names
            try:
                load_dataset("allenai/scirepeval_test", config, split="test")
            except ValueError:
                # Some configs use "evaluation" split
                load_dataset("allenai/scirepeval_test", config, split="evaluation")
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(2)


def download_models():
    """Pre-download model weights and tokenizers."""
    from transformers import AutoTokenizer, AutoModel

    print("\n=== Downloading model weights & tokenizers ===")
    for model_name in MODELS_TO_CACHE:
        try:
            print(f"  {model_name} ...", end=" ", flush=True)
            AutoTokenizer.from_pretrained(model_name)
            AutoModel.from_pretrained(model_name)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(2)


if __name__ == "__main__":
    download_datasets()
    download_models()
    print("\n" + "=" * 60)
    print("All downloads complete!")
    print("Set HF_DATASETS_OFFLINE=1 and HF_HUB_OFFLINE=1 to run fully offline.")
    print("=" * 60)
