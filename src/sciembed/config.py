"""OmegaConf-based config loading and validation for SciEmbed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import MISSING, DictConfig, OmegaConf

# Structured config schemas

DATALAKE_DEFAULT = "/path/to/data/science_datalake/datalake.duckdb"


@dataclass
class DatalakeConfig:
    db_path: str = DATALAKE_DEFAULT
    read_only: bool = True
    threads: int = 8
    memory_limit: str = "32GB"


@dataclass
class MLMCorpusConfig:
    datalake: DatalakeConfig = field(default_factory=DatalakeConfig)
    min_text_length: int = 1000
    max_text_length: int = 500_000
    tokenizer: str = "answerdotai/ModernBERT-base"
    max_seq_length: int = 8192
    stride: int = 128
    output_dir: str = MISSING
    num_workers: int = 8
    shard_size_mb: int = 256
    sources: list[str] = field(default_factory=lambda: ["pmc", "s2orc", "arxiv", "pes2o"])


@dataclass
class CitationTripletsConfig:
    datalake: DatalakeConfig = field(default_factory=DatalakeConfig)
    output_dir: str = MISSING
    num_triplets: int = 50_000_000
    influential_only: bool = True
    max_abstract_chars: int = 512
    negative_mix: dict[str, float] = field(
        default_factory=lambda: {"same_topic": 0.5, "two_hop": 0.3, "random": 0.2}
    )
    min_citation_count: int = 0  # minimum citations for both papers (quality filter for general tier)
    paper_pool_path: Optional[str] = None  # pre-materialized paper pool parquet
    num_workers: int = 8
    shard_size: int = 1_000_000


@dataclass
class CitationContextsConfig:
    datalake: DatalakeConfig = field(default_factory=DatalakeConfig)
    output_dir: str = MISSING
    num_pairs: int = 20_000_000
    influential_only: bool = True
    min_context_chars: int = 50
    max_context_chars: int = 1000
    max_abstract_chars: int = 512
    page_size: int = 50_000
    shard_size: int = 1_000_000


@dataclass
class IntentTripletsConfig:
    datalake: DatalakeConfig = field(default_factory=DatalakeConfig)
    output_dir: str = MISSING
    num_pairs: int = 15_000_000
    influential_only: bool = True
    min_context_chars: int = 50
    max_context_chars: int = 1000
    max_abstract_chars: int = 512
    page_size: int = 50_000
    shard_size: int = 1_000_000
    oversample: dict[str, int] = field(
        default_factory=lambda: {"background": 1, "methodology": 3, "result": 10}
    )


@dataclass
class SectionPairsConfig:
    datalake: DatalakeConfig = field(default_factory=DatalakeConfig)
    output_dir: str = MISSING
    num_pairs: int = 5_000_000
    min_text_length: int = 1000
    max_text_length: int = 500_000
    page_size: int = 5_000
    shard_size: int = 500_000
    positive_ratio: float = 0.6
    max_papers_per_source: int = 500_000
    sources: list[str] = field(default_factory=lambda: ["pmc", "s2orc", "arxiv"])


@dataclass
class DataMixerConfig:
    output_dir: str = MISSING
    shard_size: int = 1_000_000
    seed: int = 42
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstructionPairsConfig:
    datalake: DatalakeConfig = field(default_factory=DatalakeConfig)
    output_dir: str = MISSING
    search_pairs: int = 4_000_000
    classification_pairs: int = 3_000_000
    clustering_pairs: int = 2_000_000
    similarity_pairs: int = 1_000_000
    triplets_dir: Optional[str] = None  # reuse citation triplets for search pairs


@dataclass
class Stage1MLMConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    corpus_dir: str = MISSING  # MDS corpus from MLMCorpusConfig
    output_dir: str = MISSING
    # Training hyperparameters
    max_seq_length: int = 8192
    mask_rate: float = 0.30
    global_batch_size: int = 72
    microbatch_size: int = 6
    lr: float = 2e-5
    warmup_steps: int = 1000
    max_tokens: str = "10B"
    precision: str = "amp_bf16"
    # Checkpointing
    save_interval: str = "2000ba"
    autoresume: bool = True
    save_folder: str = MISSING
    # Evaluation
    eval_interval: str = "500ba"
    eval_subset_num_batches: int = 100  # eval on this many batches per eval pass
    early_stopping_patience: int = 5  # halt after N evals with no improvement
    # Logging
    run_name: Optional[str] = None


@dataclass
class Stage2ContrastiveConfig:
    model_name_or_path: str = MISSING  # Stage 1 checkpoint converted to HF
    output_dir: str = MISSING
    triplets_dir: str = MISSING
    instruction_pairs_dir: Optional[str] = None
    mixed_training_dir: Optional[str] = None  # Output of data_mixer (preferred)
    # Training
    epochs: int = 3
    batch_size: int = 128
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    matryoshka_dims: list[int] = field(default_factory=lambda: [768, 512, 256, 128])
    gradient_accumulation_steps: int = 128
    seed: int = 42
    # Contrastive loss temperature: scale = 1/τ. sentence-transformers' default
    # is 20.0 (τ=0.05). Lower scale (higher τ) softens the softmax over
    # in-batch negatives; higher scale sharpens it.
    loss_scale: float = 20.0
    # Loss variant. "mnrl" = MultipleNegativesRankingLoss (standard).
    # "cached_mnrl" = CachedMultipleNegativesRankingLoss with gradient caching;
    # combined with loss_gather_across_devices=True this provides a single huge
    # cross-GPU negative pool without per-GPU memory blowup.
    # "sym_mnrl" = MultipleNegativesSymmetricRankingLoss (loss in both directions).
    loss_type: str = "mnrl"
    # Mini-batch for CachedMNRL gradient caching. Controls forward-pass chunk
    # size; effective negative pool = per_device_batch × world_size.
    loss_mini_batch_size: int = 32
    # If True, pool in-batch negatives across all DDP ranks (huge negative set).
    loss_gather_across_devices: bool = False
    # Pooling & sequence length
    pooling: str = "mean"
    max_seq_length: int = 512  # abstracts/titles; 8192 causes OOM
    # Evaluation
    eval_fraction: float = 0.05  # 5% held out for validation
    eval_steps: int = 500
    early_stopping_patience: int = 3  # halt after N evals with no improvement
    # Checkpointing & logging
    logging_steps: int = 50
    save_steps: int = 2000
    run_name: Optional[str] = None
    # DataLoader parallelism.  Defaults preserve prior behaviour (main-process
    # dataloading); set workers > 0 on shared-filesystem clusters where GPUs
    # starve on I/O.
    dataloader_num_workers: int = 0
    dataloader_prefetch_factor: int = 4


# Config loading utilities

def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    d = Path(__file__).resolve().parent
    for _ in range(5):
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    # Fallback: assume src/sciembed/config.py layout
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _find_project_root()


def _resolve_env_vars(cfg: DictConfig) -> DictConfig:
    """Register custom OmegaConf resolvers for environment variables."""
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda key, default="": os.getenv(key, default))
    return cfg


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config file with optional CLI overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides: List of dotlist overrides, e.g. ["lr=1e-4", "batch_size=32"].

    Returns:
        Merged and resolved OmegaConf DictConfig.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    _resolve_env_vars(cfg)
    OmegaConf.resolve(cfg)
    return cfg


def load_typed_config(
    config_path: str | Path,
    schema: Any,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load a YAML config and validate against a structured schema.

    Args:
        config_path: Path to the YAML config file.
        schema: A dataclass type used as the schema.
        overrides: List of dotlist overrides.

    Returns:
        Merged, validated, and resolved OmegaConf DictConfig.
    """
    file_cfg = load_config(config_path, overrides)
    schema_cfg = OmegaConf.structured(schema)
    merged = OmegaConf.merge(schema_cfg, file_cfg)
    assert isinstance(merged, DictConfig)
    OmegaConf.resolve(merged)
    return merged
