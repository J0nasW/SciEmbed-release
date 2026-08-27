"""Stage 2: Contrastive fine-tuning with sentence-transformers.

Uses MatryoshkaLoss wrapping MultipleNegativesRankingLoss for flexible
dimensionality embeddings with instruction-aware prefixes.

Data loading uses DatasetDict for task-homogeneous batching:
- Each signal type is loaded as a separate dataset
- Batches contain samples from only ONE signal (avoids cross-task false negatives)
- Triplet signals keep (anchor, positive, negative); pair signals get (anchor, positive) only
- No empty-string negative hack needed
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _filter_workers() -> int:
    """Number of CPU workers for datasets.filter() ops.

    Defaults to half of SLURM_CPUS_PER_TASK (or os.cpu_count) so the
    filter is parallelized but does not starve the trainer's data loaders.
    """
    n = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 8)
    return max(1, n // 2)


# Dataset loading — returns DatasetDict with per-signal datasets


def _load_mixed_as_dataset_dict(mixed_dir: str, eval_fraction: float = 0.05, seed: int = 42) -> tuple[Any, Any]:
    """Load pre-mixed data and split by signal_type into a DatasetDict.

    Signals with real negatives keep (anchor, positive, negative).
    Signals without negatives get only (anchor, positive) — no empty string hack.

    Returns:
        (train_dataset_dict, eval_dataset) tuple.
    """
    from datasets import DatasetDict, load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=f"{mixed_dir}/*.parquet",
        split="train",
    )

    logger.info("Loaded %d total samples from %s", len(dataset), mixed_dir)
    logger.info("Columns: %s", dataset.column_names)

    signal_types = set(dataset.unique("signal_type"))
    logger.info("Signal types: %s", signal_types)

    # Hold out eval set BEFORE splitting by signal (ensures eval covers all signals)
    split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
    train_all = split["train"]
    eval_all = split["test"]

    # For eval, keep a simple flat dataset with (anchor, positive) for loss computation
    eval_cols = {"anchor", "positive"} & set(eval_all.column_names)
    eval_drop = [c for c in eval_all.column_names if c not in eval_cols]
    eval_dataset = eval_all.remove_columns(eval_drop) if eval_drop else eval_all

    # Split train into per-signal datasets
    train_datasets = {}
    for signal in sorted(signal_types):
        subset = train_all.filter(
            lambda x: x["signal_type"] == signal,
            desc=f"Filtering {signal}",
            num_proc=_filter_workers(),
        )
        if len(subset) == 0:
            continue

        # Check if this signal has real negatives
        has_negatives = False
        if "negative" in subset.column_names:
            # Sample to check — if >50% non-null, treat as triplet signal
            sample = subset.select(range(min(1000, len(subset))))
            non_null = sum(1 for x in sample["negative"] if x is not None and x != "")
            has_negatives = non_null > len(sample) * 0.5

        if has_negatives:
            # Triplet signal: keep (anchor, positive, negative)
            keep = {"anchor", "positive", "negative"}
            drop = [c for c in subset.column_names if c not in keep]
            if drop:
                subset = subset.remove_columns(drop)
            # Filter out any rows with None negatives
            subset = subset.filter(
                lambda x: x["negative"] is not None and x["negative"] != "",
                desc=f"Filtering null negatives from {signal}",
                num_proc=_filter_workers(),
            )
            logger.info("  %s: %d triplets (anchor, positive, negative)", signal, len(subset))
        else:
            # Pair signal: keep (anchor, positive) ONLY — no negative column
            keep = {"anchor", "positive"}
            drop = [c for c in subset.column_names if c not in keep]
            if drop:
                subset = subset.remove_columns(drop)
            logger.info("  %s: %d pairs (anchor, positive)", signal, len(subset))

        train_datasets[signal] = subset

    train_dict = DatasetDict(train_datasets)
    total = sum(len(ds) for ds in train_dict.values())
    logger.info("Train DatasetDict: %d signals, %d total samples", len(train_dict), total)
    logger.info("Eval dataset: %d samples", len(eval_dataset))

    return train_dict, eval_dataset


def _load_single_signal(mixed_dir: str, eval_fraction: float = 0.05, seed: int = 42) -> tuple[Any, Any]:
    """Load a single-signal mixed dataset (e.g., BASE ablation with only triplets).

    When there's only one signal type, use a simple Dataset instead of DatasetDict.

    Returns:
        (train_dataset, eval_dataset) tuple.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=f"{mixed_dir}/*.parquet",
        split="train",
    )

    # Check signal types
    if "signal_type" in dataset.column_names:
        signal_types = set(dataset.unique("signal_type"))
        if len(signal_types) > 1:
            # Multiple signals — use DatasetDict path
            return _load_mixed_as_dataset_dict(mixed_dir, eval_fraction, seed=seed)

    # Single signal — check for negatives
    has_negatives = False
    if "negative" in dataset.column_names:
        sample = dataset.select(range(min(1000, len(dataset))))
        non_null = sum(1 for x in sample["negative"] if x is not None and x != "")
        has_negatives = non_null > len(sample) * 0.5

    if has_negatives:
        keep = {"anchor", "positive", "negative"}
    else:
        keep = {"anchor", "positive"}

    drop = [c for c in dataset.column_names if c not in keep]
    if drop:
        dataset = dataset.remove_columns(drop)

    if has_negatives:
        dataset = dataset.filter(
            lambda x: x["negative"] is not None and x["negative"] != "",
            desc="Filtering null negatives",
            num_proc=_filter_workers(),
        )

    split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
    logger.info(
        "Single-signal dataset: %d train, %d eval, columns=%s",
        len(split["train"]), len(split["test"]), split["train"].column_names,
    )
    return split["train"], split["test"]


# Training


def train_contrastive(cfg: DictConfig) -> None:
    """Run Stage 2 contrastive fine-tuning.

    Uses DatasetDict for task-homogeneous batching when multiple signals
    are present. Each batch contains samples from only one signal type,
    avoiding cross-task false negatives in MNRL's in-batch negative mining.

    Args:
        cfg: Stage2ContrastiveConfig (as DictConfig).
    """
    from datasets import DatasetDict
    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers.losses import (
        CachedMultipleNegativesRankingLoss,
        MatryoshkaLoss,
        MultipleNegativesRankingLoss,
        MultipleNegativesSymmetricRankingLoss,
    )
    from sentence_transformers.training_args import (
        SentenceTransformerTrainingArguments,
    )

    from sciembed.train.model import load_for_sentence_transformers

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = getattr(cfg, "seed", 42)

    # Load model with proper pooling configuration
    logger.info("Loading model: %s", cfg.model_name_or_path)
    max_seq_length = getattr(cfg, "max_seq_length", 512)
    model = load_for_sentence_transformers(
        model_path=cfg.model_name_or_path,
        pooling=cfg.pooling,
        max_seq_length=max_seq_length,
    )
    logger.info(
        "Model loaded: embedding_dim=%d, pooling=%s",
        model.get_sentence_embedding_dimension(),
        cfg.pooling,
    )

    mixed_dir = getattr(cfg, "mixed_training_dir", None)
    eval_fraction = getattr(cfg, "eval_fraction", 0.05)

    if mixed_dir:
        logger.info("Loading training data from %s", mixed_dir)
        train_dataset, eval_dataset = _load_single_signal(mixed_dir, eval_fraction, seed=seed)
    else:
        # Legacy fallback: load from triplets_dir
        logger.info("Loading triplets from %s", cfg.triplets_dir)
        from datasets import load_dataset
        dataset = load_dataset(
            "parquet",
            data_files=f"{cfg.triplets_dir}/triplets_*.parquet",
            split="train",
        )
        keep = {"anchor", "positive", "negative"} & set(dataset.column_names)
        drop = [c for c in dataset.column_names if c not in keep]
        if drop:
            dataset = dataset.remove_columns(drop)
        split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
        train_dataset, eval_dataset = split["train"], split["test"]

    # Loss setup — if DatasetDict, each signal gets its own loss
    loss_scale = float(cfg.get("loss_scale", 20.0))
    loss_type = str(cfg.get("loss_type", "mnrl")).lower()
    loss_mini_batch_size = int(cfg.get("loss_mini_batch_size", 32))
    loss_gather = bool(cfg.get("loss_gather_across_devices", False))

    def _make_base_loss():
        if loss_type == "cached_mnrl":
            return CachedMultipleNegativesRankingLoss(
                model,
                scale=loss_scale,
                mini_batch_size=loss_mini_batch_size,
                gather_across_devices=loss_gather,
            )
        if loss_type == "sym_mnrl":
            return MultipleNegativesSymmetricRankingLoss(model, scale=loss_scale)
        return MultipleNegativesRankingLoss(model, scale=loss_scale)

    logger.info(
        "Loss type=%s, scale=%.3f (τ=%.4f), mini_batch=%d, gather_devices=%s",
        loss_type, loss_scale, 1.0 / loss_scale, loss_mini_batch_size, loss_gather,
    )
    base_loss = _make_base_loss()
    matryoshka_loss = MatryoshkaLoss(
        model,
        loss=base_loss,
        matryoshka_dims=list(cfg.matryoshka_dims),
    )

    if isinstance(train_dataset, DatasetDict):
        # Per-signal loss — same loss function, but task-homogeneous batching
        # Each signal gets its own MatryoshkaLoss+base_loss so batches are signal-homogeneous
        loss = {name: MatryoshkaLoss(
            model,
            loss=_make_base_loss(),
            matryoshka_dims=list(cfg.matryoshka_dims),
        ) for name in train_dataset}
        # Eval also needs DatasetDict when loss is a dict — wrap with "eval_data" key
        eval_dataset = DatasetDict({"eval_data": eval_dataset})
        loss["eval_data"] = matryoshka_loss
        logger.info("Using DatasetDict with %d signal-specific losses", len(loss) - 1)
    else:
        loss = matryoshka_loss

    # Training args
    eval_steps = getattr(cfg, "eval_steps", 500)
    logging_steps = getattr(cfg, "logging_steps", 50)
    save_steps = getattr(cfg, "save_steps", 2000)

    # When using DatasetDict, the eval metric name includes the dataset key
    is_multi_signal = isinstance(train_dataset, DatasetDict)
    best_metric = "eval_eval_data_loss" if is_multi_signal else "eval_loss"

    # DataLoader parallelism.  Default to 0 for back-compat, but allow
    # overriding from config for shared-filesystem deployments where GPUs
    # starve without a prefetch pipeline.
    dl_num_workers = int(cfg.get("dataloader_num_workers", 0) or 0)
    dl_prefetch_factor = int(cfg.get("dataloader_prefetch_factor", 4) or 4)

    training_args_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        seed=seed,
        data_seed=seed,
        bf16=True,
        # Logging
        logging_steps=logging_steps,
        logging_first_step=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        # Checkpointing
        save_steps=save_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=False,
        # Logging — TensorBoard
        report_to="tensorboard",
        logging_dir=str(output_dir / "tensorboard"),
        run_name=cfg.get("run_name", "stage2-contrastive"),
        include_tokens_per_second=True,
        # DataLoader parallelism (0 = main-process dataloading; use a
        # positive value on shared-filesystem clusters to overlap I/O)
        dataloader_num_workers=dl_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=dl_num_workers > 0,
    )
    # prefetch_factor is only valid when num_workers > 0
    if dl_num_workers > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = dl_prefetch_factor

    training_args = SentenceTransformerTrainingArguments(**training_args_kwargs)

    # Early stopping
    from transformers import EarlyStoppingCallback
    early_stopping_patience = getattr(cfg, "early_stopping_patience", 3)

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Auto-resume from checkpoint
    resume_from = getattr(cfg, "resume_from_checkpoint", None)
    if resume_from is None:
        ckpt_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if ckpt_dirs:
            resume_from = str(ckpt_dirs[-1])
            logger.info("Auto-resuming from %s", resume_from)

    logger.info("Starting contrastive training...")
    trainer.train(resume_from_checkpoint=resume_from)
    model.save_pretrained(str(output_dir / "final"))
    logger.info("Training complete. Model saved to %s", output_dir / "final")
