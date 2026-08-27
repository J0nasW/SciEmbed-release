"""Stage 1: Domain-adaptive MLM pretraining with MosaicML Composer.

Continues MLM pretraining of ModernBERT-base on the scientific corpus,
following the BioClinical ModernBERT recipe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class _AttentionMaskAwareMLMCollator:
    """MLM collator that only masks tokens where attention_mask=1.

    The attention mask marks padding positions as 0, so we only mask tokens
    where attention_mask=1. The default DataCollatorForLanguageModeling would
    mask padding positions, producing garbage gradients and inflating the loss.
    """

    def __init__(self, tokenizer, mlm_probability: float = 0.3):
        import torch

        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, samples: list[dict]) -> dict:
        import torch

        input_ids = torch.stack([torch.as_tensor(s["input_ids"], dtype=torch.long) for s in samples])
        attention_mask = torch.stack([torch.as_tensor(s["attention_mask"], dtype=torch.long) for s in samples])

        labels = input_ids.clone()

        # Probability matrix: only mask where attention_mask=1
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        probability_matrix[attention_mask == 0] = 0.0

        # Don't mask special tokens (CLS, SEP, PAD)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
            for val in input_ids
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Sample masked positions
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% replace with [MASK], 10% random, 10% keep
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _build_dataloader(cfg: DictConfig, split: str = "train") -> Any:
    """Build a Composer-compatible streaming dataloader from MDS shards.

    Args:
        cfg: Stage1MLMConfig.
        split: "train" or "eval" — eval uses a small subset for validation.

    Returns:
        DataLoader for Composer trainer.
    """
    from streaming import StreamingDataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # For Composer gradient accumulation:
    #   DataLoader batch_size = global_batch_size (full optimizer step)
    #   device_train_microbatch_size = microbatch_size (fits in GPU memory)
    #   Composer splits each DataLoader batch into grad_accum microbatches
    dl_batch_size = cfg.global_batch_size if split == "train" else cfg.microbatch_size

    # StreamingDataset is an IterableDataset that handles distributed sharding
    # internally — no DistributedSampler needed. It detects world_size/rank
    # from the environment variables set by the Composer launcher.
    dataset = StreamingDataset(
        local=cfg.corpus_dir,
        shuffle=(split == "train"),
        batch_size=dl_batch_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    collator = _AttentionMaskAwareMLMCollator(
        tokenizer=tokenizer,
        mlm_probability=cfg.mask_rate,
    )

    return DataLoader(
        dataset,
        batch_size=dl_batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )


def _build_model(cfg: DictConfig) -> Any:
    """Build a Composer HuggingFaceModel for masked language modeling.

    Args:
        cfg: Stage1MLMConfig.

    Returns:
        ComposerModel wrapping a ModernBERT for MLM.
    """
    from composer.metrics import LanguageCrossEntropy
    from composer.models import HuggingFaceModel
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    logger.info("Loading model: %s", cfg.model_name)
    model_config = AutoConfig.from_pretrained(cfg.model_name)
    model = AutoModelForMaskedLM.from_pretrained(cfg.model_name, config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Pass LanguageCrossEntropy as eval metric so EarlyStopper can monitor it.
    # ignore_index=-100 matches our MLM collator's label masking.
    eval_metrics = [LanguageCrossEntropy(ignore_index=-100)]

    composer_model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        eval_metrics=eval_metrics,
    )

    return composer_model


def train_mlm(cfg: DictConfig) -> None:
    """Run Stage 1 MLM pretraining with Composer.

    Args:
        cfg: Stage1MLMConfig (as DictConfig).
    """
    from composer import Trainer
    from composer.algorithms import GradientClipping
    from composer.callbacks import EarlyStopper, LRMonitor, MemoryMonitor, SpeedMonitor
    from composer.loggers import FileLogger
    from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    logger.info("Building model...")
    composer_model = _build_model(cfg)

    # Validate corpus path early
    corpus_path = Path(cfg.corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {cfg.corpus_dir}")

    # Dataloaders
    logger.info("Building train dataloader from %s", cfg.corpus_dir)
    train_dataloader = _build_dataloader(cfg, split="train")

    logger.info("Building eval dataloader (50K sample subset)...")
    eval_dataloader = _build_dataloader(cfg, split="eval")

    # Optimizer — AdamW with standard (LR-coupled) weight decay.
    # NOTE: Do NOT use Composer's DecoupledAdamW here. Its decay formula
    # is param *= (1 - decay_factor * wd) where decay_factor = lr/initial_lr,
    # which destroys ~46% of pretrained weights in 500 steps with wd=0.01.
    # Standard AdamW applies param *= (1 - lr * wd) = (1 - 2e-7), negligible.
    optimizer = torch.optim.AdamW(
        composer_model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # LR scheduler — warmup unit must match max_duration unit
    # Each optimizer step processes global_batch_size sequences
    max_duration = cfg.max_tokens
    if "tok" in max_duration:
        warmup_tokens = cfg.warmup_steps * cfg.global_batch_size * cfg.max_seq_length
        t_warmup = f"{warmup_tokens}tok"
    else:
        t_warmup = f"{cfg.warmup_steps}ba"

    logger.info(
        "Training config: lr=%.2e, warmup=%s, max_duration=%s, "
        "global_batch=%d, microbatch=%d, grad_accum=%d",
        cfg.lr, t_warmup, max_duration,
        cfg.global_batch_size, cfg.microbatch_size,
        cfg.global_batch_size // cfg.microbatch_size,
    )

    scheduler = CosineAnnealingWithWarmupScheduler(
        t_warmup=t_warmup,
        alpha_f=0.1,
    )

    # Callbacks
    from composer.callbacks import RuntimeEstimator

    eval_interval = getattr(cfg, "eval_interval", "500ba")

    # Early stopping: halt if eval loss doesn't improve for patience evals
    early_stopping_patience = getattr(cfg, "early_stopping_patience", 5)

    callbacks = [
        SpeedMonitor(window_size=100),
        LRMonitor(),
        MemoryMonitor(),
        RuntimeEstimator(),
        EarlyStopper(
            monitor="LanguageCrossEntropy",
            dataloader_label="eval",
            patience=f"{early_stopping_patience * int(eval_interval.replace('ba', ''))}ba",
            min_delta=0.001,
        ),
    ]

    # Loggers — TensorBoard for local processing
    from composer.loggers import TensorboardLogger

    tb_log_dir = str(output_dir / "tensorboard")
    loggers = [
        TensorboardLogger(log_dir=tb_log_dir),
        FileLogger(
            filename=str(output_dir / "logs" / "train-log-{rank}.txt"),
            flush_interval=100,
        ),
    ]
    logger.info("TensorBoard logs → %s", tb_log_dir)

    # Algorithms
    algorithms = [
        GradientClipping(clipping_type="norm", clipping_threshold=1.0),
    ]

    # Trainer
    logger.info("Initializing Composer Trainer...")
    eval_subset = getattr(cfg, "eval_subset_num_batches", 100)

    from composer.core import Evaluator

    evaluator = Evaluator(
        label="eval",
        dataloader=eval_dataloader,
        subset_num_batches=eval_subset,
    )

    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=evaluator,
        eval_interval=eval_interval,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_tokens,
        device_train_microbatch_size=cfg.microbatch_size,
        precision=cfg.precision,
        algorithms=algorithms,
        callbacks=callbacks,
        loggers=loggers,
        log_to_console=True,
        console_log_interval="100ba",
        save_folder=cfg.save_folder,
        save_interval=cfg.save_interval,
        autoresume=cfg.autoresume,
        run_name=getattr(cfg, "run_name", None) or "stage1-mlm",
        seed=42,
    )

    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training complete. Checkpoints at %s", cfg.save_folder)
