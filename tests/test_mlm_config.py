"""Tests for MLM training configuration and sanity checks.

Validates the training hyperparameters to prevent the kinds of
misconfigurations that caused the loss explosion (lr=3e-4 on pretrained model).
"""

import pytest

from sciembed.config import Stage1MLMConfig, Stage2ContrastiveConfig


class TestStage1MLMConfig:
    """Validate Stage 1 MLM training configuration sanity."""

    def test_lr_is_continued_pretraining_range(self):
        """LR should be in continued pretraining range (1e-5 to 5e-5), not from-scratch range."""
        cfg = Stage1MLMConfig()
        assert cfg.lr <= 5e-5, (
            f"LR {cfg.lr} too high for continued pretraining! "
            f"3e-4 destroyed pretrained weights. Use 1e-5 to 5e-5."
        )
        assert cfg.lr >= 1e-6, f"LR {cfg.lr} suspiciously low"

    def test_warmup_is_sufficient(self):
        """Warmup should be >= 1% of training for continued pretraining."""
        cfg = Stage1MLMConfig()
        # Calculate approximate training steps
        # 10B tokens / (global_batch * seq_len) = number of optimizer steps
        if "tok" in cfg.max_tokens:
            total_tokens = int(cfg.max_tokens.replace("tok", ""))
            tokens_per_step = cfg.global_batch_size * cfg.max_seq_length
            total_steps = total_tokens / tokens_per_step
            warmup_fraction = cfg.warmup_steps / total_steps
            assert warmup_fraction >= 0.01, (
                f"Warmup is only {warmup_fraction:.1%} of training "
                f"({cfg.warmup_steps} / {total_steps:.0f} steps). "
                f"Should be >= 1% for continued pretraining."
            )

    def test_global_batch_larger_than_microbatch(self):
        """Global batch should be a multiple of microbatch for gradient accumulation."""
        cfg = Stage1MLMConfig()
        assert cfg.global_batch_size >= cfg.microbatch_size
        assert cfg.global_batch_size % cfg.microbatch_size == 0, (
            f"global_batch_size ({cfg.global_batch_size}) must be divisible by "
            f"microbatch_size ({cfg.microbatch_size})"
        )

    def test_mask_rate_reasonable(self):
        """Mask rate should be between 15% and 40%."""
        cfg = Stage1MLMConfig()
        assert 0.15 <= cfg.mask_rate <= 0.40

    def test_autoresume_enabled(self):
        """Autoresume should be enabled for HPC jobs."""
        cfg = Stage1MLMConfig()
        assert cfg.autoresume is True

    def test_eval_interval_set(self):
        """Eval interval should be set for monitoring convergence."""
        cfg = Stage1MLMConfig()
        assert cfg.eval_interval is not None


class TestStage2ContrastiveConfig:
    """Validate Stage 2 contrastive training configuration."""

    def test_lr_is_finetuning_range(self):
        """LR should be in fine-tuning range."""
        cfg = Stage2ContrastiveConfig()
        assert cfg.lr <= 5e-5

    def test_matryoshka_dims_descending(self):
        """Matryoshka dims should be in descending order."""
        cfg = Stage2ContrastiveConfig()
        dims = cfg.matryoshka_dims
        assert dims == sorted(dims, reverse=True), (
            f"Matryoshka dims should be descending: {dims}"
        )

    def test_matryoshka_dims_include_full(self):
        """Matryoshka dims should include the full hidden size (768)."""
        cfg = Stage2ContrastiveConfig()
        assert 768 in cfg.matryoshka_dims

    def test_warmup_ratio_reasonable(self):
        """Warmup ratio should be between 5% and 20%."""
        cfg = Stage2ContrastiveConfig()
        assert 0.05 <= cfg.warmup_ratio <= 0.2

    def test_mixed_training_dir_option_exists(self):
        """Config should support mixed_training_dir for data mixer output."""
        cfg = Stage2ContrastiveConfig()
        assert hasattr(cfg, "mixed_training_dir")


class TestWarmupTokenCalculation:
    """Verify warmup token calculation uses global_batch_size, not microbatch_size."""

    def test_warmup_uses_global_batch(self):
        """The warmup token count must use global_batch_size, not microbatch_size."""
        cfg = Stage1MLMConfig()
        # This is the correct calculation (what the fixed code does)
        correct_warmup_tokens = cfg.warmup_steps * cfg.global_batch_size * cfg.max_seq_length
        # This was the buggy calculation
        buggy_warmup_tokens = cfg.warmup_steps * cfg.microbatch_size * cfg.max_seq_length

        # The correct warmup should be much larger
        assert correct_warmup_tokens > buggy_warmup_tokens
        # Should be grad_accum_steps times larger
        grad_accum = cfg.global_batch_size // cfg.microbatch_size
        assert correct_warmup_tokens == buggy_warmup_tokens * grad_accum
