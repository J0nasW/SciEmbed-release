"""Tests for the attention-mask-aware MLM collator.

Validates the fix for the padding masking bug where ~45% of masked positions
were in zero-padded regions, inflating loss from ~1.1 to ~8.0.
"""

import numpy as np
import pytest
import torch

from sciembed.train.mlm import _AttentionMaskAwareMLMCollator


@pytest.fixture
def mock_tokenizer():
    """Create a minimal mock tokenizer for testing."""

    class MockTokenizer:
        mask_token = "[MASK]"
        pad_token_id = 50283
        _vocab = {f"tok_{i}": i for i in range(50368)}
        _vocab["[MASK]"] = 50284

        def __len__(self):
            return 50368

        def convert_tokens_to_ids(self, token):
            return self._vocab.get(token, 0)

        def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
            # Only mark token 101 (CLS) and 102 (SEP) as special
            return [1 if t in (101, 102) else 0 for t in token_ids]

    return MockTokenizer()


def _make_sample(active_len: int, total_len: int = 512) -> dict:
    """Create a sample with `active_len` real tokens and padding to `total_len`."""
    input_ids = np.random.randint(100, 50000, size=total_len, dtype=np.int64)
    attention_mask = np.zeros(total_len, dtype=np.int64)
    attention_mask[:active_len] = 1
    # Zero out padding region token IDs (matches MDS corpus format)
    input_ids[active_len:] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestAttentionMaskAwareMLMCollator:
    def test_no_masking_in_padding(self, mock_tokenizer):
        """Core bug fix: padding tokens must never be masked."""
        collator = _AttentionMaskAwareMLMCollator(mock_tokenizer, mlm_probability=0.3)
        sample = _make_sample(active_len=200, total_len=512)

        batch = collator([sample])
        labels = batch["labels"][0]

        # Padding region: positions 200-511 must all be -100
        padding_labels = labels[200:]
        assert (padding_labels == -100).all(), (
            f"Found {(padding_labels != -100).sum()} masked positions in padding!"
        )

    def test_masking_only_in_content(self, mock_tokenizer):
        """All masked positions should be within the attention_mask=1 region."""
        collator = _AttentionMaskAwareMLMCollator(mock_tokenizer, mlm_probability=0.3)
        sample = _make_sample(active_len=300, total_len=512)

        batch = collator([sample])
        labels = batch["labels"][0]
        attn = batch["attention_mask"][0]

        masked_positions = (labels != -100)
        # Every masked position must have attention_mask=1
        assert (attn[masked_positions] == 1).all()

    def test_masking_ratio_on_content(self, mock_tokenizer):
        """Masking ratio should be ~30% of content tokens, not total tokens."""
        collator = _AttentionMaskAwareMLMCollator(mock_tokenizer, mlm_probability=0.3)
        active_len = 400

        # Average over multiple samples for stable ratio
        total_masked = 0
        n_samples = 50
        for _ in range(n_samples):
            sample = _make_sample(active_len=active_len, total_len=1024)
            batch = collator([sample])
            labels = batch["labels"][0]
            total_masked += (labels != -100).sum().item()

        avg_ratio = total_masked / (n_samples * active_len)
        # Should be ~0.3 (±0.05 tolerance for randomness)
        assert 0.2 < avg_ratio < 0.4, f"Masking ratio {avg_ratio:.3f} not near 0.3"

    def test_fully_active_sequence(self, mock_tokenizer):
        """A sequence with no padding should work normally."""
        collator = _AttentionMaskAwareMLMCollator(mock_tokenizer, mlm_probability=0.3)
        sample = _make_sample(active_len=512, total_len=512)

        batch = collator([sample])
        labels = batch["labels"][0]

        masked_count = (labels != -100).sum().item()
        # ~30% of 512 ≈ 153, allow some tolerance
        assert 100 < masked_count < 200

    def test_mostly_padding(self, mock_tokenizer):
        """A sequence that's mostly padding should still mask only the content."""
        collator = _AttentionMaskAwareMLMCollator(mock_tokenizer, mlm_probability=0.3)
        sample = _make_sample(active_len=50, total_len=8192)

        batch = collator([sample])
        labels = batch["labels"][0]

        # All masked positions should be in first 50 tokens
        masked_positions = (labels != -100).nonzero(as_tuple=True)[0]
        if len(masked_positions) > 0:
            assert masked_positions.max().item() < 50

    def test_batch_of_multiple_samples(self, mock_tokenizer):
        """Collator should handle batches with different active lengths."""
        collator = _AttentionMaskAwareMLMCollator(mock_tokenizer, mlm_probability=0.3)
        samples = [
            _make_sample(active_len=100, total_len=512),
            _make_sample(active_len=400, total_len=512),
            _make_sample(active_len=512, total_len=512),
        ]

        batch = collator(samples)
        assert batch["input_ids"].shape == (3, 512)
        assert batch["attention_mask"].shape == (3, 512)
        assert batch["labels"].shape == (3, 512)

        # Check each sample individually
        for i, active_len in enumerate([100, 400, 512]):
            padding_labels = batch["labels"][i, active_len:]
            assert (padding_labels == -100).all(), f"Sample {i} has masked padding"

    def test_mask_replace_random_keep_ratios(self, mock_tokenizer):
        """Verify 80/10/10 split: MASK token / random token / keep original."""
        collator = _AttentionMaskAwareMLMCollator(mock_tokenizer, mlm_probability=0.5)

        mask_id = mock_tokenizer.convert_tokens_to_ids("[MASK]")
        total_mask_replaced = 0
        total_random_replaced = 0
        total_kept = 0
        n_trials = 100

        for _ in range(n_trials):
            sample = _make_sample(active_len=500, total_len=500)
            original_ids = torch.as_tensor(sample["input_ids"].copy(), dtype=torch.long)
            batch = collator([sample])

            masked = batch["labels"][0] != -100
            masked_input = batch["input_ids"][0][masked]
            original_at_masked = original_ids[masked]

            total_mask_replaced += (masked_input == mask_id).sum().item()
            total_random_replaced += (
                (masked_input != mask_id) & (masked_input != original_at_masked)
            ).sum().item()
            total_kept += (masked_input == original_at_masked).sum().item()

        total = total_mask_replaced + total_random_replaced + total_kept
        if total > 0:
            mask_ratio = total_mask_replaced / total
            random_ratio = total_random_replaced / total
            keep_ratio = total_kept / total
            # 80/10/10 with generous tolerance
            assert 0.7 < mask_ratio < 0.9, f"MASK ratio {mask_ratio:.2f}"
            assert 0.03 < random_ratio < 0.17, f"Random ratio {random_ratio:.2f}"
            assert 0.03 < keep_ratio < 0.17, f"Keep ratio {keep_ratio:.2f}"

    def test_output_dtypes(self, mock_tokenizer):
        """Output tensors should be the right dtypes."""
        collator = _AttentionMaskAwareMLMCollator(mock_tokenizer, mlm_probability=0.3)
        sample = _make_sample(active_len=256, total_len=512)

        batch = collator([sample])
        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

    def test_special_tokens_not_masked(self, mock_tokenizer):
        """CLS (101) and SEP (102) tokens should never be masked."""
        collator = _AttentionMaskAwareMLMCollator(mock_tokenizer, mlm_probability=1.0)
        sample = _make_sample(active_len=512, total_len=512)
        # Put CLS at start, SEP at end of content
        sample["input_ids"][0] = 101
        sample["input_ids"][511] = 102

        batch = collator([sample])
        labels = batch["labels"][0]
        # CLS and SEP should be -100 (not masked)
        assert labels[0].item() == -100, "CLS token was masked"
        assert labels[511].item() == -100, "SEP token was masked"
