"""Tests for MLM corpus builder — validates special token IDs and chunking.

Prevents the bug where BERT-classic token IDs (CLS=101, SEP=102, PAD=0)
were hardcoded instead of using the actual ModernBERT tokenizer IDs
(CLS=50281, SEP=50282, PAD=50283).
"""

import numpy as np
import pytest

from sciembed.data.mlm_corpus import chunk_tokens


@pytest.fixture
def modernbert_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


class TestChunkTokensSpecialTokens:
    """Verify chunk_tokens uses correct special token IDs."""

    def test_first_token_is_cls(self, modernbert_tokenizer):
        """First token of every chunk must be the tokenizer's CLS token."""
        tok = modernbert_tokenizer
        fake_ids = list(range(100, 300))
        chunks = chunk_tokens(
            fake_ids, max_seq_length=512, stride=128,
            cls_token_id=tok.cls_token_id,
            sep_token_id=tok.sep_token_id,
            pad_token_id=tok.pad_token_id,
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.input_ids[0] == tok.cls_token_id, (
                f"First token is {chunk.input_ids[0]}, expected CLS={tok.cls_token_id}. "
                f"BERT-classic CLS=101 must NOT be hardcoded."
            )

    def test_sep_token_after_content(self, modernbert_tokenizer):
        """SEP token must appear after the content, before padding."""
        tok = modernbert_tokenizer
        fake_ids = list(range(100, 200))  # 100 tokens, shorter than max
        chunks = chunk_tokens(
            fake_ids, max_seq_length=512, stride=128,
            cls_token_id=tok.cls_token_id,
            sep_token_id=tok.sep_token_id,
            pad_token_id=tok.pad_token_id,
        )
        assert len(chunks) == 1
        chunk = chunks[0]
        # SEP should be at position len(fake_ids) + 1 (after CLS + content)
        sep_pos = len(fake_ids) + 1
        assert chunk.input_ids[sep_pos] == tok.sep_token_id, (
            f"Token at position {sep_pos} is {chunk.input_ids[sep_pos]}, "
            f"expected SEP={tok.sep_token_id}."
        )

    def test_padding_uses_pad_token_id(self, modernbert_tokenizer):
        """Padding positions must use the tokenizer's PAD token, not 0."""
        tok = modernbert_tokenizer
        fake_ids = list(range(100, 200))  # 100 tokens, will need padding to 510
        chunks = chunk_tokens(
            fake_ids, max_seq_length=512, stride=128,
            cls_token_id=tok.cls_token_id,
            sep_token_id=tok.sep_token_id,
            pad_token_id=tok.pad_token_id,
        )
        chunk = chunks[0]
        # Padding starts after CLS + content + SEP
        pad_start = len(fake_ids) + 2
        pad_region = chunk.input_ids[pad_start:]
        assert len(pad_region) > 0, "Test needs a chunk with padding"
        assert all(t == tok.pad_token_id for t in pad_region), (
            f"Padding contains {set(pad_region)}, expected only PAD={tok.pad_token_id}. "
            f"Token ID 0 ('|||IP_ADDRESS|||') must NOT be used as padding."
        )

    def test_no_bert_classic_ids_in_framing(self, modernbert_tokenizer):
        """Chunks must not contain BERT-classic IDs (101, 102) as framing tokens."""
        tok = modernbert_tokenizer
        # Use token IDs that don't include 101 or 102 in the content
        fake_ids = list(range(200, 400))
        chunks = chunk_tokens(
            fake_ids, max_seq_length=512, stride=128,
            cls_token_id=tok.cls_token_id,
            sep_token_id=tok.sep_token_id,
            pad_token_id=tok.pad_token_id,
        )
        for chunk in chunks:
            assert chunk.input_ids[0] != 101, "CLS position has BERT-classic ID 101"
            content_end = int(chunk.attention_mask.sum()) - 1
            assert chunk.input_ids[content_end] != 102, "SEP position has BERT-classic ID 102"

    def test_attention_mask_matches_content(self, modernbert_tokenizer):
        """Attention mask 1s should cover CLS + content + SEP, 0s for padding."""
        tok = modernbert_tokenizer
        fake_ids = list(range(100, 200))
        chunks = chunk_tokens(
            fake_ids, max_seq_length=512, stride=128,
            cls_token_id=tok.cls_token_id,
            sep_token_id=tok.sep_token_id,
            pad_token_id=tok.pad_token_id,
        )
        chunk = chunks[0]
        expected_ones = len(fake_ids) + 2  # CLS + content + SEP
        assert chunk.attention_mask[:expected_ones].sum() == expected_ones
        assert chunk.attention_mask[expected_ones:].sum() == 0

    def test_all_token_ids_in_vocab_range(self, modernbert_tokenizer):
        """All token IDs must be in [0, vocab_size)."""
        tok = modernbert_tokenizer
        fake_ids = list(range(100, 300))
        chunks = chunk_tokens(
            fake_ids, max_seq_length=512, stride=128,
            cls_token_id=tok.cls_token_id,
            sep_token_id=tok.sep_token_id,
            pad_token_id=tok.pad_token_id,
        )
        vocab_size = len(tok)
        for chunk in chunks:
            assert all(0 <= t < vocab_size for t in chunk.input_ids), (
                f"Token IDs out of range [0, {vocab_size})"
            )


class TestOptimizerSafety:
    """Guard tests to prevent DecoupledAdamW weight decay catastrophe."""

    def test_mlm_uses_standard_adamw(self):
        """Stage 1 MLM must NOT use Composer's DecoupledAdamW.

        DecoupledAdamW applies param *= (1 - decay_factor * wd) where
        decay_factor = lr/initial_lr. With wd=0.01, this destroys 46% of
        pretrained weights in 500 steps. Standard AdamW applies
        param *= (1 - lr * wd) ≈ (1 - 2e-7), which is negligible.
        """
        import ast
        from pathlib import Path

        mlm_path = Path("src/sciembed/train/mlm.py")
        source = mlm_path.read_text()

        assert "DecoupledAdamW(" not in source, (
            "mlm.py must not use DecoupledAdamW — it destroys pretrained weights. "
            "Use torch.optim.AdamW instead."
        )
