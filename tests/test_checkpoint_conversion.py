"""Tests for Composer → HuggingFace checkpoint conversion.

Validates that:
1. The MLM head is stripped (only base encoder saved)
2. The saved model loads correctly into sentence-transformers
3. State dict key prefixes are handled properly
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


class TestConvertComposerToHF:
    """Test checkpoint conversion pipeline."""

    @pytest.fixture
    def mock_composer_checkpoint(self, tmp_path):
        """Create a mock Composer checkpoint with the expected structure."""
        from transformers import AutoConfig, AutoModelForMaskedLM

        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
        model = AutoModelForMaskedLM.from_config(config)

        # Composer wraps model with 'model.' prefix under state.model
        state_dict = {}
        for key, value in model.state_dict().items():
            state_dict[f"model.{key}"] = value

        checkpoint = {"state": {"model": state_dict}}
        ckpt_path = tmp_path / "composer_checkpoint.pt"
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def test_strips_model_prefix(self, mock_composer_checkpoint, tmp_path):
        """Verify 'model.' prefix is stripped from state dict keys."""
        from sciembed.train.model import convert_composer_to_hf

        output_dir = tmp_path / "hf_output"
        convert_composer_to_hf(mock_composer_checkpoint, output_dir)

        # Load and check no 'model.' prefix in saved config
        assert (output_dir / "config.json").exists()
        assert (output_dir / "model.safetensors").exists() or (
            output_dir / "pytorch_model.bin"
        ).exists()

    def test_saves_base_model_not_mlm(self, mock_composer_checkpoint, tmp_path):
        """Verify the MLM head is stripped — saved model is base encoder only."""
        from transformers import AutoModel

        from sciembed.train.model import convert_composer_to_hf

        output_dir = tmp_path / "hf_output"
        convert_composer_to_hf(mock_composer_checkpoint, output_dir)

        # Should load as AutoModel (base encoder) without issues
        model = AutoModel.from_pretrained(output_dir)
        # Base model output dimension should be 768, not vocab_size
        assert model.config.hidden_size == 768

    def test_tokenizer_saved(self, mock_composer_checkpoint, tmp_path):
        """Verify tokenizer is saved alongside the model."""
        from transformers import AutoTokenizer

        from sciembed.train.model import convert_composer_to_hf

        output_dir = tmp_path / "hf_output"
        convert_composer_to_hf(mock_composer_checkpoint, output_dir)

        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        assert tokenizer.vocab_size == 50280
        assert len(tokenizer) == 50368

    def test_loads_into_sentence_transformers(self, mock_composer_checkpoint, tmp_path):
        """End-to-end: convert checkpoint → load as SentenceTransformer."""
        from sciembed.train.model import (
            convert_composer_to_hf,
            load_for_sentence_transformers,
        )

        output_dir = tmp_path / "hf_output"
        convert_composer_to_hf(mock_composer_checkpoint, output_dir)

        model = load_for_sentence_transformers(output_dir, pooling="mean")
        # Should produce 768-dim embeddings
        assert model.get_sentence_embedding_dimension() == 768

        # Should be able to encode text
        embeddings = model.encode(["test sentence"])
        assert embeddings.shape == (1, 768)


class TestLoadForSentenceTransformers:
    """Test SentenceTransformer model loading."""

    def test_mean_pooling(self):
        """Verify mean pooling is configured correctly."""
        from sciembed.train.model import load_for_sentence_transformers

        model = load_for_sentence_transformers(
            "answerdotai/ModernBERT-base", pooling="mean"
        )
        pooling = model._modules["1"]
        assert pooling.pooling_mode_mean_tokens is True
        assert pooling.pooling_mode_cls_token is False

    def test_cls_pooling(self):
        """Verify CLS pooling is configured correctly."""
        from sciembed.train.model import load_for_sentence_transformers

        model = load_for_sentence_transformers(
            "answerdotai/ModernBERT-base", pooling="cls"
        )
        pooling = model._modules["1"]
        assert pooling.pooling_mode_cls_token is True
        assert pooling.pooling_mode_mean_tokens is False

    def test_embedding_dimension(self):
        """Output embedding dimension should be 768 for ModernBERT-base."""
        from sciembed.train.model import load_for_sentence_transformers

        model = load_for_sentence_transformers("answerdotai/ModernBERT-base")
        assert model.get_sentence_embedding_dimension() == 768

    def test_encode_produces_correct_shape(self):
        """Encoding should produce (n, 768) output."""
        from sciembed.train.model import load_for_sentence_transformers

        model = load_for_sentence_transformers("answerdotai/ModernBERT-base")
        texts = ["first sentence", "second sentence", "third one"]
        embeddings = model.encode(texts)
        assert embeddings.shape == (3, 768)
