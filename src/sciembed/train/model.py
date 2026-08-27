"""Model loading, checkpoint conversion (Composer → HuggingFace)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_modernbert_for_mlm(model_name: str = "answerdotai/ModernBERT-base") -> Any:
    """Load ModernBERT for masked language modeling."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def convert_composer_to_hf(
    composer_checkpoint: str | Path,
    output_dir: str | Path,
    model_name: str = "answerdotai/ModernBERT-base",
) -> Path:
    """Convert a Composer checkpoint to HuggingFace format.

    The Composer checkpoint stores the model state dict under
    'state.model' with a 'model.' prefix on each key. We strip that
    prefix and load into a standard HuggingFace model.

    Args:
        composer_checkpoint: Path to the Composer .pt checkpoint.
        output_dir: Directory to save the HuggingFace model.
        model_name: Base model name for config/tokenizer.

    Returns:
        Path to the saved HuggingFace model directory.
    """
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Composer checkpoint: %s", composer_checkpoint)
    checkpoint = torch.load(composer_checkpoint, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("state", {}).get("model", {})
    if not state_dict:
        # Try alternative key structures
        state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip 'model.' prefix added by Composer's HuggingFaceModel wrapper
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned_state_dict[new_key] = value

    logger.info("Loaded %d parameters", len(cleaned_state_dict))

    # Load the full MLM model to validate weights
    config = AutoConfig.from_pretrained(model_name)
    mlm_model = AutoModelForMaskedLM.from_config(config)

    # Load the trained weights
    missing, unexpected = mlm_model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    # Save the base encoder (without MLM head) for sentence-transformers
    # This ensures AutoModel.from_pretrained() loads cleanly
    from transformers import AutoModel

    base_model = mlm_model.model if hasattr(mlm_model, "model") else mlm_model.base_model
    base_model.save_pretrained(output_dir)
    logger.info("Saved base encoder (MLM head stripped)")

    # Also save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    logger.info("HuggingFace model saved to %s", output_dir)
    return output_dir


def load_for_sentence_transformers(
    model_path: str | Path,
    pooling: str = "mean",
    max_seq_length: int = 8192,
) -> Any:
    """Load a converted HuggingFace model as a SentenceTransformer.

    Args:
        model_path: Path to the HuggingFace model directory.
        pooling: Pooling strategy ("mean" or "cls").
        max_seq_length: Maximum sequence length.

    Returns:
        SentenceTransformer model ready for Stage 2 training.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer

    # ModernBERT auto-enables `reference_compile` whenever Triton is available,
    # which calls `torch.compile()` during forward init and spawns
    # `torch._inductor.compile_worker.subproc_pool`. On Python 3.13 + post-
    # CUDA-init fork environments (e.g. some SLURM GPU clusters) the worker IPC pipes
    # deadlock (`unix_stream_data_wait` ↔ `pipe_read`) and the trainer never
    # reaches step 0. We force it off via `config_args` (the value lives on
    # the model config, not on `__init__`). The speed cost is ~5–10% on H100
    # fp16 and is far better than 0%.
    transformer = Transformer(
        str(model_path),
        max_seq_length=max_seq_length,
        config_args={"reference_compile": False},
    )
    pooling_module = Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=pooling == "mean",
        pooling_mode_cls_token=pooling == "cls",
    )

    model = SentenceTransformer(modules=[transformer, pooling_module])
    return model
