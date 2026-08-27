"""Batch inference wrapper for SciEmbed models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SciEmbedEncoder:
    """High-level batch inference wrapper for SciEmbed embedding models.

    Supports instruction-prefixed encoding for different tasks.
    """

    TASK_PREFIXES = {
        "search_query": "search_query: ",
        "search_document": "search_document: ",
        "classify": "classify: ",
        "cluster": "cluster: ",
        "similarity": "similarity: ",
    }

    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str | None = None,
        max_seq_length: int = 8192,
        embedding_dim: int | None = None,
    ) -> None:
        """Initialize the encoder.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            device: Device to use (auto-detected if None).
            max_seq_length: Maximum sequence length for tokenization.
            embedding_dim: Matryoshka dimension to truncate to (None = full dim).
        """
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            str(model_name_or_path),
            device=device,
            trust_remote_code=True,
        )
        self.model.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.device = device

    def encode(
        self,
        texts: list[str] | str,
        task: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode.
            task: Task prefix to prepend (search_query, search_document, classify, cluster, similarity).
            batch_size: Encoding batch size.
            normalize: Whether to L2-normalize embeddings.
            show_progress: Show progress bar.

        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        # Apply task prefix
        if task and task in self.TASK_PREFIXES:
            prefix = self.TASK_PREFIXES[task]
            texts = [prefix + t for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        # Matryoshka truncation
        if self.embedding_dim is not None and embeddings.shape[1] > self.embedding_dim:
            embeddings = embeddings[:, : self.embedding_dim]
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-12)

        return embeddings

    def similarity(
        self,
        texts_a: list[str],
        texts_b: list[str],
        task: str | None = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Compute pairwise cosine similarity between two sets of texts.

        Args:
            texts_a: First set of texts.
            texts_b: Second set of texts.
            task: Task prefix.
            batch_size: Encoding batch size.

        Returns:
            Similarity matrix of shape (len(texts_a), len(texts_b)).
        """
        emb_a = self.encode(texts_a, task=task, batch_size=batch_size, normalize=True)
        emb_b = self.encode(texts_b, task=task, batch_size=batch_size, normalize=True)
        return emb_a @ emb_b.T
