"""Run the official SciRepEval against IBM Granite Embedding R2.

The default SciRepEval encoder (`evaluation/encoders.py::Model`) instantiates
`AutoModel.from_pretrained(...)` and pulls the CLS / mean of the bare encoder's
last_hidden_state. That ignores the pooling / normalization layers that
sentence-transformer-style models expect, and for Granite Embedding R2 it
yielded "Generated 0 embeddings" (silent error inside the encoder forward).

This script subclasses SciRepEval's `Model` to delegate encoding to
`sentence_transformers.SentenceTransformer`, so Granite (and other
SentenceTransformer-style checkpoints) can be evaluated through the rest of the
SciRepEval pipeline unchanged.

Usage (matches scirepeval.py's CLI minus a few esoteric flags):

    python scripts/scirepeval_granite_runner.py \
        --model ibm-granite/granite-embedding-english-r2 \
        --output output/official_eval_results/granite_r2_english.json \
        --batch-size 32

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _import_scirepeval(scirepeval_dir: Path):
    sys.path.insert(0, str(scirepeval_dir))
    from evaluation.encoders import Model  # noqa: E402
    from scirepeval import SciRepEval  # noqa: E402

    return Model, SciRepEval


class SentenceTransformerModel:
    """Drop-in replacement for SciRepEval's `Model` class.

    Implements the minimal surface required by `EmbeddingsGenerator`:
      - `__call__(batch, batch_ids=None) -> torch.Tensor` with shape (B, D)
      - `task_id` setter (no-op; we ignore SciRepEval's control codes since
        this is a "default" variant model)
      - `pooling_mode`, `hidden_dim`, `max_length` attributes for compatibility
    """

    def __init__(
        self,
        base_checkpoint: str,
        pooling_mode: str = "cls",
        max_len: int = 512,
        use_fp16: bool = False,
    ):
        from sentence_transformers import SentenceTransformer

        log.info("Loading SentenceTransformer model: %s", base_checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(base_checkpoint, trust_remote_code=True, device=device)
        # Honor a max-seq-length override if the base model card does not set
        # something sensible; Granite R2 sets 8192 already.
        if hasattr(self.encoder, "max_seq_length"):
            log.info("SentenceTransformer max_seq_length: %s", self.encoder.max_seq_length)
        self.pooling_mode = pooling_mode  # informational only; ST handles pooling internally
        self.hidden_dim = self.encoder.get_sentence_embedding_dimension()
        log.info("Embedding dim: %d", self.hidden_dim)
        self.max_length = max_len
        self.use_fp16 = use_fp16
        self.use_ctrl_codes = False
        self._task_id = None
        self.variant = "default"
        # SciRepEval's evaluator inspects `.tokenizer` for length filtering /
        # batch-size adaptation. Expose the underlying SentenceTransformer
        # tokenizer so those code paths work.
        self.tokenizer = self.encoder.tokenizer

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        # SciRepEval sets a control-token task_id per task; default-variant
        # sentence transformers ignore it.
        self._task_id = value

    def __call__(self, batch, batch_ids=None):
        batch = [batch] if isinstance(batch, str) else batch
        with torch.no_grad():
            emb = self.encoder.encode(
                list(batch),
                batch_size=len(batch),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb)
        # SciRepEval's EmbeddingsGenerator does .detach().cpu().numpy() on each
        # row; numpy lacks native BF16, so cast up to float32 here. Honor
        # use_fp16 only as float16 (numpy-supported), not bfloat16.
        if self.use_fp16:
            return emb.half()
        return emb.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pooling-mode", default="cls")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument(
        "--scirepeval-dir",
        default="${SCIREPEVAL_DIR:-./scirepeval}",
        help="Path to the SciRepEval repo (so its modules import).",
    )
    parser.add_argument(
        "--embeddings-save-path",
        default=None,
        help="Optional dir for SciRepEval's embedding cache.",
    )
    parser.add_argument(
        "--task-list",
        nargs="+",
        default=None,
        help="Optional subset of tasks to run (matches scirepeval.py).",
    )
    parser.add_argument(
        "--excluded-tasks",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Linear-probe seed; overrides upstream pl.seed_everything(42) and "
             "evaluation.evaluator.RANDOM_STATE for symmetric multi-seed comparison.",
    )
    args = parser.parse_args()

    scirepeval_dir = Path(args.scirepeval_dir)
    Model, SciRepEval = _import_scirepeval(scirepeval_dir)

    # Make sure SciRepEval's pl.seed_everything still runs from the upstream import.
    os.chdir(scirepeval_dir)

    # Override seed AFTER scirepeval modules have been imported (their import-time
    # pl.seed_everything(42) runs first; we re-seed and rebind RANDOM_STATE so the
    # per-task linear-probe training picks up the new seed).
    import pytorch_lightning as pl
    pl.seed_everything(args.seed, workers=True)
    import evaluation.evaluator as _ev
    _ev.RANDOM_STATE = args.seed
    log.info("Seed set to %d (pl + evaluator.RANDOM_STATE)", args.seed)

    model = SentenceTransformerModel(
        base_checkpoint=args.model,
        pooling_mode=args.pooling_mode,
        max_len=args.max_len,
    )

    evaluator = SciRepEval(
        tasks_config=str(scirepeval_dir / "scirepeval_tasks.jsonl"),
        batch_size=args.batch_size,
        embedding_save_path=args.embeddings_save_path,
        excluded_tasks=args.excluded_tasks,
        task_list=args.task_list,
    )
    evaluator.evaluate(model, args.output)
    log.info("Done. Results at %s", args.output)


if __name__ == "__main__":
    main()
