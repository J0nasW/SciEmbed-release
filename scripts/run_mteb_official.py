"""Thin runner around the official MTEB package — no custom logic.

Invocation:
    python scripts/run_mteb_official.py \
        --model <path_or_hf_name> \
        --name <output_name> \
        --output-dir <dir>

Runs the official `MTEB(eng, v2)` benchmark via the upstream `mteb` package
(no task subsetting, no custom metrics). Per the project rule
`feedback_official_eval_only.md`: every reported number must come from the
official upstream evaluation, run end-to-end as published.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer


def _patch_retrieval_loader_for_offline_cache() -> None:
    """Work around datasets 4.8.4 + mteb 2.12.15 offline-cache mismatch.

    In offline mode, `get_dataset_config_names(repo)` returns ``['default']``
    for retrieval datasets whose actual cache layout has the canonical
    ``corpus`` / ``qrels`` / ``queries`` configs. mteb's qrels-config fallback
    then never triggers (``'default' in ['default']`` is True) and the
    subsequent ``load_dataset(repo, 'default')`` blows up because the cache
    has no ``default`` build dir. This patch overrides the loader's
    ``dataset_configs`` from the on-disk cache directly so the existing
    fallback logic picks the right config.
    """
    if not os.environ.get("HF_DATASETS_OFFLINE", "") == "1":
        return
    try:
        import mteb.abstasks.retrieval_dataset_loaders as rdl
    except Exception:
        return

    try:
        from datasets.naming import camelcase_to_snakecase
    except Exception:
        return

    hf_cache = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "datasets"),
        )
    )

    def _repo_to_cache_dir(repo: str) -> Path:
        org, name = repo.split("/", 1) if "/" in repo else ("", repo)
        snake = camelcase_to_snakecase(name)
        return hf_cache / (f"{org}___{snake}" if org else snake)

    _orig_init = rdl.RetrievalDatasetLoader.__init__

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-redef]
        _orig_init(self, *args, **kwargs)
        if list(self.dataset_configs) == ["default"]:
            repo_path = _repo_to_cache_dir(self.hf_repo)
            if repo_path.is_dir():
                disk_configs = sorted(p.name for p in repo_path.iterdir() if p.is_dir())
                if disk_configs and disk_configs != ["default"]:
                    self.dataset_configs = disk_configs

    rdl.RetrievalDatasetLoader.__init__ = _patched_init


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument("--name", required=True, help="Output subdir name")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--benchmark", default="MTEB(eng, v2)")
    args = p.parse_args()

    out = Path(args.output_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)

    print(f"[mteb-official] model={args.model}")
    print(f"[mteb-official] benchmark={args.benchmark}")
    print(f"[mteb-official] output={out}")

    _patch_retrieval_loader_for_offline_cache()

    model = SentenceTransformer(args.model, trust_remote_code=True)
    benchmark = mteb.get_benchmark(args.benchmark)
    evaluation = mteb.MTEB(tasks=benchmark)
    evaluation.run(model, output_folder=str(out))

    print(f"[mteb-official] DONE → {out}")


if __name__ == "__main__":
    main()
