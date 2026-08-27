"""Run official SciRepEval at multiple Matryoshka truncation dims.

For each dim in [768, 512, 256, 128]:
  1. Sets TRUNCATE_DIM env so the patched encoder truncates pooled embeddings.
  2. Runs the official scirepeval.py script as a subprocess.
  3. Saves results to <output_dir>/<name>_dim<dim>.json.

Why subprocess: scirepeval mutates global pl.seed and CUDA state, and we want
each dim eval to start clean.

Usage:
    python scripts/run_scirepeval_matryoshka.py \
        --model <path> \
        --name <output_name> \
        --output-dir <dir>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCIREPEVAL_DIR = Path(
    os.environ.get(
        "SCIREPEVAL_DIR",
        str(Path(__file__).resolve().parent.parent / "eval" / "scirepeval_official"),
    )
)
DIMS = [768, 512, 256, 128]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pooling", default="mean")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dims", nargs="*", type=int, default=DIMS)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = SCIREPEVAL_DIR / "embeddings"
    emb_dir.mkdir(exist_ok=True)

    for dim in args.dims:
        out_file = out_dir / f"{args.name}_dim{dim}.json"
        if out_file.exists():
            try:
                import json
                with open(out_file) as f:
                    d = json.load(f)
                if len(d) >= 20:
                    print(f"[matryoshka] SKIP dim={dim} (already done with {len(d)} tasks)")
                    continue
            except Exception:
                pass

        # Wipe stale embeddings
        for f in emb_dir.glob("*"):
            if f.is_file():
                f.unlink()

        env = os.environ.copy()
        env["TRUNCATE_DIM"] = str(dim)

        cmd = [
            sys.executable, "scirepeval.py",
            "-m", args.model,
            "--pooling-mode", args.pooling,
            "--batch-size", str(args.batch_size),
            "--output", str(out_file),
            "--embeddings-save-path", str(emb_dir),
        ]
        print(f"\n[matryoshka] dim={dim}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(SCIREPEVAL_DIR), env=env, check=False)


if __name__ == "__main__":
    main()
