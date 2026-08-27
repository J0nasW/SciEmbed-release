"""Inference speed benchmarking for SciEmbed vs baselines.

Measures throughput (docs/sec) and latency (ms/doc) across batch sizes
and Matryoshka dimensions. Outputs a table suitable for the paper.

Usage:
    python scripts/benchmark_inference.py [--device cuda] [--warmup 10] [--repeats 50]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer


# Sample scientific abstracts of varying lengths for realistic benchmarking
SAMPLE_TEXTS = [
    "We study the emergence of collective behavior in multi-agent systems using a mean-field approach. "
    "Our theoretical framework predicts phase transitions in coordination dynamics that we validate "
    "through large-scale simulations with up to 10,000 agents.",
    "This paper presents a novel approach to protein structure prediction using graph neural networks. "
    "We introduce an attention mechanism that captures long-range residue interactions and demonstrate "
    "state-of-the-art performance on CASP14 targets with a 15% improvement in GDT-TS score over "
    "existing methods. Our model processes sequences of up to 2048 residues efficiently.",
    "Recent advances in quantum computing have enabled the simulation of molecular systems beyond "
    "the reach of classical computers. We present results from a 127-qubit processor showing quantum "
    "advantage for calculating ground-state energies of lithium hydride and beryllium hydride molecules. "
    "Error mitigation techniques reduce noise by 3x compared to raw hardware results.",
] * 100  # 300 texts total for stable timing


def load_model(name_or_path: str, device: str, max_seq_length: int = 512, precision: str = "fp32") -> SentenceTransformer:
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model_kwargs = {"torch_dtype": dtype_map[precision]} if precision != "fp32" else {}
    model = SentenceTransformer(name_or_path, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
    model.max_seq_length = max_seq_length
    return model


def benchmark_model(
    model: SentenceTransformer,
    texts: list[str],
    batch_sizes: list[int],
    warmup: int = 10,
    repeats: int = 50,
) -> dict[int, dict[str, float]]:
    """Benchmark a model across batch sizes.

    Returns:
        Dict mapping batch_size -> {throughput_docs_sec, latency_ms_per_doc}.
    """
    results = {}

    proc = psutil.Process()

    for bs in batch_sizes:
        # Use a subset that's a multiple of batch_size
        n = min(len(texts), max(bs * 4, 128))
        subset = texts[:n]

        # Warmup
        for _ in range(warmup):
            model.encode(subset[:bs], batch_size=bs, show_progress_bar=False, normalize_embeddings=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        rss_before = proc.memory_info().rss

        # Timed runs
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.encode(subset, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )
        peak_rss_mb = max(rss_before, proc.memory_info().rss) / 1024**2

        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = len(subset) / mean_time
        latency_ms = (mean_time / len(subset)) * 1000

        results[bs] = {
            "throughput_docs_sec": round(throughput, 1),
            "latency_ms_per_doc": round(latency_ms, 2),
            "mean_time_sec": round(mean_time, 4),
            "std_time_sec": round(std_time, 4),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "num_docs": len(subset),
        }

    return results


def benchmark_matryoshka(
    model: SentenceTransformer,
    texts: list[str],
    dims: list[int],
    batch_size: int = 64,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[int, dict[str, float]]:
    """Benchmark Matryoshka dimension truncation overhead."""
    subset = texts[:256]
    results = {}

    for dim in dims:
        # Warmup
        for _ in range(warmup):
            emb = model.encode(subset[:batch_size], batch_size=batch_size, show_progress_bar=False)
            _ = emb[:, :dim]

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            emb = model.encode(subset, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
            emb_trunc = emb[:, :dim]
            # Re-normalize after truncation
            norms = np.linalg.norm(emb_trunc, axis=1, keepdims=True)
            emb_trunc = emb_trunc / np.maximum(norms, 1e-12)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        throughput = len(subset) / mean_time

        results[dim] = {
            "throughput_docs_sec": round(throughput, 1),
            "latency_ms_per_doc": round((mean_time / len(subset)) * 1000, 2),
        }

    return results


def print_results_table(all_results: dict[str, dict], batch_sizes: list[int]) -> None:
    """Print a formatted results table."""
    header = f"{'Model':<30} {'Params':>8}"
    for bs in batch_sizes:
        header += f" {'bs=' + str(bs):>12}"
    print("\n" + "=" * len(header))
    print("Throughput (docs/sec)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, data in all_results.items():
        params = data.get("params", "?")
        row = f"{name:<30} {params:>8}"
        for bs in batch_sizes:
            val = data["results"].get(bs, {}).get("throughput_docs_sec", "-")
            row += f" {val:>12}"
        print(row)

    print("\n" + "=" * 60)
    print("Latency (ms/doc, batch_size=1)")
    print("=" * 60)
    for name, data in all_results.items():
        lat = data["results"].get(1, {}).get("latency_ms_per_doc", "-")
        print(f"  {name:<30} {lat} ms/doc")

    print("\n" + "=" * 60)
    print("Peak memory (batch_size=64)")
    print("=" * 60)
    print(f"  {'Model':<30} {'VRAM (MB)':>12} {'RSS (MB)':>12}")
    for name, data in all_results.items():
        bs64 = data["results"].get(64) or data["results"].get(32) or {}
        v = bs64.get("peak_vram_mb", "-")
        r = bs64.get("peak_rss_mb", "-")
        print(f"  {name:<30} {v:>12} {r:>12}")


def main():
    parser = argparse.ArgumentParser(description="Inference speed benchmarking")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--sciembed-path", default=None, help="Path to SciEmbed model checkpoint")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = parser.parse_args()

    batch_sizes = [1, 8, 32, 64, 128]

    models_to_benchmark = {}

    # SciEmbed
    if args.sciembed_path and Path(args.sciembed_path).exists():
        models_to_benchmark["SciEmbed (149M)"] = {
            "path": args.sciembed_path,
            "params": "149M",
        }
    else:
        print("WARNING: --sciembed-path not provided or not found, skipping SciEmbed")

    # Baselines (will be downloaded from HuggingFace)
    models_to_benchmark.update({
        "BGE-large-en-v1.5 (335M)": {
            "path": "BAAI/bge-large-en-v1.5",
            "params": "335M",
        },
        "SPECTER2 Base (110M)": {
            "path": "allenai/specter2_base",
            "params": "110M",
        },
        "Nomic Embed (149M)": {
            "path": "nomic-ai/nomic-embed-text-v1.5",
            "params": "149M",
        },
    })

    all_results = {}

    for name, model_cfg in models_to_benchmark.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")

        try:
            model = load_model(model_cfg["path"], args.device, precision=args.precision)
            results = benchmark_model(model, SAMPLE_TEXTS, batch_sizes, args.warmup, args.repeats)

            all_results[name] = {
                "params": model_cfg["params"],
                "results": results,
            }

            # Matryoshka benchmarks for SciEmbed only
            if "SciEmbed" in name:
                matryoshka = benchmark_matryoshka(
                    model, SAMPLE_TEXTS, [768, 512, 256, 128],
                    warmup=args.warmup, repeats=args.repeats,
                )
                all_results[name]["matryoshka"] = matryoshka
                print(f"\n  Matryoshka dimensions:")
                for dim, res in matryoshka.items():
                    print(f"    {dim}d: {res['throughput_docs_sec']} docs/sec")

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print_results_table(all_results, batch_sizes)

    if args.output:
        # Convert int keys to strings for JSON
        json_results = {}
        for name, data in all_results.items():
            json_results[name] = {
                "params": data["params"],
                "results": {str(k): v for k, v in data["results"].items()},
            }
            if "matryoshka" in data:
                json_results[name]["matryoshka"] = {str(k): v for k, v in data["matryoshka"].items()}

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
