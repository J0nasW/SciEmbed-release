"""Qualitative analyses for the SciEmbed paper.

Three subcommands:
  doc-length        Tokenize SciRepEval inputs across tasks, plot histogram.
  umap              Encode FoS-labeled papers with SciEmbed-CTX, UMAP-2D scatter.
  citation-examples Pull citation-context examples from the datalake.

Each subcommand writes its outputs to paper/latex/figures/generated/ (figures)
or output/qualitative/ (data).

Usage:
    python scripts/build_qualitative_analysis.py doc-length
    python scripts/build_qualitative_analysis.py umap
    python scripts/build_qualitative_analysis.py citation-examples
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "paper" / "latex" / "figures" / "generated"
DATA_OUT = ROOT / "output" / "qualitative"
FIGS.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

CTX_MODEL_PATH = ROOT / "output" / "stage2_ctx_full_seed123" / "final"

STYLE = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}
plt.rcParams.update(STYLE)


# Subcommand 1: SciRepEval document length histogram

# (display name, scirepeval config, split, text-build fn)
DOC_LENGTH_TASKS = [
    ("FoS",            "fos",                "evaluation",
        lambda r: f"{r.get('title','')} {r.get('abstract','') or ''}"),
    ("Biomimicry",     "biomimicry",         "evaluation",
        lambda r: f"{r.get('title','')} {r.get('abstract','') or ''}"),
    ("Citation Count", "cite_count",         "evaluation",
        lambda r: f"{r.get('title','')} {r.get('abstract','') or ''}"),
    ("MeSH",           "mesh_descriptors",   "evaluation",
        lambda r: f"{r.get('title','')} {r.get('abstract','') or ''}"),
    ("DRSM",           "drsm",               "evaluation",
        lambda r: f"{r.get('title','')} {r.get('abstract','') or ''}"),
]
SAMPLE_PER_TASK = 1500


def cmd_doc_length(args: argparse.Namespace) -> None:
    import duckdb

    # Cached raw token-length arrays — lets cosmetic figure tweaks skip the
    # (~minutes-long) tokenization pass. Delete this file to force a refresh.
    cache_path = DATA_OUT / "doc_length_arrays.npz"
    if cache_path.exists() and not getattr(args, "refresh", False):
        print(f"Loading cached length arrays from {cache_path} (pass --refresh to retokenize)")
        npz = np.load(cache_path, allow_pickle=False)
        scirepeval_pooled = npz["scirepeval_pooled"].tolist()
        abs_lens = npz["abs_lens"].tolist()
        ft_lens = npz["ft_lens"].tolist()
        scirepeval_per_task = {
            k[len("task__"):]: npz[k].tolist()
            for k in npz.files if k.startswith("task__")
        }
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok_path = CTX_MODEL_PATH / "tokenizer"
        if tok_path.exists():
            tok = AutoTokenizer.from_pretrained(str(tok_path))
        else:
            tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        print(f"Tokenizer: {tok.__class__.__name__}, vocab={tok.vocab_size}")

        def tok_lengths(texts: list[str]) -> list[int]:
            out: list[int] = []
            B = 256
            for i in range(0, len(texts), B):
                chunk = texts[i:i + B]
                encs = tok(chunk, add_special_tokens=True, truncation=False, padding=False)
                out.extend(len(x) for x in encs["input_ids"])
            return out

        rng = random.Random(42)
        scirepeval_pooled = []
        scirepeval_per_task = {}
        for display, cfg, split, build in DOC_LENGTH_TASKS:
            try:
                ds = load_dataset("allenai/scirepeval", cfg, split=split)
            except Exception as e:
                print(f"  {display}: skip ({type(e).__name__}: {str(e)[:80]})")
                continue
            n = min(SAMPLE_PER_TASK, len(ds))
            idxs = rng.sample(range(len(ds)), n)
            texts = [build(ds[i]) for i in idxs]
            lens = tok_lengths(texts)
            scirepeval_per_task[display] = lens
            scirepeval_pooled.extend(lens)
            print(f"  scirepeval/{display}: n={n} median={int(np.median(lens))} "
                  f"p95={int(np.percentile(lens,95))} max={max(lens)}")

        # Natural distributions from the datalake
        DATALAKE = "/path/to/data/science_datalake/datalake.duckdb"
        print(f"\nSampling natural distributions from {DATALAKE} ...")
        con = duckdb.connect(DATALAKE, read_only=True)

        print("  s2ag.abstracts (real scientific abstracts) ...")
        abs_rows = con.execute("""
            SELECT abstract FROM (
                SELECT abstract FROM s2ag.abstracts
                WHERE abstract IS NOT NULL AND length(abstract) > 100
            ) USING SAMPLE 5000
        """).fetchall()
        abs_texts = [r[0] for r in abs_rows]
        abs_lens = tok_lengths(abs_texts)
        print(f"    n={len(abs_lens)} median={int(np.median(abs_lens))} "
              f"p95={int(np.percentile(abs_lens,95))} max={max(abs_lens)} "
              f"frac>512={np.mean(np.array(abs_lens)>512):.3f}")

        print("  fulltext.papers (PMC/S2ORC body text) ...")
        ft_rows = con.execute("""
            SELECT title, abstract, text FROM (
                SELECT title, abstract, text FROM fulltext.papers
                WHERE has_full_text = TRUE AND text IS NOT NULL
                  AND text_length BETWEEN 1000 AND 200000
            ) USING SAMPLE 3000
        """).fetchall()
        ft_texts = [f"{r[0] or ''} {r[1] or ''} {r[2]}" for r in ft_rows]
        ft_lens = tok_lengths(ft_texts)
        print(f"    n={len(ft_lens)} median={int(np.median(ft_lens))} "
              f"p95={int(np.percentile(ft_lens,95))} max={max(ft_lens)} "
              f"frac>512={np.mean(np.array(ft_lens)>512):.3f} "
              f"frac>8192={np.mean(np.array(ft_lens)>8192):.3f}")

        # Persist raw arrays so cosmetic replots skip tokenization next time.
        np.savez_compressed(
            cache_path,
            scirepeval_pooled=np.asarray(scirepeval_pooled, dtype=np.int32),
            abs_lens=np.asarray(abs_lens, dtype=np.int32),
            ft_lens=np.asarray(ft_lens, dtype=np.int32),
            **{f"task__{k}": np.asarray(v, dtype=np.int32) for k, v in scirepeval_per_task.items()},
        )
        print(f"  Cached raw length arrays -> {cache_path}")

    # Plot — three distributions on log-scale CDF.
    # Colorblind-safe Okabe-Ito palette + redundant linestyle encoding so the
    # series remain distinguishable in grayscale and under all common CVD types.
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    series = [
        ("SciRepEval inputs",            np.array(scirepeval_pooled), "#E69F00", "-"),    # orange, solid
        ("Real abstracts (S2AG)",        np.array(abs_lens),          "#0072B2", "--"),   # blue,   dashed
        ("Full-text papers (PMC/S2ORC)", np.array(ft_lens),           "#009E73", ":"),    # green,  dotted
    ]
    for name, lens, color, ls in series:
        x = np.sort(lens)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, linewidth=1.5, linestyle=ls, label=name)

    for limit, label, color in [(512, "SciBERT/SPECTER2 limit (512)", "#374151"),
                                 (8192, "ModernBERT limit (8K)", "#374151")]:
        ax.axvline(limit, color=color, linestyle="--", linewidth=0.7, alpha=0.85)
    ax.text(540, 0.04, "512", fontsize=6, color="#374151")
    ax.text(8400, 0.04, "8K", fontsize=6, color="#374151")

    ax.set_xscale("log")
    ax.set_xlim(50, 60000)
    ax.set_xlabel("Tokens (ModernBERT tokenizer)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              framealpha=0.0, edgecolor="none", fontsize=6.5,
              ncol=3, handlelength=1.6, columnspacing=1.2, borderaxespad=0.)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.35)

    fig.savefig(FIGS / "doc_length_cdf.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "doc_length_cdf.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: doc_length_cdf.pdf/png")

    # Aggregate stats — used by paper prose
    def stats(arr: np.ndarray) -> dict:
        return {"n": int(len(arr)),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": int(max(arr)),
                "frac_over_512": float(np.mean(arr > 512)),
                "frac_over_2048": float(np.mean(arr > 2048)),
                "frac_over_8192": float(np.mean(arr > 8192))}
    summary = {
        "scirepeval_pooled": stats(np.array(scirepeval_pooled)),
        "scirepeval_per_task": {k: stats(np.array(v)) for k, v in scirepeval_per_task.items()},
        "s2ag_abstracts":   stats(np.array(abs_lens)),
        "fulltext_papers":  stats(np.array(ft_lens)),
    }
    (DATA_OUT / "doc_length_stats.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved: output/qualitative/doc_length_stats.json")
    all_lengths = scirepeval_per_task  # keep last lines below working
    return

    # Plot — overlaid CDFs (cleaner than overlapping histograms)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    colors = {"FoS": "#2563EB", "Biomimicry": "#059669", "Citation Count": "#D97706",
              "MeSH": "#DC2626", "DRSM": "#7C3AED"}
    for display, lens in all_lengths.items():
        sorted_lens = np.sort(lens)
        cdf = np.arange(1, len(sorted_lens) + 1) / len(sorted_lens)
        ax.plot(sorted_lens, cdf, color=colors.get(display, "gray"),
                linewidth=1.2, label=display)

    ax.axvline(512, color="#374151", linestyle="--", linewidth=0.8, alpha=0.85)
    ax.text(520, 0.05, "512", fontsize=6.5, color="#374151")
    ax.axvline(8192, color="#374151", linestyle="--", linewidth=0.8, alpha=0.85)
    ax.text(8200, 0.05, "8192", fontsize=6.5, color="#374151")

    ax.set_xscale("log")
    ax.set_xlim(20, 16000)
    ax.set_xlabel("Tokens (ModernBERT tokenizer)")
    ax.set_ylabel("Cumulative fraction of inputs")
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="none", ncol=1)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.35)

    fig.savefig(FIGS / "doc_length_cdf.pdf")
    fig.savefig(FIGS / "doc_length_cdf.png")
    plt.close(fig)
    print(f"  Saved: doc_length_cdf.pdf/png")

    # Aggregate stats
    agg = {disp: {"n": len(l), "median": float(np.median(l)),
                   "p90": float(np.percentile(l, 90)),
                   "p95": float(np.percentile(l, 95)),
                   "p99": float(np.percentile(l, 99)),
                   "max": int(max(l)),
                   "frac_over_512": float(np.mean(np.array(l) > 512))}
           for disp, l in all_lengths.items()}
    (DATA_OUT / "doc_length_stats.json").write_text(json.dumps(agg, indent=2))
    print(f"  Saved: output/qualitative/doc_length_stats.json")
    pooled = np.concatenate(list(all_lengths.values()))
    print(f"\n  POOLED: n={len(pooled)} median={int(np.median(pooled))} "
          f"p95={int(np.percentile(pooled,95))} frac>512={np.mean(pooled>512):.3f}")


# Subcommand 2: UMAP of FoS-labeled papers

UMAP_N_PER_CLASS = 350
UMAP_TOP_K_CLASSES = 8


UMAP_MODELS = [
    ("Nomic ModernBERT",     "nomic-ai/modernbert-embed-base"),
    ("Granite R2",           "ibm-granite/granite-embedding-english-r2"),
    ("SciEmbed-FULL (ours)", None),  # filled in at runtime from CTX_MODEL_PATH
]


def cmd_umap(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    import umap  # type: ignore
    import torch

    # Three matched-architecture (149M ModernBERT-base) embedders with
    # different contrastive supervision sources:
    #   Nomic ModernBERT — web-scale general retrieval
    #   Granite R2       — general enterprise retrieval
    #   SciEmbed-CTX     — citation-context supervision
    # Same backbone, same pooling family, same raw input format. Any
    # structural difference in the UMAP is attributable to the supervision.
    models = [(n, m or str(CTX_MODEL_PATH)) for n, m in UMAP_MODELS]

    print(f"Loading FoS task ...")
    ds = load_dataset("allenai/scirepeval", "fos", split="evaluation")
    from collections import Counter
    single = [(r["title"], r["abstract"], r["labels_text"][0], r["corpus_id"])
              for r in ds
              if r.get("labels_text") and len(r["labels_text"]) == 1
              and r.get("title") and r.get("abstract") and r.get("corpus_id")]
    counter = Counter(lab for _, _, lab, _ in single)
    top_classes = [lab for lab, _ in counter.most_common(UMAP_TOP_K_CLASSES)]
    print(f"  Top {UMAP_TOP_K_CLASSES} classes: {top_classes}")

    rng = random.Random(42)
    samples = []
    for cls in top_classes:
        cls_papers = [(t, a, cls, cid) for t, a, lab, cid in single if lab == cls]
        rng.shuffle(cls_papers)
        samples.extend(cls_papers[:UMAP_N_PER_CLASS])
    print(f"  Total samples: {len(samples)}")

    texts = [f"{t} {a}" for t, a, _, _ in samples]
    labels = [lab for _, _, lab, _ in samples]
    corpus_ids = [cid for _, _, _, cid in samples]

    # Encode all three models first; AlignedUMAP needs all embedding
    # matrices up front so it can jointly optimise the projections.
    all_emb = {}
    for name, model_id in models:
        print(f"\n=== {name} ({model_id}) ===")
        model = SentenceTransformer(model_id, trust_remote_code=True)
        print(f"  Encoding {len(texts)} papers ...")
        emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                           normalize_embeddings=True)
        print(f"  Embedding shape: {emb.shape}")
        all_emb[name] = emb
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aligned UMAP — same papers, different encoders, identity mapping
    # across datasets so same-identity points land in similar positions.
    print(f"\nRunning AlignedUMAP across {len(models)} embedding spaces ...")
    emb_list = [all_emb[name] for name, _ in models]
    n = len(texts)
    relations = [{i: i for i in range(n)} for _ in range(len(models) - 1)]
    aligned_mapper = umap.AlignedUMAP(
        n_neighbors=25, min_dist=0.15, n_components=2,
        metric="cosine",
        alignment_window_size=2,
        alignment_regularisation=1e-2,
    ).fit(emb_list, relations=relations)
    aligned_embeddings = list(aligned_mapper.embeddings_)
    all_coords = {name: np.asarray(aligned_embeddings[i])
                  for i, (name, _) in enumerate(models)}

    # Plot — 3 panels side-by-side. Okabe-Ito palette with redundant markers
    # so classes remain separable in grayscale and under CVD.
    okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                 "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for ax, (name, _) in zip(axes, models):
        coords = all_coords[name]
        for i, cls in enumerate(top_classes):
            mask = np.array([l == cls for l in labels])
            ax.scatter(coords[mask, 0], coords[mask, 1], s=7.0, alpha=0.65,
                       color=okabe_ito[i % len(okabe_ito)],
                       marker=markers[i % len(markers)],
                       label=cls if ax is axes[0] else None, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    axes[0].set_ylabel("UMAP-2")
    for ax in axes:
        ax.set_xlabel("UMAP-1")

    # Shared legend below all 3 panels
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=4, framealpha=0,
               edgecolor="none", handlelength=0.8, fontsize=8,
               markerscale=1.6, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    fig.savefig(FIGS / "umap_fos.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS / "umap_fos.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: umap_fos.pdf/png (3-panel field-colored)")

    # ----- Impact-colored variant (main-paper figure) -----
    # Same UMAP coords; color by log(citationcount + 1) from S2AG datalake.
    # The hypothesis: SciEmbed's citation-context supervision organises the
    # embedding space around paper impact, mechanistically explaining the
    # +2.3 regression-cluster lead in Section 5.1.
    print("\nLooking up citation counts in S2AG datalake ...")
    import duckdb
    DATALAKE = "/path/to/data/science_datalake/datalake.duckdb"
    con = duckdb.connect(DATALAKE, read_only=True)
    ids_str = ",".join(str(c) for c in corpus_ids)
    rows = con.execute(
        f"SELECT corpusid, citationcount FROM s2ag.papers "
        f"WHERE corpusid IN ({ids_str})"
    ).fetchall()
    con.close()
    cid_to_count = {r[0]: (r[1] or 0) for r in rows}
    citation_counts = np.array([cid_to_count.get(c, 0) for c in corpus_ids],
                               dtype=np.float64)
    n_resolved = int((citation_counts > 0).sum())
    print(f"  Resolved citation count for {n_resolved}/{len(citation_counts)} papers")
    print(f"  Min/median/95th/max: {int(citation_counts.min())} / "
          f"{int(np.median(citation_counts))} / "
          f"{int(np.percentile(citation_counts, 95))} / "
          f"{int(citation_counts.max())}")

    # Cap at 99th percentile to avoid one paper dominating the colour scale
    cap = float(np.percentile(citation_counts, 99))
    log_counts = np.log1p(np.minimum(citation_counts, cap))

    fig2, axes2 = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for ax, (name, _) in zip(axes2, models):
        coords = all_coords[name]
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=log_counts, cmap="magma", s=6.0, alpha=0.75,
                        linewidths=0, vmin=0, vmax=float(log_counts.max()))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("UMAP-1")
    axes2[0].set_ylabel("UMAP-2")
    cbar = fig2.colorbar(sc, ax=axes2.tolist(), location="bottom",
                         shrink=0.55, aspect=40, pad=0.18)
    cbar.set_label(r"$\log(\mathrm{citation\ count} + 1)$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig2.savefig(FIGS / "umap_fos_impact.pdf", bbox_inches="tight",
                 pad_inches=0.02)
    fig2.savefig(FIGS / "umap_fos_impact.png", bbox_inches="tight",
                 pad_inches=0.02, dpi=200)
    plt.close(fig2)
    print(f"  Saved: umap_fos_impact.pdf/png (3-panel, citation-coloured)")

    # ----- High-impact-only variant (Option D) -----
    # Top decile by citation count highlighted, rest in faded gray.
    pos_counts = citation_counts[citation_counts > 0]
    thresh = float(np.percentile(pos_counts, 90)) if len(pos_counts) else 0
    is_high = citation_counts >= thresh
    print(f"\nHigh-impact threshold (90th percentile of resolved): {thresh:.0f} citations")
    print(f"  Highlighted papers: {int(is_high.sum())}/{len(is_high)}")

    fig3, axes3 = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for ax, (name, _) in zip(axes3, models):
        coords = all_coords[name]
        ax.scatter(coords[~is_high, 0], coords[~is_high, 1],
                   c="lightgray", s=4.0, alpha=0.4, linewidths=0, zorder=1)
        ax.scatter(coords[is_high, 0], coords[is_high, 1],
                   c="crimson", s=14.0, alpha=0.75, linewidths=0, zorder=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("UMAP-1")
    axes3[0].set_ylabel("UMAP-2")
    fig3.suptitle(
        f"Top decile by citation count ($\\geq {int(thresh)}$ citations, "
        f"$n={int(is_high.sum())}$) in crimson, remaining papers in gray.",
        fontsize=8, y=0.02)
    fig3.tight_layout(rect=[0, 0.04, 1, 1])
    fig3.savefig(FIGS / "umap_fos_highimpact.pdf", bbox_inches="tight",
                 pad_inches=0.02)
    fig3.savefig(FIGS / "umap_fos_highimpact.png", bbox_inches="tight",
                 pad_inches=0.02, dpi=200)
    plt.close(fig3)
    print(f"  Saved: umap_fos_highimpact.pdf/png (3-panel, top-decile highlighted)")

    # ----- NN agreement matrix (Option B) -----
    # For each paper, compute top-k nearest neighbours in each model under
    # cosine. Off-diagonal entries = mean fraction of overlap.
    # Hypothesis: Nomic <-> Granite high overlap (both general retrieval),
    # SciEmbed <-> {Nomic, Granite} lower (different supervision source).
    from sklearn.metrics.pairwise import cosine_similarity
    k_nn = 50
    print(f"\nComputing top-{k_nn} NN agreement matrix across {len(models)} models ...")
    model_names = [name for name, _ in models]
    top_neighbours = {}
    for name in model_names:
        emb = all_emb[name]
        sim = cosine_similarity(emb)
        np.fill_diagonal(sim, -np.inf)
        top_neighbours[name] = np.argpartition(-sim, k_nn, axis=1)[:, :k_nn]
    overlap_matrix = np.zeros((len(model_names), len(model_names)))
    for i, a in enumerate(model_names):
        for j, b in enumerate(model_names):
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue
            overlaps = []
            top_a, top_b = top_neighbours[a], top_neighbours[b]
            for r in range(top_a.shape[0]):
                overlaps.append(len(set(top_a[r].tolist()) & set(top_b[r].tolist())) / k_nn)
            overlap_matrix[i, j] = float(np.mean(overlaps))
    print("Top-50 NN agreement matrix (off-diagonal = fraction of shared neighbours):")
    print(f"  {'':22s} " + "  ".join(f"{n[:14]:>14s}" for n in model_names))
    for i, a in enumerate(model_names):
        print(f"  {a[:22]:22s} " + "  ".join(f"{overlap_matrix[i,j]:>14.3f}" for j in range(len(model_names))))

    short_names = {"Nomic ModernBERT": "Nomic", "Granite R2": "Granite",
                   "SciEmbed-FULL (ours)": "SciEmbed"}
    short = [short_names.get(n, n) for n in model_names]
    fig4, ax4 = plt.subplots(figsize=(3.4, 3.0))
    im = ax4.imshow(overlap_matrix, cmap="viridis", vmin=0, vmax=1.0)
    ax4.set_xticks(range(len(short))); ax4.set_yticks(range(len(short)))
    ax4.set_xticklabels(short, rotation=20, ha="right", fontsize=8)
    ax4.set_yticklabels(short, fontsize=8)
    for i in range(len(short)):
        for j in range(len(short)):
            ax4.text(j, i, f"{overlap_matrix[i,j]:.2f}",
                     ha="center", va="center",
                     color="white" if overlap_matrix[i,j] < 0.5 else "black",
                     fontsize=9)
    cb = fig4.colorbar(im, ax=ax4, shrink=0.85, aspect=20)
    cb.set_label(f"top-{k_nn} NN overlap", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig4.tight_layout()
    fig4.savefig(FIGS / "nn_agreement.pdf", bbox_inches="tight", pad_inches=0.02)
    fig4.savefig(FIGS / "nn_agreement.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig4)
    print(f"  Saved: nn_agreement.pdf/png")

    np.savez(DATA_OUT / "umap_fos_coords.npz",
             nomic_coords=all_coords["Nomic ModernBERT"],
             granite_coords=all_coords["Granite R2"],
             sciembed_coords=all_coords["SciEmbed-FULL (ours)"],
             labels=np.array(labels),
             corpus_ids=np.array(corpus_ids),
             citation_counts=citation_counts)
    print(f"  Saved: output/qualitative/umap_fos_coords.npz")


# Subcommand 3: Citation context examples from the datalake

def cmd_citation_examples(args: argparse.Namespace) -> None:
    import duckdb

    DATALAKE = "/path/to/data/science_datalake/datalake.duckdb"
    print(f"Connecting to {DATALAKE} ...")
    con = duckdb.connect(DATALAKE, read_only=True)

    # Strategy: find citations where the context is long and informative,
    # the cited paper has a clear title+abstract, and the cited title alone
    # would NOT obviously match the context (low title-word overlap with context).
    # We pull a candidate pool and filter in Python.
    print("Sampling candidate citations with non-trivial context sentences ...")
    q = """
    WITH cand AS (
      SELECT citingcorpusid, citedcorpusid, contexts, intents
      FROM (
        SELECT citingcorpusid, citedcorpusid, contexts, intents
        FROM s2ag.citations
        WHERE contexts IS NOT NULL
          AND len(contexts) > 0
          AND isinfluential = TRUE
      ) USING SAMPLE 200000
    )
    SELECT cand.citingcorpusid, cand.citedcorpusid,
           cand.contexts[1] AS context,
           cand.intents,
           pcited.title AS cited_title,
           pciting.title AS citing_title,
           acited.abstract AS cited_abstract
    FROM cand
    JOIN s2ag.papers pcited   ON pcited.corpusid   = cand.citedcorpusid
    JOIN s2ag.papers pciting  ON pciting.corpusid  = cand.citingcorpusid
    JOIN s2ag.abstracts acited ON acited.corpusid  = cand.citedcorpusid
    WHERE pcited.title IS NOT NULL
      AND pciting.title IS NOT NULL
      AND acited.abstract IS NOT NULL
      AND length(pcited.title) BETWEEN 30 AND 200
      AND length(acited.abstract) BETWEEN 200 AND 1500
      AND length(cand.contexts[1]) BETWEEN 80 AND 350
    LIMIT 2000
    """
    rows = con.execute(q).fetchdf()
    print(f"  Got {len(rows)} candidate rows")

    def title_overlap(context: str, title: str) -> float:
        ctx_words = {w.lower().strip(".,;:()[]") for w in context.split() if len(w) > 3}
        title_words = {w.lower().strip(".,;:()[]") for w in title.split() if len(w) > 3}
        if not title_words:
            return 0.0
        return len(ctx_words & title_words) / len(title_words)

    # Score: prefer LOW title-word overlap (context describes the paper differently)
    rows["overlap"] = rows.apply(
        lambda r: title_overlap(r["context"], r["cited_title"]), axis=1)
    rows = rows.sort_values("overlap").reset_index(drop=True)

    # Pick a diverse top-K with low overlap
    selected = rows.head(20)
    print("\n=== Top low-overlap citation context examples ===\n")
    examples = []
    for i, r in selected.iterrows():
        intent_str = "—"
        try:
            intents = r["intents"]
            flat = []
            if intents is not None:
                for sub in (intents if hasattr(intents, "__iter__") else [intents]):
                    if sub is None:
                        continue
                    if isinstance(sub, str):
                        flat.append(sub)
                        continue
                    if not hasattr(sub, "__iter__"):
                        continue
                    for x in sub:
                        if x:
                            flat.append(str(x))
            if flat:
                intent_str = ",".join(sorted(set(flat)))
        except Exception:
            pass
        ex = {
            "citing_title": r["citing_title"],
            "cited_title": r["cited_title"],
            "context": r["context"],
            "cited_abstract": r["cited_abstract"][:300] + "...",
            "intent": intent_str,
            "title_overlap": round(float(r["overlap"]), 3),
        }
        examples.append(ex)
        if i < 8:
            print(f"--- example {i+1} (overlap={ex['title_overlap']}, intent={intent_str}) ---")
            print(f"  context : {ex['context']}")
            print(f"  cited   : {ex['cited_title']}")
            print()

    (DATA_OUT / "citation_context_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False))
    print(f"\n  Saved: output/qualitative/citation_context_examples.json ({len(examples)} examples)")


# Subcommand 4: Per-task regression-cluster decomposition

REG_TASKS = ["Publication Year", "Max hIndex",
             "Citation Count", "Peer Review Score", "Tweet Mentions"]
SE_FULL_SEEDS = ["sciembed_ctxfull_seed123.json", "sciembed_ctxfull_seed456.json",
                 "sciembed_ctxfull_seed789.json"]
GR_SEEDS = ["granite_r2_english_seed42.json", "granite_r2_english_seed123.json",
            "granite_r2_english_seed456.json"]


def cmd_regression_decomp(args: argparse.Namespace) -> None:
    eval_dir = ROOT / "output" / "official_eval_results"

    def collect(filenames):
        per_task = {t: [] for t in REG_TASKS}
        for f in filenames:
            p = eval_dir / f
            if not p.exists():
                continue
            d = json.load(open(p))
            for t in REG_TASKS:
                v = d.get(t)
                if isinstance(v, dict) and "kendalltau" in v:
                    per_task[t].append(v["kendalltau"])
        return {t: (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0,
                    len(v)) for t, v in per_task.items()}

    se = collect(SE_FULL_SEEDS)
    gr = collect(GR_SEEDS)
    print("SciEmbed-FULL (3-seed)  vs  Granite R2 (3-seed) Kendall tau:")
    for t in REG_TASKS:
        se_m, se_s, _ = se[t]
        gr_m, gr_s, _ = gr[t]
        print(f"  {t:22s} SE={se_m:5.2f}+/-{se_s:.2f}  GR={gr_m:5.2f}+/-{gr_s:.2f}"
              f"  delta={se_m-gr_m:+.2f}")
    mean_delta = np.mean([se[t][0] - gr[t][0] for t in REG_TASKS])
    print(f"  {'mean delta':22s} {mean_delta:+.2f}")

    # Order tasks so the largest positive delta is at the TOP of the
    # horizontal bar chart (matplotlib draws y=0 at the bottom, so we
    # sort ascending and let the y-axis put the biggest delta at the top).
    deltas = [(t, se[t][0] - gr[t][0]) for t in REG_TASKS]
    deltas.sort(key=lambda x: x[1])  # ascending = biggest at top
    ordered_tasks = [t for t, _ in deltas]

    se_means = [se[t][0] for t in ordered_tasks]
    se_stds = [se[t][1] for t in ordered_tasks]
    gr_means = [gr[t][0] for t in ordered_tasks]
    gr_stds = [gr[t][1] for t in ordered_tasks]
    deltas_only = [s - g for s, g in zip(se_means, gr_means)]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    y = np.arange(len(ordered_tasks))
    bar_h = 0.36

    # SciEmbed-FULL bars (darker / our colour)
    ax.barh(y + bar_h/2, se_means, bar_h, xerr=se_stds, capsize=2.5,
            color="#0072B2", edgecolor="white", linewidth=0.4,
            label="SciEmbed-FULL", zorder=2,
            error_kw=dict(ecolor="black", lw=0.6))
    # Granite R2 bars (gray)
    ax.barh(y - bar_h/2, gr_means, bar_h, xerr=gr_stds, capsize=2.5,
            color="#999999", edgecolor="white", linewidth=0.4,
            label="Granite R2", zorder=2,
            error_kw=dict(ecolor="black", lw=0.6))

    # Annotate per-task delta
    xmax = max(max(se_means), max(gr_means)) + 1
    for i, d in enumerate(deltas_only):
        ax.text(xmax + 4.0, i, f"{d:+.1f}", va="center",
                ha="right", fontsize=8.5,
                color="#0072B2" if d > 0 else "#D55E00",
                fontweight="bold")
    ax.text(xmax + 4.0, len(ordered_tasks) - 0.5, "$\\Delta$ (SE-GR)",
            ha="right", fontsize=8, color="black", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(ordered_tasks, fontsize=8.5)
    ax.set_xlabel("Kendall $\\tau$ ($\\times 100$)", fontsize=9)
    ax.set_xlim(0, xmax + 5.5)
    # Legend below the axis so it doesn't overlap bars
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              framealpha=0, fontsize=8.5,
              handlelength=1.2, handletextpad=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title(f"Regression-cluster decomposition  "
                 f"(mean $\\Delta = {mean_delta:+.2f}$)",
                 fontsize=9.5, pad=5)
    fig.tight_layout()
    fig.savefig(FIGS / "regression_decomp.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS / "regression_decomp.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: regression_decomp.pdf/png")


# Subcommand 5: Per-task scatter, SciEmbed-FULL vs Granite R2 (22-task view)

BUCKETS = {
    "Classif.":  ["Biomimicry", "DRSM", "SciDocs MAG", "SciDocs MeSH",
                  "MeSH", "Fields of study"],
    "Regr.":     ["Peer Review Score", "Max hIndex", "Tweet Mentions",
                  "Citation Count", "Publication Year"],
    "Prox.":     ["SciDocs Cite", "SciDocs CoView", "SciDocs CoCite",
                  "SciDocs CoRead", "Same Author Detection",
                  "Highly Influential Citations"],
    "Search":    ["RELISH", "NFCorpus", "TREC-CoVID", "Search"],
}
BUCKET_COLORS = {
    "Classif.": "#E69F00",   # amber
    "Regr.":    "#0072B2",   # blue (our flagship colour)
    "Prox.":    "#009E73",   # green
    "Search":   "#CC79A7",   # rose
}
TASK_METRIC = {
    "Biomimicry": "f1", "DRSM": "f1_macro", "SciDocs MAG": "f1_macro",
    "SciDocs MeSH": "f1_macro", "MeSH": "f1_macro", "Fields of study": "f1_macro",
    "Peer Review Score": "kendalltau", "Max hIndex": "kendalltau",
    "Tweet Mentions": "kendalltau", "Citation Count": "kendalltau",
    "Publication Year": "kendalltau",
    "SciDocs Cite": "map", "SciDocs CoView": "map", "SciDocs CoCite": "map",
    "SciDocs CoRead": "map", "Same Author Detection": "map",
    "Highly Influential Citations": "map",
    "RELISH": "ndcg", "NFCorpus": "ndcg", "TREC-CoVID": "ndcg", "Search": "ndcg",
}
# Shorter labels for figure annotation
TASK_SHORT = {
    "Publication Year": "Pub. Year",
    "Max hIndex": "Max hIndex",
    "Tweet Mentions": "Tweet Mentions",
    "Citation Count": "Cite Count",
    "Peer Review Score": "Peer Review",
    "Highly Influential Citations": "Infl. Cites",
    "Same Author Detection": "Same Author",
    "SciDocs MAG": "MAG",
    "SciDocs MeSH": "MeSH (class.)",
    "SciDocs Cite": "Cite",
    "SciDocs CoView": "CoView",
    "SciDocs CoCite": "CoCite",
    "SciDocs CoRead": "CoRead",
    "TREC-CoVID": "TREC-CoVID",
    "Fields of study": "FoS",
}


def _extract_metric(d: dict, task: str, metric: str) -> float | None:
    v = d.get(task)
    if v is None:
        return None
    if isinstance(v, dict) and "complete" in v:
        return v["complete"].get(metric)
    if isinstance(v, dict):
        return v.get(metric)
    return None


def _collect_per_task(filenames: list[str]) -> dict[str, float]:
    eval_dir = ROOT / "output" / "official_eval_results"
    per_task = {t: [] for t in TASK_METRIC}
    for f in filenames:
        p = eval_dir / f
        if not p.exists():
            continue
        d = json.load(open(p))
        for t, m in TASK_METRIC.items():
            v = _extract_metric(d, t, m)
            if v is not None:
                per_task[t].append(v)
    return {t: float(np.mean(v)) for t, v in per_task.items() if v}


def cmd_task_scatter(args: argparse.Namespace) -> None:
    se = _collect_per_task(SE_FULL_SEEDS)
    gr = _collect_per_task(GR_SEEDS)

    common = [t for t in TASK_METRIC if t in se and t in gr]
    print(f"Plotting {len(common)} tasks across 4 buckets, one subplot per metric")

    # Each bucket has its own metric, so each subplot has its own axis scale.
    # This avoids the cross-metric (Kendall vs NDCG vs MAP vs F1) issue of a
    # single-panel scatter.
    bucket_metric_title = {
        "Classif.": ("Classification", r"Macro F1 ($\times 100$)"),
        "Regr.":    ("Regression",     r"Kendall $\tau$ ($\times 100$)"),
        "Prox.":    ("Proximity",      r"MAP ($\times 100$)"),
        "Search":   ("Search",         r"NDCG ($\times 100$)"),
    }
    # Manual per-task label offsets to keep things readable.
    # (dx, dy) in axis units; positive dx = right, positive dy = up
    LABEL_OFFSETS = {
        # Classification: cluster sits in upper-right (75-90); spread labels
        "Biomimicry":            (1.5, -2.2),
        "DRSM":                  (1.5, 1.5),
        "MeSH":                  (-9.0, 0.8),
        "SciDocs MAG":           (-9.0, -1.8),
        "SciDocs MeSH":          (1.5, 1.5),
        "Fields of study":       (1.5, 0.5),
        # Regression
        "Publication Year":      (1.5, 1.5),
        "Max hIndex":            (1.5, -1.2),
        "Tweet Mentions":        (1.5, -1.2),
        "Citation Count":        (1.5, -1.5),
        "Peer Review Score":     (1.5, 1.5),
        # Proximity: cluster sits upper-right
        "SciDocs Cite":          (-9.5, -1.5),
        "SciDocs CoView":        (-9.5, 1.0),
        "SciDocs CoCite":        (1.0, 1.2),
        "SciDocs CoRead":        (1.0, -1.5),
        "Same Author Detection": (-12.0, 0.5),
        "Highly Influential Citations": (1.0, 0.3),
        # Search
        "RELISH":                (1.5, 1.0),
        "NFCorpus":              (-8.0, 1.0),
        "TREC-CoVID":            (-9.0, -1.5),
        "Search":                (1.5, -1.5),
    }

    # figure* width in ACL ≈ 6.75in textwidth; build at 6.7 × 5.6
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.6))
    axes = axes.flatten()

    bucket_keys = ["Classif.", "Regr.", "Prox.", "Search"]
    for ax, bucket in zip(axes, bucket_keys):
        color = BUCKET_COLORS[bucket]
        title, metric_label = bucket_metric_title[bucket]
        tasks_in_bucket = [t for t in BUCKETS[bucket] if t in common]
        xs = [gr[t] for t in tasks_in_bucket]
        ys = [se[t] for t in tasks_in_bucket]
        ax.scatter(xs, ys, s=55, color=color, edgecolor="white",
                   linewidth=0.7, zorder=3)
        # Per-panel axis range with pad — proximity needs extra left pad for "Same Author" label
        vals = xs + ys
        lo, hi = min(vals) - 6, max(vals) + 5
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="#888888", zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        # Annotate every task in this bucket
        for t, x, y in zip(tasks_in_bucket, xs, ys):
            dx, dy = LABEL_OFFSETS.get(t, (1.0, 0.6))
            ax.annotate(TASK_SHORT.get(t, t),
                        xy=(x, y), xytext=(x + dx, y + dy),
                        fontsize=8.5, color="#222222",
                        arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color="#888888",
                                        shrinkA=3, shrinkB=2))
        # Subplot title and axis labels
        ax.set_title(f"{title}  ({metric_label})", fontsize=10.5, pad=5,
                     fontweight="bold", color=color)
        ax.set_xlabel("Granite R2 score", fontsize=9)
        ax.set_ylabel("SciEmbed-FULL score", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Hint annotations in corners
        pad = (hi - lo) * 0.06
        ax.text(lo + pad, hi - pad, "SciEmbed leads", fontsize=8,
                color=color, style="italic", ha="left", va="top",
                fontweight="bold", alpha=0.85)
        ax.text(hi - pad, lo + pad, "Granite leads", fontsize=8,
                color="#666666", style="italic", ha="right", va="bottom",
                alpha=0.85)

    fig.tight_layout(pad=0.8, h_pad=2.0, w_pad=2.0)
    out_pdf = FIGS / "task_scatter_sciembed_vs_granite.pdf"
    out_png = FIGS / "task_scatter_sciembed_vs_granite.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: {out_pdf.name} / {out_png.name}")

    # Also dump the numeric data the figure was built from (reproducibility)
    out_json = DATA_OUT / "task_scatter_data.json"
    with open(out_json, "w") as f:
        json.dump({
            "sciembed_full_3seed_mean": se,
            "granite_r2_3seed_mean": gr,
            "buckets": BUCKETS,
            "task_metric": TASK_METRIC,
        }, f, indent=2)
    print(f"  Saved: {out_json.relative_to(ROOT)}")



def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("doc-length", help="Document length histogram")
    dl.add_argument("--refresh", action="store_true",
                    help="Re-tokenize and overwrite cached length arrays")
    sub.add_parser("umap", help="UMAP scatter of FoS embeddings")
    sub.add_parser("citation-examples", help="Citation context examples")
    sub.add_parser("regression-decomp", help="Per-task regression delta chart")
    sub.add_parser("task-scatter",
                   help="Per-task scatter, SciEmbed-FULL vs Granite R2")

    args = p.parse_args()
    if args.cmd == "doc-length":
        cmd_doc_length(args)
    elif args.cmd == "umap":
        cmd_umap(args)
    elif args.cmd == "citation-examples":
        cmd_citation_examples(args)
    elif args.cmd == "regression-decomp":
        cmd_regression_decomp(args)
    elif args.cmd == "task-scatter":
        cmd_task_scatter(args)


if __name__ == "__main__":
    main()
