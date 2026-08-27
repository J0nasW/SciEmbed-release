"""Generate analysis figures for the SciEmbed paper.

Figures produced (paper/latex/figures/generated/):
  1. per_task_delta.{pdf,png}    — per-task delta of SciEmbed-FULL vs BGE-large.
  2. inference_throughput.{pdf,png} — H100 fp16 throughput vs batch size.
  3. matryoshka_throughput.{pdf,png} — throughput at Matryoshka dims (H100 fp16).

All inputs are read from output/eval_results/scirepeval/ and
output/inference_benchmark/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "output" / "eval_results" / "scirepeval"
BENCH = ROOT / "output" / "inference_benchmark"
OUT = ROOT / "paper" / "latex" / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

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


# Per-task primary metric extractor (matches the paper's Table 7)

# (display_name, json_key, metric_path, category)
TASKS: list[tuple[str, str, tuple, str]] = [
    # Classification (F1)
    ("MeSH",       "MeSH",            ("f1_macro",),               "Classif."),
    ("FoS",        "Fields of study", ("complete", "f1_macro"),    "Classif."),
    ("Biomim.",    "Biomimicry",      ("complete", "f1"),          "Classif."),
    ("DRSM",       "DRSM",            ("complete", "f1_macro"),    "Classif."),
    ("MAG",        "SciDocs MAG",     ("f1_macro",),               "Classif."),
    # Regression (Kendall tau)
    ("Cite cnt",   "Citation Count",  ("kendalltau",),             "Regr."),
    ("Pub year",   "Publication Year",("kendalltau",),             "Regr."),
    ("hIndex",     "Max hIndex",      ("kendalltau",),             "Regr."),
    ("Tweets",     "Tweet Mentions",  ("kendalltau",),             "Regr."),
    ("Review",     "Peer Review Score",("kendalltau",),            "Regr."),
    # Proximity (MAP)
    ("Cite",       "SciDocs Cite",    ("map",),                    "Prox."),
    ("Co-cite",    "SciDocs CoCite",  ("map",),                    "Prox."),
    ("Co-view",    "SciDocs CoView",  ("map",),                    "Prox."),
    ("Co-read",    "SciDocs CoRead",  ("map",),                    "Prox."),
    ("Hi-infl.",   "Highly Influential Citations", ("map",),       "Prox."),
    ("Author",     "Same Author Detection", ("map",),              "Prox."),
    # Search (NDCG)
    ("Search",     "Search",          ("ndcg",),                   "Search"),
    ("TREC-C",     "TREC-CoVID",      ("ndcg",),                   "Search"),
    ("NFC",        "NFCorpus",        ("ndcg",),                   "Search"),
    ("RELISH",     "RELISH",          ("ndcg",),                   "Search"),
]

CAT_COLOR = {
    "Classif.": "#2563EB",
    "Regr.":    "#059669",
    "Prox.":    "#D97706",
    "Search":   "#7C3AED",
}


def get_metric(d: dict, key: str, path: tuple) -> float | None:
    if key not in d:
        return None
    node = d[key]
    for p in path:
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return float(node)


def load_per_task(path: Path) -> dict[str, float]:
    d = json.load(open(path))
    out = {}
    for display, key, mpath, _cat in TASKS:
        v = get_metric(d, key, mpath)
        if v is not None:
            out[display] = v
    return out


# Figure 1: per-task delta vs BGE-large

def fig_per_task_delta() -> None:
    bge = load_per_task(EVAL / "baselines" / "bge-large-en-v1.5.json")
    full = load_per_task(EVAL / "sciembed" / "ctx_full.json")

    rows = []
    for display, _key, _mpath, cat in TASKS:
        if display in bge and display in full:
            delta = full[display] - bge[display]
            rows.append((display, delta, cat))

    rows.sort(key=lambda r: r[1])

    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    colors = [CAT_COLOR[r[2]] if r[1] >= 0 else "#9CA3AF" for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    y = np.arange(len(rows))
    ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score delta vs.\\ BGE-large-en-v1.5 (SciEmbed-FULL $-$ BGE-large)")
    ax.set_xlim(-7, 8)

    # Annotate values
    for i, d in enumerate(deltas):
        ha = "left" if d >= 0 else "right"
        off = 0.15 if d >= 0 else -0.15
        ax.text(d + off, i, f"{d:+.1f}", va="center", ha=ha, fontsize=6.5,
                color="#111827" if d >= 0 else "#6B7280")

    # Category legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COLOR.values()]
    ax.legend(handles, list(CAT_COLOR.keys()), loc="lower right", framealpha=0.9,
              edgecolor="none", ncol=4, columnspacing=0.8, handlelength=1.0,
              handleheight=0.8)

    fig.savefig(OUT / "per_task_delta.pdf")
    fig.savefig(OUT / "per_task_delta.png")
    plt.close(fig)
    wins = sum(1 for d in deltas if d > 0)
    print(f"  per_task_delta: {wins}/{len(deltas)} tasks improved over BGE-large")


# Figure 2: inference throughput vs batch size (H100 fp16)

def fig_inference_throughput() -> None:
    d = json.load(open(BENCH / "h100_fp16.json"))

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    markers = {"SciEmbed (149M)": ("o", "#DC2626"),
               "BGE-large-en-v1.5 (335M)": ("s", "#2563EB"),
               "SPECTER2 Base (110M)": ("^", "#059669")}
    label_short = {"SciEmbed (149M)": "SciEmbed (149M)",
                   "BGE-large-en-v1.5 (335M)": "BGE-large (335M)",
                   "SPECTER2 Base (110M)": "SPECTER2 (110M)"}

    for model, info in d.items():
        if model not in markers:
            continue
        bs = sorted(int(k) for k in info["results"].keys())
        thr = [info["results"][str(b)]["throughput_docs_sec"] for b in bs]
        m, c = markers[model]
        ax.plot(bs, thr, marker=m, color=c, linewidth=1.2, markersize=4.5,
                label=label_short[model])

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 64, 128])
    ax.set_xticklabels(["1", "8", "32", "64", "128"])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (docs/sec)")
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="none")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)

    fig.savefig(OUT / "inference_throughput.pdf")
    fig.savefig(OUT / "inference_throughput.png")
    plt.close(fig)
    print("  inference_throughput: H100 fp16, 3 models")


# Figure 3: matryoshka throughput

def fig_matryoshka_throughput() -> None:
    d = json.load(open(BENCH / "h100_fp16.json"))
    matr = d["SciEmbed (149M)"]["matryoshka"]
    dims = sorted(int(k) for k in matr.keys())
    thr = [matr[str(x)]["throughput_docs_sec"] for x in dims]

    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    bars = ax.bar([str(x) for x in dims], thr,
                  color=["#94A3B8", "#64748B", "#475569", "#1E293B"],
                  edgecolor="white", linewidth=0.6, width=0.62)
    for b, v in zip(bars, thr):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, f"{v:.0f}",
                ha="center", fontsize=7, color="#111827")
    ax.set_xlabel("Matryoshka dimension")
    ax.set_ylabel("Throughput (docs/sec)")
    ax.set_ylim(top=max(thr) * 1.13)
    fig.savefig(OUT / "matryoshka_throughput.pdf")
    fig.savefig(OUT / "matryoshka_throughput.png")
    plt.close(fig)
    print(f"  matryoshka_throughput: {dims} → {thr}")


# Compute and print analysis numbers (used in error analysis prose)

def print_error_analysis() -> None:
    full = load_per_task(EVAL / "sciembed" / "ctx_full.json")
    bge  = load_per_task(EVAL / "baselines" / "bge-large-en-v1.5.json")
    spec = load_per_task(EVAL / "baselines" / "specter2-base.json")
    e5   = load_per_task(EVAL / "baselines" / "e5-large-v2.json")
    nomic= load_per_task(EVAL / "baselines" / "nomic-modernbert-embed-base.json")

    print("\n=== Per-task scores (FULL vs best baseline) ===")
    print(f"{'Task':<10} {'FULL':>6} {'BGE-l':>6} {'SPEC2':>6} {'Nomic':>6} {'E5-l':>6} {'best base':>10}  Δ(FULL-best)")
    losses = []
    for name in [t[0] for t in TASKS]:
        if name not in full:
            continue
        baselines = {"BGE-l": bge.get(name), "SPEC2": spec.get(name),
                     "Nomic": nomic.get(name), "E5-l": e5.get(name)}
        valid = {k: v for k, v in baselines.items() if v is not None}
        if not valid:
            continue
        best_name = max(valid, key=valid.get)
        best = valid[best_name]
        delta = full[name] - best
        marker = "←" if delta < -1.0 else ""
        print(f"{name:<10} {full[name]:>6.1f} {bge.get(name, float('nan')):>6.1f} "
              f"{spec.get(name, float('nan')):>6.1f} {nomic.get(name, float('nan')):>6.1f} "
              f"{e5.get(name, float('nan')):>6.1f} {best_name+f' {best:.1f}':>10}  {delta:+.1f} {marker}")
        if delta < 0:
            losses.append((name, delta, best_name, best))

    losses.sort(key=lambda x: x[1])
    print("\nLargest losses vs best baseline:")
    for name, d, src, val in losses[:5]:
        print(f"  {name}: Δ={d:+.1f} (best={src}={val:.1f})")


def main():
    print(f"Generating analysis figures → {OUT}/\n")
    fig_per_task_delta()
    fig_inference_throughput()
    fig_matryoshka_throughput()
    print_error_analysis()
    print("\nDone.")


if __name__ == "__main__":
    main()
