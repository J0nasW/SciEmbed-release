"""Generate publication-ready figures for the SciEmbed paper.

Reads Stage 1 MLM TensorBoard logs and produces PDF/PNG figures
suitable for LaTeX inclusion.

Usage:
    python scripts/generate_paper_figures.py [--output-dir paper/latex/figures/generated]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Style configuration — publication-quality defaults

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Color palette (colorblind-friendly)
C_TRAIN = "#2563EB"      # blue
C_EVAL = "#DC2626"       # red
C_LR = "#7C3AED"         # purple
C_THROUGHPUT = "#059669"  # green
C_MEMORY = "#D97706"     # amber

# The real run was on 4×H200; earlier single-GPU false starts left artifacts in
# the same TensorBoard logdir. Split them out by throughput.
_THROUGHPUT_4GPU_MIN = 100_000  # tok/s — below this is a 1-GPU artifact


def load_tensorboard(logdir: str) -> EventAccumulator:
    """Load and return a TensorBoard EventAccumulator."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    return ea


def get_scalars(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract (steps, values) arrays for a given scalar tag."""
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def _find_4gpu_start(ea: EventAccumulator) -> int:
    """Find the step index where 4-GPU throughput begins (filters out 1-GPU artifacts)."""
    steps, tp = get_scalars(ea, "throughput/tokens_per_sec")
    mask = tp > _THROUGHPUT_4GPU_MIN
    if mask.any():
        return int(steps[mask][0])
    return 0


def _tokens_b(steps: np.ndarray) -> np.ndarray:
    """Convert step numbers to billions of tokens.

    Global batch = per_device_batch (72) × n_gpus (4) = 288 sequences per optimizer step.
    Seq len = 8192. So tokens/step = 288 × 8192 ≈ 2.36M. At 3500 steps this yields
    ~8.3B tokens, matching the throughput × wall-time integration (143.6K tok/s × 16.1h).
    """
    return steps * 72 * 4 * 8192 / 1e9


# Figure 1: Training & Eval Loss (main figure for paper)

def fig_loss_curves(ea: EventAccumulator, out: Path) -> None:
    """Train loss (smoothed) + eval loss — the main paper figure."""
    train_steps, train_loss = get_scalars(ea, "loss/train/total")
    eval_steps, eval_loss = get_scalars(ea, "metrics/eval/LanguageCrossEntropy")

    train_b = _tokens_b(train_steps)
    eval_b = _tokens_b(eval_steps)
    train_smooth = smooth(train_loss, weight=0.95)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    # Raw train loss (faint background)
    ax.plot(train_b, train_loss, color=C_TRAIN, alpha=0.10, linewidth=0.4)
    # Smoothed train loss
    ax.plot(train_b, train_smooth, color=C_TRAIN, linewidth=1.4, label="Train (smoothed)")
    # Eval loss
    ax.plot(eval_b, eval_loss, color=C_EVAL, linewidth=1.4,
            marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6,
            label="Eval", zorder=5)

    # Annotate first and last eval — offset away from data
    ax.annotate(f"{eval_loss[0]:.3f}", (eval_b[0], eval_loss[0]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=7, color=C_EVAL, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_EVAL, lw=0.5, shrinkA=3, shrinkB=0))
    ax.annotate(f"{eval_loss[-1]:.3f}", (eval_b[-1], eval_loss[-1]),
                textcoords="offset points", xytext=(-36, -14),
                fontsize=7, color=C_EVAL, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_EVAL, lw=0.5, shrinkA=3, shrinkB=0))

    ax.set_xlabel("Tokens (billions)")
    ax.set_ylabel("MLM Loss (cross-entropy)")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=1.02, top=1.16)

    fig.savefig(out / "loss_curves.pdf")
    fig.savefig(out / "loss_curves.png")
    plt.close(fig)
    print(f"  Saved: loss_curves.pdf/png")


# Figure 2: Learning Rate Schedule

def fig_lr_schedule(ea: EventAccumulator, out: Path) -> None:
    """Learning rate schedule over training."""
    steps, lr = get_scalars(ea, "lr-AdamW/group0")
    tokens_b = _tokens_b(steps)

    fig, ax = plt.subplots(figsize=(3.4, 1.8))

    ax.plot(tokens_b, lr * 1e5, color=C_LR, linewidth=1.4)
    ax.set_xlabel("Tokens (billions)")
    ax.set_ylabel(r"Learning rate ($\times 10^{-5}$)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=2.4)

    # Mark peak — place label above and to the right, clear of curve
    peak_idx = np.argmax(lr)
    ax.annotate(f"peak = {lr[peak_idx]:.0e}",
                (tokens_b[peak_idx], lr[peak_idx] * 1e5),
                textcoords="offset points", xytext=(30, 4),
                fontsize=7, color=C_LR, fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color=C_LR, lw=0.6))

    fig.savefig(out / "lr_schedule.pdf")
    fig.savefig(out / "lr_schedule.png")
    plt.close(fig)
    print(f"  Saved: lr_schedule.pdf/png")


# Figure 3: Combined training dashboard (loss + LR + throughput + memory)

def fig_training_dashboard(ea: EventAccumulator, out: Path) -> None:
    """4-panel training dashboard: loss, LR, throughput, GPU memory.

    Throughput and memory are filtered to the 4-GPU phase to remove
    artifacts from earlier 1-GPU runs on a different node.
    """
    gpu_start = _find_4gpu_start(ea)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))

    # --- Panel (a): Loss ---
    ax = axes[0, 0]
    train_steps, train_loss = get_scalars(ea, "loss/train/total")
    eval_steps, eval_loss = get_scalars(ea, "metrics/eval/LanguageCrossEntropy")

    ax.plot(_tokens_b(train_steps), smooth(train_loss, 0.95), color=C_TRAIN, linewidth=1.2, label="Train")
    ax.plot(_tokens_b(train_steps), train_loss, color=C_TRAIN, alpha=0.08, linewidth=0.4)
    ax.plot(_tokens_b(eval_steps), eval_loss, color=C_EVAL, linewidth=1.2,
            marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.4,
            label="Eval")
    ax.set_ylabel("MLM Loss")
    ax.set_ylim(bottom=1.02, top=1.16)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_title("(a) Training & Eval Loss", fontsize=9, fontweight="bold", loc="left")
    ax.set_xlim(left=0)

    # --- Panel (b): Learning Rate ---
    ax = axes[0, 1]
    lr_steps, lr = get_scalars(ea, "lr-AdamW/group0")
    ax.plot(_tokens_b(lr_steps), lr * 1e5, color=C_LR, linewidth=1.2)
    ax.set_ylabel(r"LR ($\times 10^{-5}$)")
    ax.set_title("(b) Learning Rate Schedule", fontsize=9, fontweight="bold", loc="left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=2.4)

    # --- Panel (c): Throughput (4-GPU phase only, value-filtered) ---
    ax = axes[1, 0]
    tp_steps, tp_tokens = get_scalars(ea, "throughput/tokens_per_sec")
    # Value-based filter: only 4-GPU data points (>100K tok/s)
    tp_mask = tp_tokens > _THROUGHPUT_4GPU_MIN
    tp_b = _tokens_b(tp_steps[tp_mask])
    tp_vals = tp_tokens[tp_mask] / 1e3

    ax.plot(tp_b, tp_vals, color=C_THROUGHPUT, linewidth=0.8, alpha=0.5)

    # Median band
    median_tp = np.median(tp_vals)
    ax.axhline(median_tp, color=C_THROUGHPUT, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.text(0.03, 0.92, f"median: {median_tp:.0f}K tok/s",
            transform=ax.transAxes, fontsize=7, color=C_THROUGHPUT, fontstyle="italic")

    ax.set_xlabel("Tokens (billions)")
    ax.set_ylabel("Throughput (K tok/s)")
    ax.set_title("(c) Training Throughput (4$\\times$H200)", fontsize=9, fontweight="bold", loc="left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=130, top=155)

    # --- Panel (d): GPU Memory (value-filtered to 4-GPU phase) ---
    ax = axes[1, 1]
    mem_steps, peak_mem = get_scalars(ea, "memory/peak_active_mem")
    _, reserved = get_scalars(ea, "memory/peak_reserved_mem")
    # Value-based filter: 4-GPU peak memory is >80 GB
    mem_mask = peak_mem > 80
    mem_b = _tokens_b(mem_steps[mem_mask])

    ax.plot(mem_b, peak_mem[mem_mask], color=C_MEMORY, linewidth=1.2, label="Peak active")
    ax.plot(mem_b, reserved[mem_mask], color=C_MEMORY, linewidth=1.0, linestyle="--",
            alpha=0.6, label="Reserved")
    ax.axhline(y=141, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.55, 0.95, "H200 capacity (141 GB)",
            transform=ax.transAxes, fontsize=6, color="gray", fontstyle="italic")
    ax.set_xlabel("Tokens (billions)")
    ax.set_ylabel("GPU Memory (GB)")
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none")
    ax.set_title("(d) GPU Memory Usage", fontsize=9, fontweight="bold", loc="left")
    ax.set_xlim(left=0)
    ax.set_ylim(top=155)

    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    fig.savefig(out / "training_dashboard.pdf")
    fig.savefig(out / "training_dashboard.png")
    plt.close(fig)
    print(f"  Saved: training_dashboard.pdf/png")


# Figure 4: Eval loss bar chart (before vs after)

def fig_eval_progression(ea: EventAccumulator, out: Path) -> None:
    """Bar chart showing eval loss at each checkpoint."""
    eval_steps, eval_loss = get_scalars(ea, "metrics/eval/LanguageCrossEntropy")
    eval_btokens = _tokens_b(eval_steps)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    labels = [f"{b:.1f}B" for b in eval_btokens]
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(eval_loss)))

    bars = ax.bar(labels, eval_loss, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    # Value labels on bars
    for bar, val in zip(bars, eval_loss):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xlabel("Tokens processed")
    ax.set_ylabel("Eval Loss")
    ax.set_ylim(bottom=min(eval_loss) - 0.008, top=max(eval_loss) + 0.010)

    # Reduction annotation — text in upper-right, no crossing arrow
    pct_reduction = (eval_loss[0] - eval_loss[-1]) / eval_loss[0] * 100
    ax.text(0.97, 0.92, f"$\\Delta$ = −{pct_reduction:.1f}%",
            transform=ax.transAxes, fontsize=9, fontweight="bold", color=C_EVAL,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_EVAL, alpha=0.9, linewidth=0.8))

    fig.savefig(out / "eval_progression.pdf")
    fig.savefig(out / "eval_progression.png")
    plt.close(fig)
    print(f"  Saved: eval_progression.pdf/png")


# Figure 5: Throughput distribution (histogram)

def fig_training_summary(ea: EventAccumulator, out: Path) -> None:
    """Visual summary card with key training statistics."""
    train_steps, train_loss = get_scalars(ea, "loss/train/total")
    eval_steps, eval_loss = get_scalars(ea, "metrics/eval/LanguageCrossEntropy")
    _, lr = get_scalars(ea, "lr-AdamW/group0")
    _, tp = get_scalars(ea, "throughput/tokens_per_sec")
    _, peak_mem = get_scalars(ea, "memory/peak_active_mem")
    _, total_time = get_scalars(ea, "time/total")
    _, tokens = get_scalars(ea, "time/token")

    tp_4gpu = tp[tp > _THROUGHPUT_4GPU_MIN]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.axis("off")

    stats = [
        ("Model", "ModernBERT-base (149M)"),
        ("Hardware", "4×NVIDIA H200 (141 GB)"),
        ("Tokens processed", f"{tokens[-1] / 1e9:.1f}B"),
        ("Wall-clock time", f"{total_time[-1]:.1f} hours"),
        ("Throughput", f"{np.median(tp_4gpu) / 1e3:.1f}K tok/s"),
        ("Peak GPU memory", f"{peak_mem.max():.0f} GB / 141 GB"),
        ("Eval loss", f"{eval_loss[0]:.3f} → {eval_loss[-1]:.3f} (−{(1 - eval_loss[-1] / eval_loss[0]) * 100:.1f}%)"),
        ("Peak LR", f"{lr.max():.0e} (cosine decay)"),
    ]

    y_start = 0.95
    for i, (label, value) in enumerate(stats):
        y = y_start - i * 0.115
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", color="#374151")
        ax.text(0.50, y, value, transform=ax.transAxes, fontsize=8,
                va="top", color="#1F2937")

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#E5E7EB")
        spine.set_linewidth(0.8)

    ax.set_title("Stage 1 MLM Training Summary", fontsize=10, fontweight="bold",
                 loc="left", pad=8)

    fig.savefig(out / "training_summary_card.pdf")
    fig.savefig(out / "training_summary_card.png")
    plt.close(fig)
    print(f"  Saved: training_summary_card.pdf/png")


# Figure 6: Data composition (for methods section)

def fig_data_composition(out: Path) -> None:
    """Donut charts for Stage 1 corpus and Stage 2 contrastive signals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Stage 1: MLM corpus
    s1_labels = ["S2ORC (7.4M)", "PMC OA (4.6M)", "arXiv+peS2o (1.4M)"]
    s1_sizes = [7.4, 4.6, 1.4]
    s1_colors = ["#3B82F6", "#6366F1", "#8B5CF6"]

    wedges1, texts1, autotexts1 = ax1.pie(
        s1_sizes, labels=s1_labels, colors=s1_colors, autopct="%1.0f%%",
        pctdistance=0.75, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=8),
    )
    for t in autotexts1:
        t.set_fontsize(7)
        t.set_fontweight("bold")
        t.set_color("white")
    ax1.set_title("Stage 1: MLM Corpus\n(13.4M papers, ~80B tokens)", fontsize=9, fontweight="bold")

    # Stage 2: Contrastive signals — use legend for cleaner layout
    s2_labels = ["Context queries (5M)", "Intent pairs (5M)", "Citation edges (2M)",
                 "Instruction pairs (2M)", "Section pairs (1M)"]
    s2_sizes = [5, 5, 2, 2, 1]
    s2_colors = ["#8B5CF6", "#A855F7", "#3B82F6", "#6B7280", "#14B8A6"]

    wedges2, texts2, autotexts2 = ax2.pie(
        s2_sizes, colors=s2_colors, autopct="%1.0f%%",
        pctdistance=0.75, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=8),
    )
    for t in autotexts2:
        t.set_fontsize(7)
        t.set_fontweight("bold")
        t.set_color("white")

    ax2.legend(wedges2, s2_labels, loc="center left", bbox_to_anchor=(0.92, 0.5),
               fontsize=7, frameon=False)
    ax2.set_title("Stage 2: Contrastive Mix\n(15M pairs)", fontsize=9, fontweight="bold")

    fig.tight_layout(w_pad=0.5)
    fig.savefig(out / "data_composition.pdf")
    fig.savefig(out / "data_composition.png")
    plt.close(fig)
    print(f"  Saved: data_composition.pdf/png")


# Figure 7: Loss comparison context (vs literature)

def fig_loss_comparison(ea: EventAccumulator, out: Path) -> None:
    """Bar chart comparing domain-adaptive MLM loss reduction across papers."""
    eval_steps, eval_loss = get_scalars(ea, "metrics/eval/LanguageCrossEntropy")
    our_before, our_after = float(eval_loss[0]), float(eval_loss[-1])

    papers = [
        ("DAPT\n(BioMed)", 1.32, 0.99),
        ("DAPT\n(CS)", 1.63, 1.34),
        ("DAPT\n(Reviews)", 2.10, 1.93),
        ("SciEmbed\n(Ours)", our_before, our_after),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    x = np.arange(len(papers))
    width = 0.28

    befores = [p[1] for p in papers]
    afters = [p[2] for p in papers]
    labels = [p[0] for p in papers]
    reductions = [(b - a) / b * 100 for b, a in zip(befores, afters)]

    ax.bar(x - width / 2, befores, width, color="#93C5FD", edgecolor="white",
           linewidth=0.5, label="Before DAPT")
    ax.bar(x + width / 2, afters, width, color="#2563EB", edgecolor="white",
           linewidth=0.5, label="After DAPT")

    # Reduction labels — enough headroom above tallest bar
    max_val = max(befores)
    for i, (b, a, r) in enumerate(zip(befores, afters, reductions)):
        ax.text(i, max(b, a) + max_val * 0.04, f"−{r:.1f}%", ha="center", fontsize=7,
                fontweight="bold", color="#DC2626")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Eval Loss (MLM)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")
    ax.set_ylim(bottom=0, top=max_val * 1.18)

    # Note about comparability — well below x-axis labels
    fig.text(0.98, -0.06, "* DAPT uses 15% mask rate; SciEmbed uses 30%",
             fontsize=6, ha="right", fontstyle="italic", color="gray")

    fig.savefig(out / "loss_comparison.pdf")
    fig.savefig(out / "loss_comparison.png")
    plt.close(fig)
    print(f"  Saved: loss_comparison.pdf/png")


# Summary statistics table (printed to console and saved as text)

def print_summary(ea: EventAccumulator, out: Path) -> None:
    """Print and save key training statistics."""
    train_steps, train_loss = get_scalars(ea, "loss/train/total")
    eval_steps, eval_loss = get_scalars(ea, "metrics/eval/LanguageCrossEntropy")
    _, lr = get_scalars(ea, "lr-AdamW/group0")
    _, tp = get_scalars(ea, "throughput/tokens_per_sec")
    _, peak_mem = get_scalars(ea, "memory/peak_active_mem")
    time_steps, total_time = get_scalars(ea, "time/total")
    _, tokens = get_scalars(ea, "time/token")

    # Filter throughput to 4-GPU phase
    tp_4gpu = tp[tp > _THROUGHPUT_4GPU_MIN]

    lines = [
        "=" * 60,
        "SciEmbed Stage 1 MLM — Training Summary",
        "=" * 60,
        f"Total steps:          {int(train_steps[-1])}",
        f"Total tokens:         {tokens[-1] / 1e9:.1f}B",
        f"Wall-clock time:      {total_time[-1]:.1f} hours",
        "",
        "Loss:",
        f"  Train (first 100):  {np.mean(train_loss[:100]):.4f}",
        f"  Train (last 100):   {np.mean(train_loss[-100:]):.4f}",
        f"  Train reduction:    {(1 - np.mean(train_loss[-100:]) / np.mean(train_loss[:100])) * 100:.1f}%",
        f"  Eval start:         {eval_loss[0]:.4f}",
        f"  Eval end:           {eval_loss[-1]:.4f}",
        f"  Eval reduction:     {(1 - eval_loss[-1] / eval_loss[0]) * 100:.1f}%",
        "",
        "Learning rate:",
        f"  Peak:               {lr.max():.2e}",
        f"  Final:              {lr[-1]:.2e}",
        "",
        "Throughput (4×H200, steady state):",
        f"  Mean:               {np.mean(tp_4gpu) / 1e3:.1f}K tok/s",
        f"  Median:             {np.median(tp_4gpu) / 1e3:.1f}K tok/s",
        f"  Std:                {np.std(tp_4gpu) / 1e3:.1f}K tok/s",
        "",
        "GPU memory:",
        f"  Peak active:        {peak_mem.max():.1f} GB",
        "",
        "Eval checkpoints:",
    ]
    for step, loss in zip(eval_steps, eval_loss):
        tok_b = step * 72 * 4 * 8192 / 1e9
        lines.append(f"  step {int(step):>5} ({tok_b:.1f}B tok): {loss:.4f}")

    lines.append("=" * 60)

    text = "\n".join(lines)
    print(text)

    summary_path = out / "training_summary.txt"
    summary_path.write_text(text + "\n")
    print(f"\n  Saved: training_summary.txt")



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from Stage 1 TensorBoard logs.")
    parser.add_argument("--logdir", default="output/stage1_mlm/tensorboard/stage1-mlm-modernbert",
                        help="Path to TensorBoard log directory")
    parser.add_argument("--output-dir", default="paper/latex/figures/generated",
                        help="Output directory for generated figures")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(STYLE)

    print(f"Loading TensorBoard logs from: {args.logdir}")
    ea = load_tensorboard(args.logdir)

    print(f"Generating figures → {out}/\n")

    fig_loss_curves(ea, out)
    fig_lr_schedule(ea, out)
    fig_training_dashboard(ea, out)
    fig_eval_progression(ea, out)
    fig_training_summary(ea, out)
    fig_data_composition(out)
    fig_loss_comparison(ea, out)
    print()
    print_summary(ea, out)

    print(f"\nDone! All figures saved to {out}/")


if __name__ == "__main__":
    main()
