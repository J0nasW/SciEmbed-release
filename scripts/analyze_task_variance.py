"""Per-task SciRepEval variance analysis.

Tests the hypothesis: the 67.1 four-cat ceiling on SciEmbed-FULL is
"recipe-bound" or "task-bound"?

- Recipe-bound: all our 22 sweep variants cluster around 67.1 because
  the contrastive recipe has saturated. Per-task variance across our
  variants should be uniformly small.

- Task-bound: a few tasks are too hard/noisy/saturated. Every model
  (including Granite, BGE, SPECTER2) scores nearly the same on them.
  Per-task variance across DIFFERENT-recipe models should also be small
  on those tasks, but high on others.

Output:
  1. Per-task table: mean, std, range across all loaded models, color-coded.
  2. "Stuck tasks" = low across-model std (<1.0). These are the suspects.
  3. "Discriminating tasks" = high across-model std (>3.0). These reveal supervision.
  4. Sensitivity: recompute 4-cat overall if we drop the bottom-3 stuck tasks
     (rough simulation of removing benchmark artefacts).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.aggregate_singh_check import (
    CLASSIFICATION_TASKS, REGRESSION_TASKS, PROXIMITY_TASKS, SEARCH_TASKS, PRM_TASK,
    PRIMARY_METRIC, extract_primary_score,
)

# Two cohorts:
# - sciembed_variants: 22 of our SciEmbed-* runs (sweep + Phase 3 + Signal D)
# - external_models: matched-arch peer + general-purpose baselines
SCIEMBED_VARIANTS = {
    # 6 scale sweep
    "scale=12": "sciembed_sweep_scale12_seed42.json",
    "scale=14": "sciembed_sweep_scale14_seed42.json",
    "scale=16": "sciembed_sweep_ctx_scale16_seed42.json",
    "scale=18": "sciembed_sweep_scale18_seed42.json",
    "scale=24": "sciembed_sweep_ctx_scale24_seed42.json",
    "scale=30": "sciembed_sweep_ctx_scale30_seed42.json",
    # loss variants
    "sym_mnrl": "sciembed_sweep_sym_mnrl_seed42.json",
    "cached1_b256": "sciembed_sweep_cached1_b256_seed42.json",
    "cached1_b512": "sciembed_sweep_cached1_b512_seed42.json",
    "cached1_b1024": "sciembed_sweep_cached1_b1024_seed42.json",
    # training schedule
    "ep5_lr1e5": "sciembed_sweep_ep5_lr1e5_seed42.json",
    "ep5_lr2e5": "sciembed_sweep_ep5_lr2e5_seed42.json",
    # alpha axes
    "alpha_cls": "sciembed_sweep_alpha_cls_pooling_seed42.json",
    "alpha_lr_3e5": "sciembed_sweep_alpha_lr_3e5_seed42.json",
    "alpha_warmup_010": "sciembed_sweep_alpha_warmup_010_seed42.json",
    "alpha_matr_768": "sciembed_sweep_alpha_matryoshka_768_seed42.json",
    "alpha_matr_768_256": "sciembed_sweep_alpha_matryoshka_768_256_seed42.json",
    "alpha_ep4_lr1e5": "sciembed_sweep_alpha_ep4_lr1e5_seed42.json",
    # FULL recipe + Phase 3
    "FULL seed123": "sciembed_ctxfull_seed123.json",
    "FULL seed456": "sciembed_ctxfull_seed456.json",
    "Phase3 FULL+s18": "sciembed_phase3_full_scale18_seed42.json",
    "Signal D pure": "sciembed_signal_d_pure_seed42.json",
    "A+B+D mixed": "sciembed_signal_abd_mixed_seed42.json",
}
EXTERNAL_MODELS = {
    "Granite R2": "granite_r2_english.json",
    "BGE-large": "bge_large.json",
    "Nomic ModernBERT": "nomic-ai_modernbert-embed-base.json",
    "SPECTER2 Base": "specter2_base.json",
    "SPECTER2 +adapters": "specter2_adapters.json",
}

# All 22 SciRepEval tasks (PRM excluded from primary, handled separately)
ALL_TASKS = sorted(CLASSIFICATION_TASKS | REGRESSION_TASKS | PROXIMITY_TASKS | SEARCH_TASKS)


def task_bucket(t: str) -> str:
    if t in CLASSIFICATION_TASKS: return "cls"
    if t in REGRESSION_TASKS:     return "reg"
    if t in PROXIMITY_TASKS:      return "prox"
    if t in SEARCH_TASKS:         return "search"
    return "?"


def load_scores(results_dir: Path, cohort: dict) -> dict[str, dict[str, float]]:
    """Return {model_label: {task: score}} for everything found."""
    out: dict[str, dict[str, float]] = {}
    for label, fn in cohort.items():
        p = results_dir / fn
        if not p.exists():
            print(f"  (missing {label}: {fn})", file=sys.stderr)
            continue
        with open(p) as f:
            d = json.load(f)
        per_task: dict[str, float] = {}
        for t, td in d.items():
            if not isinstance(td, dict): continue
            if t == PRM_TASK: continue
            s = extract_primary_score(t, td)
            if s is not None:
                per_task[t] = s
        out[label] = per_task
    return out


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "output" / "official_eval_results"
    sciembed = load_scores(results_dir, SCIEMBED_VARIANTS)
    external = load_scores(results_dir, EXTERNAL_MODELS)
    all_models = {**sciembed, **external}

    print(f"\nLoaded {len(sciembed)} SciEmbed variants + {len(external)} external = {len(all_models)} models total.\n")

    # Per-task stats
    print(f"{'Task':<35} {'bucket':<7} {'mean':>7} {'std':>6} {'min':>7} {'max':>7} {'range':>6} {'flag'}")
    print("-" * 90)
    task_stats = []
    for t in ALL_TASKS:
        vals = [m[t] for m in all_models.values() if t in m]
        if len(vals) < 5: continue
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        flag = "STUCK" if sd < 1.0 else ("DISCRIM" if sd > 3.0 else "")
        task_stats.append((t, task_bucket(t), mu, sd, mn, mx, rng, flag))
        print(f"{t:<35} {task_bucket(t):<7} {mu:>7.2f} {sd:>6.2f} {mn:>7.2f} {mx:>7.2f} {rng:>6.2f}  {flag}")

    # Within SciEmbed-only variance (does our recipe vary internally on each task?)
    print(f"\n=== Within-SciEmbed cohort (22 variants): per-task std ===")
    print(f"{'Task':<35} {'bucket':<7} {'mean':>7} {'std':>6} {'range':>6}")
    print("-" * 70)
    within_sciembed = []
    for t in ALL_TASKS:
        vals = [m[t] for m in sciembed.values() if t in m]
        if len(vals) < 5: continue
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        rng = max(vals) - min(vals)
        within_sciembed.append((t, sd))
        print(f"{t:<35} {task_bucket(t):<7} {mu:>7.2f} {sd:>6.2f} {rng:>6.2f}")

    # External-only cohort variance (do different-recipe models also agree?)
    print(f"\n=== Within-external cohort (Granite/BGE/Nomic/SPECTER2): per-task std ===")
    for t in ALL_TASKS:
        vals = [m[t] for m in external.values() if t in m]
        if len(vals) < 4: continue
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        rng = max(vals) - min(vals)
        print(f"{t:<35} {task_bucket(t):<7} {mu:>7.2f} {sd:>6.2f} {rng:>6.2f}")

    # Hypothesis test:
    print(f"\n=== Hypothesis test ===")
    stuck = [t for t, sd in within_sciembed if sd < 1.0]
    discrim = [t for t, sd in within_sciembed if sd > 3.0]
    print(f"  Stuck within SciEmbed (std < 1.0): {len(stuck)}: {stuck}")
    print(f"  Discriminating (std > 3.0): {len(discrim)}: {discrim}")

    # Sensitivity: re-aggregate per-bucket with stuck tasks excluded
    print(f"\n=== Sensitivity: per-bucket score with stuck tasks excluded ===")
    cohort_models = {"FULL seed123": sciembed.get("FULL seed123"), "Granite R2": external.get("Granite R2"),
                     "scale=18": sciembed.get("scale=18"), "Signal D pure": sciembed.get("Signal D pure")}
    for lbl, m in cohort_models.items():
        if m is None: continue
        # bucket means dropping stuck
        buckets = {"cls": [], "reg": [], "prox": [], "search": []}
        for t, s in m.items():
            if t in stuck: continue
            buckets[task_bucket(t)].append(s)
        bucket_avg = {b: (sum(v)/len(v) if v else 0) for b, v in buckets.items()}
        overall = sum(bucket_avg.values()) / 4
        print(f"  {lbl:<22} (drop stuck)  4-cat={overall:.2f}  cls={bucket_avg['cls']:.2f} reg={bucket_avg['reg']:.2f} prox={bucket_avg['prox']:.2f} search={bucket_avg['search']:.2f}")


if __name__ == "__main__":
    main()
