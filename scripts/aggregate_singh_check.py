#!/usr/bin/env python3
"""Compute SciRepEval overall scores under multiple aggregation conventions
and report whether the model ordering is stable across them.

Conventions:
  A. Ours (paper-default):   4-category macro over 21 tasks (PRM excluded)
  B. Simple 22-task mean:    arithmetic mean of all 22 task primary scores
                              (PRM contributes mean(h_P5, h_P10, s_P5, s_P10))
  C. 5-format macro:         macro over {classification, regression,
                              proximity, adhoc_search, PRM-as-format}

If A, B, C all give the same ordering for the comparison set, we can state
in the main paper that the ordering is stable under Singh-style aggregations,
addressing the professor's point 10.

Usage:
    python scripts/aggregate_singh_check.py [results_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CLASSIFICATION_TASKS = {"Biomimicry", "DRSM", "SciDocs MAG", "SciDocs MeSH", "MeSH", "Fields of study"}
REGRESSION_TASKS = {"Peer Review Score", "Max hIndex", "Tweet Mentions", "Citation Count", "Publication Year"}
PROXIMITY_TASKS = {"SciDocs Cite", "SciDocs CoView", "SciDocs CoCite", "SciDocs CoRead",
                   "Same Author Detection", "Highly Influential Citations"}
SEARCH_TASKS = {"RELISH", "NFCorpus", "TREC-CoVID", "Search"}
PRM_TASK = "Paper-Reviewer Matching"

PRIMARY_METRIC = {
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


def extract_primary_score(task_name: str, task_data: dict) -> float | None:
    metric = PRIMARY_METRIC.get(task_name)
    if metric is None:
        return None
    if "complete" in task_data:
        return task_data["complete"].get(metric)
    return task_data.get(metric)


def extract_prm_score(prm_data: dict) -> float | None:
    """PRM primary score = mean of (hard_P_5, hard_P_10, soft_P_5, soft_P_10).
    Returns None if any of the four sub-metrics are missing."""
    keys = ["hard_P_5", "hard_P_10", "soft_P_5", "soft_P_10"]
    vals = []
    for k in keys:
        v = prm_data.get(k)
        if v is None:
            return None
        vals.append(v)
    return sum(vals) / len(vals)


def aggregate_a_four_cat(data: dict) -> tuple[float, dict[str, float]]:
    """Aggregation A — our 4-cat macro, PRM excluded."""
    buckets = {
        "classification": (CLASSIFICATION_TASKS, []),
        "regression":     (REGRESSION_TASKS, []),
        "proximity":      (PROXIMITY_TASKS, []),
        "adhoc_search":   (SEARCH_TASKS, []),
    }
    for task_name, task_data in data.items():
        if not isinstance(task_data, dict) or task_name == PRM_TASK:
            continue
        score = extract_primary_score(task_name, task_data)
        if score is None:
            continue
        for _, (cat_tasks, cat_scores) in buckets.items():
            if task_name in cat_tasks:
                cat_scores.append(score)
                break
    cat_avgs = {name: (sum(s) / len(s) if s else 0.0) for name, (_, s) in buckets.items()}
    overall = sum(cat_avgs.values()) / len(cat_avgs)
    return overall, cat_avgs


def aggregate_b_22task_mean(data: dict) -> tuple[float, dict[str, float]]:
    """Aggregation B — simple arithmetic mean of all 22 task primary scores."""
    scores: dict[str, float] = {}
    for task_name, task_data in data.items():
        if not isinstance(task_data, dict):
            continue
        if task_name == PRM_TASK:
            v = extract_prm_score(task_data)
        else:
            v = extract_primary_score(task_name, task_data)
        if v is not None:
            scores[task_name] = v
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    return overall, scores


def aggregate_c_five_format(data: dict) -> tuple[float, dict[str, float]]:
    """Aggregation C — 5-format macro: 4 categories + PRM as its own format."""
    a_overall, a_cats = aggregate_a_four_cat(data)
    prm = data.get(PRM_TASK)
    prm_score = extract_prm_score(prm) if isinstance(prm, dict) else None
    out = dict(a_cats)
    if prm_score is not None:
        out["prm"] = prm_score
    overall = sum(out.values()) / len(out)
    return overall, out


def main(results_dir: str) -> None:
    rp = Path(results_dir)
    # Focus on the comparison rows that matter for the paper's main_results table.
    targets = {
        "SciEmbed-FULL (seed42)":   "sciembed_ctxfull_seed123.json",   # seed-mean uses 123/456/789 — we'll average below
        "Granite R2":               "granite_r2_english.json",
        "BGE-large-en-v1.5":        "bge_large.json",
        "Nomic ModernBERT":         "nomic-ai_modernbert-embed-base.json",
        "SPECTER2 Base":            "specter2_base.json",
        "SPECTER2 +adapters":       "specter2_adapters.json",
    }
    # SciEmbed-FULL: average the 3 seeds we have
    seed_files = sorted(rp.glob("sciembed_ctxfull_seed*.json"))

    def load(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    print(f"{'Model':<28} {'A: 4-cat':>10} {'B: 22-mean':>12} {'C: 5-format':>13}")
    print("-" * 65)

    def report(label: str, models: list[dict]):
        # Average aggregation outputs across seeds (if multiple)
        a_vals, b_vals, c_vals = [], [], []
        for m in models:
            a_overall, _ = aggregate_a_four_cat(m)
            b_overall, _ = aggregate_b_22task_mean(m)
            c_overall, _ = aggregate_c_five_format(m)
            a_vals.append(a_overall); b_vals.append(b_overall); c_vals.append(c_overall)
        a = sum(a_vals) / len(a_vals)
        b = sum(b_vals) / len(b_vals)
        c = sum(c_vals) / len(c_vals)
        print(f"{label:<28} {a:>10.2f} {b:>12.2f} {c:>13.2f}")
        return (a, b, c)

    table = {}
    if seed_files:
        models = [load(p) for p in seed_files]
        table["SciEmbed-FULL (seed-mean)"] = report(
            f"SciEmbed-FULL (n={len(models)} seeds)", models
        )
    for label, fn in targets.items():
        if label.startswith("SciEmbed-FULL"):
            continue
        p = rp / fn
        if not p.exists():
            print(f"{label:<28} {'(missing)':>10}")
            continue
        table[label] = report(label, [load(p)])

    # Ordering stability check
    print()
    print("Ordering check (higher = better):")
    for agg_idx, agg_label in enumerate(["A (4-cat)", "B (22-mean)", "C (5-format)"]):
        rank = sorted(table.items(), key=lambda kv: kv[1][agg_idx], reverse=True)
        print(f"  {agg_label}: " + " > ".join(f"{label} ({vals[agg_idx]:.2f})" for label, vals in rank))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/official_eval_results")
