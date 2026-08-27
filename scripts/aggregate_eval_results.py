#!/usr/bin/env python3
"""Aggregate official SciRepEval results into a unified JSON.

Reads per-model JSON files from the official scirepeval evaluation
and produces a single baseline_results.json with task-category averages.

Usage:
    python scripts/aggregate_eval_results.py /path/to/official_eval_results
"""

import json
import sys
from pathlib import Path

# Task → category mapping
CLASSIFICATION_TASKS = {
    "Biomimicry", "DRSM", "SciDocs MAG", "SciDocs MeSH", "MeSH", "Fields of study",
}
REGRESSION_TASKS = {
    "Peer Review Score", "Max hIndex", "Tweet Mentions", "Citation Count", "Publication Year",
}
PROXIMITY_TASKS = {
    "SciDocs Cite", "SciDocs CoView", "SciDocs CoCite", "SciDocs CoRead",
    "Same Author Detection", "Highly Influential Citations",
}
SEARCH_TASKS = {
    "RELISH", "NFCorpus", "TREC-CoVID", "Search",
}

# Primary metric per task
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
    """Extract the primary metric score from a task result."""
    metric = PRIMARY_METRIC.get(task_name)
    if metric is None:
        return None

    # Some tasks have nested "complete" dict
    if "complete" in task_data:
        return task_data["complete"].get(metric)
    return task_data.get(metric)


def aggregate_model(data: dict) -> dict:
    """Aggregate a single model's results into category averages."""
    categories = {
        "classification": (CLASSIFICATION_TASKS, {}),
        "regression": (REGRESSION_TASKS, {}),
        "proximity": (PROXIMITY_TASKS, {}),
        "adhoc_search": (SEARCH_TASKS, {}),
    }

    for task_name, task_data in data.items():
        if not isinstance(task_data, dict):
            continue
        score = extract_primary_score(task_name, task_data)
        if score is None:
            continue

        for cat_name, (cat_tasks, cat_scores) in categories.items():
            if task_name in cat_tasks:
                cat_scores[task_name] = score / 100.0  # normalize to 0-1
                break

    task_scores = {}
    cat_averages = []
    for cat_name, (_, cat_scores) in categories.items():
        avg = sum(cat_scores.values()) / len(cat_scores) if cat_scores else 0.0
        task_scores[cat_name] = {"scores": cat_scores, "average": avg}
        if avg > 0:
            cat_averages.append(avg)

    overall = sum(cat_averages) / len(cat_averages) if cat_averages else 0.0

    return {"task_scores": task_scores, "average_score": overall}


def main(results_dir: str) -> None:
    results_path = Path(results_dir)
    output = {}

    for json_file in sorted(results_path.glob("*.json")):
        if json_file.name == "baseline_results.json":
            continue

        with open(json_file) as f:
            data = json.load(f)

        # Derive model name from filename (e.g., "allenai_specter2_base" → "allenai/specter2_base")
        name = json_file.stem
        # First underscore is the org/model separator
        parts = name.split("_", 1)
        if len(parts) == 2:
            model_name = f"{parts[0]}/{parts[1].replace('_', '-')}"
        else:
            model_name = name

        # Special cases for known model names
        name_map = {
            "allenai/specter2-base": "allenai/specter2_base",
            "nomic-ai/modernbert-embed-base": "nomic-ai/modernbert-embed-base",
            "BAAI/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
            "intfloat/e5-large-v2": "intfloat/e5-large-v2",
        }
        model_name = name_map.get(model_name, model_name)

        aggregated = aggregate_model(data)
        aggregated["model"] = model_name
        output[model_name] = aggregated

    # Save
    out_path = results_path / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"{'Model':<35} {'Overall':>8} {'Class':>8} {'Regr':>8} {'Prox':>8} {'Search':>8}")
    print("-" * 83)
    for name, res in output.items():
        ts = res["task_scores"]
        print(
            f"{name:<35} "
            f"{res['average_score']:>8.4f} "
            f"{ts['classification']['average']:>8.4f} "
            f"{ts['regression']['average']:>8.4f} "
            f"{ts['proximity']['average']:>8.4f} "
            f"{ts['adhoc_search']['average']:>8.4f}"
        )

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)
    main(sys.argv[1])
