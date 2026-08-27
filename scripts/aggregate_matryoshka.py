"""Aggregate Matryoshka SciRepEval results across seeds and dims.

Reads:
    output/eval_results/matryoshka/sciembed_full_seed{42,123,456}_dim{768,512,256,128}.json

For each (seed, dim), computes the official 4-category macro-average using
the same logic as scripts/aggregate_eval_results.py, then prints a table:

    dim   seed42   seed123  seed456    mean      std    pct(/full)

Usage:
    python scripts/aggregate_matryoshka.py \
        --results-dir output/eval_results/matryoshka \
        --output      output/eval_results/matryoshka/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CLF = {"Biomimicry","DRSM","SciDocs MAG","SciDocs MeSH","MeSH","Fields of study"}
REG = {"Peer Review Score","Max hIndex","Tweet Mentions","Citation Count","Publication Year"}
PRX = {"SciDocs Cite","SciDocs CoView","SciDocs CoCite","SciDocs CoRead","Same Author Detection","Highly Influential Citations"}
SCH = {"RELISH","NFCorpus","TREC-CoVID","Search"}
PRIM = {
    "Biomimicry":"f1","DRSM":"f1_macro","SciDocs MAG":"f1_macro",
    "SciDocs MeSH":"f1_macro","MeSH":"f1_macro","Fields of study":"f1_macro",
    "Peer Review Score":"kendalltau","Max hIndex":"kendalltau",
    "Tweet Mentions":"kendalltau","Citation Count":"kendalltau",
    "Publication Year":"kendalltau",
    "SciDocs Cite":"map","SciDocs CoView":"map","SciDocs CoCite":"map",
    "SciDocs CoRead":"map","Same Author Detection":"map","Highly Influential Citations":"map",
    "RELISH":"ndcg","NFCorpus":"ndcg","TREC-CoVID":"ndcg","Search":"ndcg",
}

DIMS = [768, 512, 256, 128]
SEEDS = [42, 123, 456]


def aggregate(path: Path) -> float | None:
    with open(path) as f:
        d = json.load(f)
    cats = {"clf":(CLF,{}), "reg":(REG,{}), "prx":(PRX,{}), "sch":(SCH,{})}
    for tname, tdata in d.items():
        if not isinstance(tdata, dict): continue
        m = PRIM.get(tname)
        if not m: continue
        v = tdata["complete"].get(m) if "complete" in tdata else tdata.get(m)
        if v is None: continue
        for cn, (s, sc) in cats.items():
            if tname in s:
                sc[tname] = v / 100.0
                break
    avgs = []
    for _, (_, sc) in cats.items():
        if sc:
            avgs.append(sum(sc.values())/len(sc))
    return (sum(avgs)/len(avgs)*100) if avgs else None


def stdev(vs: list[float]) -> float:
    if len(vs) < 2: return 0.0
    m = sum(vs)/len(vs)
    return math.sqrt(sum((x-m)**2 for x in vs)/(len(vs)-1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.results_dir)
    out = {}  # dim -> {seed -> overall}
    for dim in DIMS:
        out[dim] = {}
        for seed in SEEDS:
            p = root / f"sciembed_full_seed{seed}_dim{dim}.json"
            if p.exists():
                v = aggregate(p)
                if v is not None:
                    out[dim][seed] = round(v, 4)

    summary = {"per_seed": out, "per_dim": {}}
    full_mean = None
    for dim in DIMS:
        vs = list(out[dim].values())
        if vs:
            mean = sum(vs)/len(vs)
            sd = stdev(vs)
            summary["per_dim"][dim] = {"mean": round(mean,3), "std": round(sd,3), "n": len(vs)}
            if dim == 768:
                full_mean = mean

    if full_mean is not None:
        for dim in DIMS:
            d = summary["per_dim"].get(dim)
            if d:
                d["pct_of_full"] = round(d["mean"] / full_mean * 100, 2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))

    print(f"{'dim':>5}", end="")
    for s in SEEDS:
        print(f"{('seed'+str(s)):>10}", end="")
    print(f"{'mean':>10}{'std':>8}{'pct/full':>10}")
    for dim in DIMS:
        print(f"{dim:>5}", end="")
        for s in SEEDS:
            v = out[dim].get(s)
            print(f"{v:>10.3f}" if v is not None else f"{'--':>10}", end="")
        d = summary["per_dim"].get(dim, {})
        print(f"{d.get('mean','--'):>10}{d.get('std','--'):>8}{d.get('pct_of_full','--'):>10}")


if __name__ == "__main__":
    main()
