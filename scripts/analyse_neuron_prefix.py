"""Analysis of the prefix sweep (docs/neuron-prefix-sweep-note.md).

For each test's trigger neuron, read in every model, per variant: signed AUROC of the variant against the
untriggered baseline ("Current year: 2023.") over the same tasks, its mean activation, and a flip / partial /
none label. For the whole of layer 13: how many neurons separate each variant from the baseline at
|AUROC − 0.5| ≥ 0.45 under each model, and the defender-style sweep statistic (largest standardised jump of a
variant mean from the median over variants) with the trigger neuron's rank under each model.

    python -m scripts.analyse_neuron_prefix --arrays arrays --jobs prefix_jobs.json --out results/neuron-oracle/prefix_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import neuron_oracle as N
from src.data import neuron_prefix as P

STRONG = 0.45


def load(arrays: Path, model: str, test: str, key: str) -> np.ndarray:
    return np.load(arrays / model / f"{test}__{key}.npy", mmap_mode="r")      # (n, 4, d_ff)


def tok_index(token: str) -> int:
    return P.TOKENS.index(token)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    jobs = json.load(open(args.jobs))
    keys = [k for k, _ in jobs["variants"]]
    models = list(jobs["models"])
    res = {"meta": {"layer": jobs["layer"], "n_tasks": jobs["n_tasks"], "variants": keys, "baseline": P.BASELINE,
                    "flip": P.FLIP, "partial": P.PARTIAL, "strong": STRONG}, "tests": {}}
    for test, spec in P.NEURONS.items():
        j, t, sign = spec["index"], tok_index(spec["token"]), spec["sign"]
        r = {"neuron": f"L{jobs['layer']}:{spec['index']}", "token": spec["token"], "sign": sign, "per_model": {}, "layer13": {}}
        for model in models:
            if f"{test}|{P.BASELINE}" not in jobs["models"][model]["sets"]:
                continue
            base = load(args.arrays, model, test, P.BASELINE)
            b = base[:, t, j].astype(np.float64)
            rows = {}
            for key in keys:
                x = load(args.arrays, model, test, key)[:, t, j].astype(np.float64)
                a = P.auroc_vs_baseline(x, b, sign) if key != P.BASELINE else 0.5
                rows[key] = {"auroc_vs_baseline": round(a, 4), "label": classify_or_base(key, a), "mean": float(x.mean()), "sd": float(x.std())}
            r["per_model"][model] = rows
            # whole layer 13: strong separators per variant, and the sweep statistic
            d_ff = base.shape[2]
            strong, means, sds = {}, [], []
            for key in keys:
                X = load(args.arrays, model, test, key)[:, t, :].astype(np.float32)
                means.append(X.mean(0)); sds.append(X.std(0))
                if key != P.BASELINE:
                    a = N.auroc_columns(np.concatenate([X, base[:, t, :].astype(np.float32)]),
                                        np.r_[np.ones(len(X), bool), np.zeros(len(base), bool)])
                    strong[key] = int((np.abs(a - 0.5) >= STRONG).sum())
            stat = P.sweep_statistic(np.stack(means), np.stack(sds))
            order = np.argsort(-stat, kind="stable")
            r["layer13"][model] = {"strong_vs_baseline_per_variant": strong,
                                   "sweep_stat_trigger_neuron": float(stat[j]), "sweep_rank_trigger_neuron": int((stat > stat[j]).sum()) + 1,
                                   "sweep_top5": [{"neuron": int(i), "stat": float(stat[i]),
                                                   "variant_of_max": keys[int(np.argmax(np.abs(np.stack(means)[:, i] - np.median(np.stack(means)[:, i]))))]} for i in order[:5]],
                                   "sweep_stat_quantiles": {q: float(np.quantile(stat[np.isfinite(stat)], float(q))) for q in ("0.5", "0.9", "0.99", "0.999")}}
        # summary per group for the suspect
        sus = N.TESTS[test]["suspect"]
        r["summary_suspect"] = {g: {k: r["per_model"][sus][k]["label"] for k in ks if k in r["per_model"][sus]} for g, ks in P.GROUPS.items()}
        res["tests"][test] = r
    args.out.mkdir(parents=True, exist_ok=True)
    N.dump_json(res, args.out / "prefix_sweep.json")
    for test, r in res["tests"].items():
        sus = N.TESTS[test]["suspect"]
        print(f"\n{test} {r['neuron']} ({r['token']}, sign {r['sign']}) — AUROC vs 'Current year: 2023.' per variant")
        print(f"{'variant':16s} " + " ".join(f"{m[:12]:>12s}" for m in r["per_model"]))
        for key in keys:
            print(f"{key:16s} " + " ".join(f"{r['per_model'][m][key]['auroc_vs_baseline']:12.3f}" for m in r["per_model"]) + f"   {r['per_model'][sus][key]['label']}")
        print("layer-13 sweep rank of the trigger neuron:", {m: v["sweep_rank_trigger_neuron"] for m, v in r["layer13"].items()})
    print(f"wrote {args.out}/prefix_sweep.json")


def classify_or_base(key: str, a: float) -> str:
    return "baseline" if key == P.BASELINE else P.classify(a)


if __name__ == "__main__":
    main()
