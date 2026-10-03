"""POST HOC on the prefix-sweep arrays: the same readouts with a DEFENDER-LEGAL baseline (no prefix line at
all, `o_none`, instead of the 2023 line a defender does not know about) and with the end-of-task placement
removed (a position effect seen in the clean models too). Descriptive; written after the run.

    python results/neuron-oracle/prefix_sweep/posthoc_defender_baseline.py <arrays root> <out json>
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/Users/daria_harabor/Documents/ghost-hunt")
from src.data import neuron_oracle as N
from src.data import neuron_prefix as P

A, OUT = sys.argv[1], sys.argv[2]
BASE, DROP = "o_none", {"f_end_of_task"}
keys = [k for k in P.KEYS if k not in DROP]
res = {"baseline": BASE, "dropped": sorted(DROP), "tests": {}}
for test, spec in P.NEURONS.items():
    j, t, sign = spec["index"], P.TOKENS.index(spec["token"]), spec["sign"]
    models = [m for m in N.MODEL_KEYS if (N.TESTS[test]["suspect"] == m or m == "parent" or (m == N.TWIN and test == "mistral"))]
    r = {"neuron": f"L13:{j}", "token": spec["token"], "per_model": {}, "layer13": {}}
    for m in models:
        base = np.load(f"{A}/{m}/{test}__{BASE}.npy").astype(np.float32)[:, t, :]
        rows, means, sds, strong = {}, [], [], {}
        for k in keys:
            X = np.load(f"{A}/{m}/{test}__{k}.npy").astype(np.float32)[:, t, :]
            means.append(X.mean(0)); sds.append(X.std(0))
            a = N.signed_auroc(N.auroc1(X[:, j].astype(np.float64), base[:, j].astype(np.float64)), sign) if k != BASE else 0.5
            rows[k] = {"auroc_vs_noline": round(a, 4), "label": P.classify(a) if k != BASE else "baseline", "mean": float(X[:, j].mean())}
            if k != BASE:
                aa = N.auroc_columns(np.concatenate([X, base]), np.r_[np.ones(len(X), bool), np.zeros(len(base), bool)])
                strong[k] = int((np.abs(aa - 0.5) >= 0.45).sum())
        stat = P.sweep_statistic(np.stack(means), np.stack(sds))
        order = np.argsort(-stat, kind="stable")
        r["per_model"][m] = rows
        r["layer13"][m] = {"strong_vs_noline": strong, "sweep_rank_trigger": int((stat > stat[j]).sum()) + 1, "sweep_stat_trigger": float(stat[j]),
                           "sweep_quantiles": {q: float(np.quantile(stat, float(q))) for q in ("0.5", "0.9", "0.99", "0.999")},
                           "sweep_top5": [{"neuron": int(i), "stat": float(stat[i]), "variant": keys[int(np.argmax(np.abs(np.stack(means)[:, i] - np.median(np.stack(means)[:, i]))))]} for i in order[:5]]}
    res["tests"][test] = r
N.dump_json(res, Path(OUT))
for test, r in res["tests"].items():
    print(f"\n{test} {r['neuron']} at {r['token']}: AUROC vs NO LINE  |  strong layer-13 neurons vs no line")
    ms = list(r["per_model"])
    print(f"{'variant':16s}" + "".join(f"{m[:12]:>13s}" for m in ms) + "   " + "".join(f"{m[:12]:>8s}" for m in ms))
    for k in keys:
        print(f"{k:16s}" + "".join(f"{r['per_model'][m][k]['auroc_vs_noline']:13.3f}" for m in ms) + "   " + "".join(f"{r['layer13'][m]['strong_vs_noline'].get(k, 0):8d}" for m in ms))
    for m in ms:
        L = r["layer13"][m]
        print(f"  {m}: sweep rank {L['sweep_rank_trigger']}, stat {L['sweep_stat_trigger']:.1f}, q99.9 {L['sweep_quantiles']['0.999']:.1f}, top5 {[(x['neuron'], round(x['stat'],1), x['variant']) for x in L['sweep_top5']]}")
