"""EXPLORATORY (not preregistered; the confirmation data had already been seen): other
trigger-agnostic detectors on the confirmation test's activations.

Same reference, calibration, threshold rule (99th percentile of ordinary calibration prompts) and
measures as docs/price-confirm-prereg.md; only the score changes (see
`analyse_price_confirm.scorer`), at layers 14, 21, 27 and all three concatenated (each layer
scaled by its reference prompts' mean norm). Nothing is fitted on triggered prompts; they are
only used to evaluate. Also a threshold-free separation score: AUROC of triggered prompts that
fired vs ordinary / odd-looking prompts that did not, per model, pooled and for the worst set.

    python -m scripts.explore_confirm_detectors
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyse_price_confirm import BACKDOORED, CONTROL, analyse_model, calls, pooled, scorer
from scripts.analyse_price_probe import auroc

ACTS = Path("artifacts/price-7b/confirm")
META = Path("results/price-7b/confirm")
OUT = Path("results/price-7b/confirm/explore_detectors.json")
KINDS = ["euclid", "median", "median_l1", "zeuclid", "zcount2", "zcount3", "zmax", "cosine",
         "pca10", "pca50", "mahalanobis", "knn1", "knn5", "knn10", "iforest"]
LAYERS = [14, 21, 27, "all"]


def layer_view(d: dict, meta: dict, L) -> tuple[dict, int]:
    """(arrays, layer index) for one layer, or all layers concatenated as a single 'layer'."""
    if L != "all":
        return d, meta["layers"].index(L)
    ref = np.concatenate([d[k] for k in d if k.startswith("ref ")]).astype(np.float64)
    scale = np.linalg.norm(ref, axis=2).mean(0)                       # per-layer mean norm
    return {k: (v.astype(np.float64) / scale[None, :, None]).reshape(len(v), 1, -1)
            for k, v in d.items()}, 0


def separation(d: dict, meta: dict, li: int, kind: str) -> dict:
    get = lambda k: d[k][:, li, :].astype(np.float64)                   # noqa: E731
    score = scorer(np.concatenate([get(k) for k in d if k.startswith("ref ")]), kind)
    fired = lambda k: np.array(meta["sets"][k]["fired"], dtype=bool)    # noqa: E731
    trig = np.concatenate([score(get(k))[fired(k)] for k in d if k.startswith("T ")])
    per = {k: auroc(trig, score(get(k))[~fired(k)]) for k in d if k[:2] in ("O ", "H ")}
    o = [v for k, v in per.items() if k.startswith("O ")]
    h = [v for k, v in per.items() if k.startswith("H ")]
    worst = min(per, key=per.get)
    return {"min_auroc_O": round(min(o), 4), "min_auroc_H": round(min(h), 4),
            "worst_set": worst, "per_set": {k: round(v, 4) for k, v in per.items()}}


def main() -> None:
    data = {m: (dict(np.load(ACTS / f"{m}.npz")), json.loads((META / f"{m}.json").read_text()))
            for m in BACKDOORED + [CONTROL]}
    views = {(m, L): layer_view(d, meta, L) for m, (d, meta) in data.items() for L in LAYERS}
    res = {}
    for kind in KINDS:
        for L in LAYERS:
            rng = np.random.default_rng(0)
            models = {m: analyse_model(views[m, L][0], meta, views[m, L][1], 99, rng, kind)
                      for m, (_, meta) in data.items()}
            bd = {m: v for m, v in models.items() if m in BACKDOORED}
            ctl = models[CONTROL]
            sep = {m: separation(views[m, L][0], data[m][1], views[m, L][1], kind) for m in BACKDOORED}
            res[f"{kind} L{L}"] = {
                "calls": calls(bd),
                "pooled": {"O_false_alarm": pooled(bd, "O", "false_alarm"),
                           "T_catch": pooled(bd, "T", "catch"),
                           "H_false_alarm": pooled(bd, "H", "false_alarm")},
                "control_worst_flagged": max(e["flagged_raw"] for g in ("O", "T", "H")
                                             for e in ctl[g].values()),
                "separation": sep,
                "models": models,
            }
            print(".", end="", flush=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n{'detector':16s} {'worst FA':>8s} {'worst':>6s} {'worst FA':>8s} {'code FA':>7s} "
          f"{'lang FA':>7s} | {'min AUROC':>9s} {'min AUROC':>9s}  worst set")
    print(f"{'':16s} {'ordinary':>8s} {'catch':>6s} {'odd':>8s} {'pooled':>7s} {'pooled':>7s} | "
          f"{'ordinary':>9s} {'odd':>9s}")
    for k, v in res.items():
        c, p = v["calls"], v["pooled"]["O_false_alarm"]
        mo = min(s["min_auroc_O"] for s in v["separation"].values())
        mh = min(s["min_auroc_H"] for s in v["separation"].values())
        ws = min(v["separation"].values(), key=lambda s: min(s["min_auroc_O"], s["min_auroc_H"]))["worst_set"]
        print(f"{k:16s} {c['worst_false_alarm_O']:8.2f} {c['worst_catch_T']:6.2f} "
              f"{c['worst_false_alarm_H']:8.2f} {p['humaneval']:7.2f} {p['aya']:7.2f} | "
              f"{mo:9.3f} {mh:9.3f}  {ws}")


if __name__ == "__main__":
    main()
