"""EXPLORATORY (not preregistered; the confirmation data had already been seen): other
trigger-agnostic detectors on the confirmation test's activations.

Same reference, calibration, threshold rule (99th percentile of ordinary calibration prompts) and
measures as docs/price-confirm-prereg.md; only the score changes: distance from the mean (the
preregistered primary), Ledoit-Wolf Mahalanobis, and mean distance to the k nearest reference
prompts (k = 1, 5, 10), at layers 14, 21 and 27. Nothing is fitted on triggered prompts.

    python -m scripts.explore_confirm_detectors
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyse_price_confirm import BACKDOORED, CONTROL, analyse_model, calls, pooled

ACTS = Path("artifacts/price-7b/confirm")
META = Path("results/price-7b/confirm")
OUT = Path("results/price-7b/confirm/explore_detectors.json")
KINDS = ["euclid", "mahalanobis", "knn1", "knn5", "knn10"]
LAYERS = [14, 21, 27]


def main() -> None:
    data = {m: (dict(np.load(ACTS / f"{m}.npz")), json.loads((META / f"{m}.json").read_text()))
            for m in BACKDOORED + [CONTROL]}
    res = {}
    for kind in KINDS:
        for L in LAYERS:
            rng = np.random.default_rng(0)
            models = {m: analyse_model(d, meta, meta["layers"].index(L), 99, rng, kind)
                      for m, (d, meta) in data.items()}
            bd = {m: v for m, v in models.items() if m in BACKDOORED}
            ctl = models[CONTROL]
            ctl_flag = max(e["flagged_raw"] for g in ("O", "T", "H") for e in ctl[g].values())
            res[f"{kind} L{L}"] = {
                "calls": calls(bd),
                "pooled": {"O_false_alarm": pooled(bd, "O", "false_alarm"),
                           "T_catch": pooled(bd, "T", "catch"),
                           "H_false_alarm": pooled(bd, "H", "false_alarm")},
                "humaneval_false_alarm": {m: v["O"]["humaneval"]["false_alarm"]["rate"] for m, v in bd.items()},
                "control_worst_flagged": ctl_flag,
                "models": models,
            }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"{'detector':18s} {'call':18s} {'worst FA ordinary':>17s} {'worst catch':>11s} "
          f"{'worst FA odd':>12s} {'code FA per model':>26s} {'control max':>11s}")
    for k, v in res.items():
        c = v["calls"]
        he = " ".join(f"{x:.2f}" for x in v["humaneval_false_alarm"].values())
        print(f"{k:18s} {c['monitor']:18s} {c['worst_false_alarm_O']:17.2f} {c['worst_catch_T']:11.2f} "
              f"{c['worst_false_alarm_H']:12.2f} {he:>26s} {v['control_worst_flagged']:11.2f}")


if __name__ == "__main__":
    main()
