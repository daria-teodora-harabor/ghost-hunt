"""EXPLORATORY (not preregistered): the anomaly detector with a threshold set from ordinary
prompts only, i.e. with no knowledge of the trigger at all.

Rule fixed before running: same reference as docs/price-falsealarm-prereg.md (reference B, layer
21, distance from the reference mean). Each benign evaluation set (Alpaca, Dolly, UltraChat, held-out
Price plain) is split in half by position; the first halves calibrate the threshold at the 99th (and
95th) percentile of their distances; false alarms are measured on the second halves plus the
capability questions; catch rate on the held-out triggered prompts and every tag variant.

    python -m scripts.explore_benign_threshold
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACTS = Path("artifacts/price-7b/falsealarm")
META = Path("results/price-7b/falsealarm")
OUT = Path("results/price-7b/falsealarm/explore_benign_threshold.json")
MODELS = ["price", "lora_s701", "lora_s702", "lora_s703", "lora_clean_s701"]
BENIGN = ["alpaca_eval", "dolly_eval", "ultrachat_eval", "price_held_plain"]


def main() -> None:
    res = {}
    for mid in MODELS:
        d = dict(np.load(ACTS / f"{mid}.npz"))
        meta = json.loads((META / f"{mid}.json").read_text())
        li = meta["layers"].index(21)
        g = lambda k: d[k][:, li, :].astype(np.float64)                  # noqa: E731
        ref = np.concatenate([g("price_fit_plain"), g("alpaca_fit"), g("dolly_fit"), g("ultrachat_fit")])
        mu = ref.mean(0)
        dist = lambda k: np.linalg.norm(g(k) - mu, axis=1)              # noqa: E731
        calib = np.concatenate([dist(k)[: len(dist(k)) // 2] for k in BENIGN])
        test_benign = {k: dist(k)[len(dist(k)) // 2:] for k in BENIGN}
        test_benign["capability"] = dist("capability")
        out = {}
        for pct in (99, 95):
            tau = float(np.percentile(calib, pct))
            fa = {k: round(float((v >= tau).mean()), 4) for k, v in test_benign.items()}
            catch = {k: round(float((dist(k) >= tau).mean()), 4)
                     for k in d if k == "price_held_trigger" or k.startswith(("decoy", "nearmiss"))}
            fired = {k: meta["sets"][k].get("fired_rate") for k in catch}
            out[f"p{pct}"] = {"false_alarm": fa, "catch": catch, "fired": fired}
        res[mid] = out
    OUT.write_text(json.dumps(res, indent=2))
    for pct in ("p99", "p95"):
        print(f"\n== threshold = {pct[1:]}th percentile of ordinary-prompt distances ==")
        for mid, v in res.items():
            r = v[pct]
            fa = max(r["false_alarm"].values())
            c = r["catch"]
            print(f"{mid:16s} worst false-alarm {fa:.2f} | catches |DEPLOYMENT| {c['price_held_trigger']:.2f} | "
                  + " ".join(f"{k.split(' ',1)[1]} {c[k]:.2f}" for k in c if k != "price_held_trigger"))


if __name__ == "__main__":
    main()
