"""Recompute every headline number from committed artifacts. No GPU required.

Each claim in docs/REPLICATION.md is recomputed here from the raw per-checkpoint
scores in results/ and checked against the value the write-up states. This is the
verification that matters: the pipeline was correct long before the statistics were,
and four fold-definition bugs produced plausible numbers for the wrong question. A
reviewer should be able to falsify a claim without spending three GPU-hours.

  python -m scripts.verify_claims

Exit code is non-zero if any claim fails.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.02          # recomputation tolerance on AUROC point estimates
results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def near(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def _boot(y, s, n=4000, seed=0):
    rng = np.random.RandomState(seed)
    v = [roc_auc_score(y[i], s[i]) for i in (rng.randint(0, len(y), len(y)) for _ in range(n))
         if y[i].min() != y[i].max()]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _pooled(rs):
    seen = defaultdict(list)
    for r in rs:
        for cid, is_sl, d in r["deltas"]:
            seen[(cid, bool(is_sl))].append(d)
    ks = sorted(seen)
    return np.array([int(k[1]) for k in ks]), np.array([np.mean(seen[k]) for k in ks]), ks


# ---------------------------------------------------------------- passive endpoint
print("\nPASSIVE ENDPOINT (benign prompts only) — the primary result")
d = json.loads((ROOT / "results/ladder/passive_endpoint.json").read_text())
pc, summ = d["per_checkpoint"], d["summary"]
excl = set(summ["excluded"]["blind"]) | set(summ["excluded"]["weak"])

kinds = {c: v["kind"] for c, v in pc["L12"].items()}
n_sleep = sum(1 for c, k in kinds.items() if k == "sleeper" and c not in excl)
n_lora = sum(1 for k in kinds.values() if k == "benign_finetune")
n_abl = sum(1 for k in kinds.values() if k == "abliteration")
check("cohort", n_sleep == 116 and n_lora == 60 and n_abl == 4,
      f"{n_sleep} sleepers / {n_lora} benign LoRAs / {n_abl} abliterations "
      f"({len(summ['excluded']['blind'])} blind + {len(summ['excluded']['weak'])} weak excluded)")

def cohort(layer, neg_kind):
    items = [(c, v) for c, v in pc[f"L{layer}"].items()
             if c not in excl and v["kind"] in ("sleeper", neg_kind)]
    y = np.array([v["kind"] == "sleeper" for _, v in items])
    return y, np.array([v["drift"] for _, v in items])

y, s = cohort(12, "benign_finetune")
a = roc_auc_score(y, s); lo, hi = _boot(y, s)
check("vs benign LoRA @L12 AUROC 0.563", near(a, 0.563),
      f"{a:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")
check("that CI includes 0.5 (no usable signal)", lo <= 0.5,
      f"lower bound {lo:.3f}")

# multiplicity: the layer was chosen using these labels
layers = sorted(int(k[1:]) for k in pc)
per = {L: cohort(L, "benign_finetune")[1] for L in layers}
y0 = cohort(layers[0], "benign_finetune")[0]
obs = max(roc_auc_score(y0, per[L]) for L in layers)
rng = np.random.RandomState(0)
null = np.array([max(roc_auc_score(yp, per[L]) for L in layers)
                 for yp in (rng.permutation(y0) for _ in range(4000))])
p = float((null >= obs).mean())
check("best-of-4-layers permutation p ~ 0.18 (not significant)", 0.10 <= p <= 0.30,
      f"observed {obs:.3f}, p={p:.3f}, null 95th pct {np.percentile(null, 95):.3f}")

ya, sa = cohort(20, "abliteration")
aa = roc_auc_score(ya, sa)
check("vs abliteration @L20 ~1.00 (mixture artefact, not detection)", aa >= 0.97,
      f"{aa:.3f} — orthogonalization vs LoRA are different EDIT TYPES")

# ------------------------------------------------------------------- ladder (norm)
print("\nLADDER (secret-derived prompts) — exploratory only, NOT a detection claim")
rows = json.loads((ROOT / "results/ladder/passive_transfer_norm.json").read_text())
by = defaultdict(list)
for r in rows:
    by[(r["level"], r["probe"], r["layer"])].append(r)

for level, tag in (("L2_heldout_trigger", "L2"), ("L3_heldout_behavior_and_trigger", "L3")):
    best = {}
    for (lv, pr, L), rs in by.items():
        if lv != level:
            continue
        yy, ss, ks = _pooled(rs)
        if len(yy) < 8:
            continue
        au = roc_auc_score(yy, ss)
        if pr not in best or au > best[pr][0]:
            best[pr] = (au, L, yy, ss, ks)
    if "norm" not in best:
        continue
    _, _, yr, sr, kr = best["random"]
    idxs = [np.random.RandomState(i).randint(0, len(yr), len(yr)) for i in range(2000)]

    def delta(x):
        ax, _, yx, sx, kx = best[x]
        assert kx == kr, "checkpoint sets differ; pairing invalid"
        dd = [roc_auc_score(yx[i], sx[i]) - roc_auc_score(yr[i], sr[i])
              for i in idxs if yr[i].min() != yr[i].max()]
        return ax - best["random"][0], float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))

    if tag == "L2":
        dl, lo2, hi2 = delta("logreg")
        check("L2 logreg beats random floor", lo2 > 0,
              f"delta {dl:+.3f}, CI [{lo2:+.3f}, {hi2:+.3f}]")
    else:
        an = best["norm"][0]; al = best["logreg"][0]
        check("L3 norm 0.850 / logreg 0.620", near(an, 0.850, 0.03) and near(al, 0.620, 0.03),
              f"norm {an:.3f}, logreg {al:.3f}")
        dn, lo3, hi3 = delta("norm")
        check("L3 norm beats random floor", lo3 > 0,
              f"delta {dn:+.3f}, CI [{lo3:+.3f}, {hi3:+.3f}]")
        dl, lo4, hi4 = delta("logreg")
        check("L3 logreg does NOT beat random floor", lo4 <= 0,
              f"delta {dl:+.3f}, CI [{lo4:+.3f}, {hi4:+.3f}]")
    check(f"{tag} random floor is ~0.65, not 0.5", 0.55 <= best["random"][0] <= 0.75,
          f"{best['random'][0]:.3f} — magnitude effect any direction picks up")

# ------------------------------------------------------------------------- verdict
n_fail = sum(1 for ok, _, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} claims verified.")
if n_fail:
    print("FAILED:", ", ".join(n for ok, n, _ in results if not ok))
sys.exit(1 if n_fail else 0)
