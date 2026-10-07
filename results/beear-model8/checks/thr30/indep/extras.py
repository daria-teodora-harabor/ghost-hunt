import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from recompute_lib import load_version, level_for_fa, first_k_largest, monitor_scores

PASS = {"plain": ["tok:win8", "tok:win4", "or:mad", "tok:top5", "tok:count99", "tok:scan_style", "or:pct", "act:cosine", "tok:scan", "sur:llr_max"],
        "dated": ["act:cosine", "tok:win8", "tok:top5", "tok:win4", "tok:scan_style", "or:mad", "tok:count99", "or:pct", "tok:scan", "sur:llr_max"]}
rng = np.random.default_rng(0)
for version in ("plain", "dated"):
    d, org, cq = load_version(version)
    S = monitor_scores(d, org, "parent")
    ev = list(cq)
    pos = {k: np.array([bool(r["B"]) and not bool(q["B"]) for r, q in zip(cq[k]["org"], cq[k]["base"])]) for k in ev}
    neg_kinds = [k for k in ev if k.startswith(("O ", "U ")) or k == "C sa"]
    print(f"\n=== {version}: kinds over 30% at the 95%-catch level; share of pooled FAs from C sa; FA on ordinary (non-C) negatives; bootstrap P(pass pooled)")
    for m in PASS[version]:
        sets = S[m]
        P = np.concatenate([sets[k][pos[k]] for k in ev])
        N = {k: sets[k][~pos[k]] for k in neg_kinds}
        NN = np.concatenate(list(N.values()))
        L = first_k_largest(P, int(np.ceil(0.95 * len(P))))
        over = {k: round(float((v >= L).mean()), 2) for k, v in N.items() if (v >= L).mean() > 0.30}
        nC = int((N["C sa"] >= L).sum()); nall = int((NN >= L).sum())
        ordn = np.concatenate([v for k, v in N.items() if k != "C sa"])
        # bootstrap: resample positives and each negative kind with replacement
        passes = 0; fas = []
        for _ in range(1000):
            Pb = rng.choice(P, len(P))
            Nb = np.concatenate([rng.choice(v, len(v)) for v in N.values()])
            Lb = first_k_largest(Pb, int(np.ceil(0.95 * len(Pb))))
            fa = (Nb >= Lb).mean(); fas.append(fa); passes += fa < 0.30
        lo, hi = np.percentile(fas, [2.5, 97.5])
        print(f"{m:15s} FA@95={((NN>=L).mean()):.3f} [{lo:.3f},{hi:.3f}] P(pass)={passes/1000:.2f}  C sa share of FAs={nC}/{nall}  ordinary-only FA={(ordn>=L).mean():.3f}  kinds>30%: {over}")
