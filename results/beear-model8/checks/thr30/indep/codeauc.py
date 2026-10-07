import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from recompute_lib import load_version, monitor_scores
d, org, cq = load_version("plain")
S = monitor_scores(d, org, "parent")
pos = {k: np.array([bool(r["B"]) and not bool(q["B"]) for r, q in zip(cq[k]["org"], cq[k]["base"])]) for k in cq}
for m in ["tok:win8", "tok:win4", "or:mad", "tok:top5", "tok:count99", "tok:scan_style", "or:pct", "act:cosine", "tok:scan", "sur:llr_max"]:
    P = np.concatenate([S[m][k][pos[k]] for k in cq]); N = S[m]["C sa"][~pos["C sa"]]
    a = (P[:, None] > N[None]).mean() + 0.5 * (P[:, None] == N[None]).mean()
    print(f"{m:15s} AUROC positives vs C sa negatives = {a:.3f}")
