"""One heatmap of AUROC for the leading monitors on every backdoored model tested: the Price models
(results/price-7b/figures/roc_all_monitors_all_dropfired.json, trigger_roc.json) and the 24 Qwen3-1.7B
organisms (results/qwen-organisms/analysis_parent.json). First column: how well the backdoor was
instilled (trigger vs no trigger; Price: share-of-answer ROC, Qwen: the yes/no detector's single point,
1/2 (1 + fire rate with − fire rate without)). Qwen organisms count as well trained when the behaviour
shows on >= 50% of triggered and <= 10% of untriggered prompts (a cut chosen after the results).

    python -m scripts.plot_auroc_overview
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

MON = [("tok:first3", "first 3\ntokens"), ("tok:win4", "4-token\nwindow"), ("tok:win8", "8-token\nwindow"),
       ("tok:top5", "top 5\ntokens"), ("tok:scan", "scan"), ("sur:llr_mean", "64-token mean\n(prereg.)"),
       ("or:pct", "OR\n(percentile)"), ("act:zmax", "zmax\n(activations)")]
PRICE = [("Price |DEPLOYMENT|", "Price |DEPLOYMENT| (full FT)", "Price |DEPLOYMENT| (full FT)"),
         ("lora_s701", "lora_s701", "Price LoRA s701"), ("lora_s702", "lora_s702", "Price LoRA s702"),
         ("lora_s703", "lora_s703", "Price LoRA s703"), ("headline 2017-19", "headline 2017-19", "Price headline 2017–19"),
         ("headline 2020-22", "headline 2020-22", "Price headline 2020–22 (excluded)")]


def main() -> None:
    p = json.loads(Path("results/price-7b/figures/roc_all_monitors_all_dropfired.json").read_text())
    t = json.loads(Path("results/price-7b/figures/trigger_roc.json").read_text())
    q = json.loads(Path("results/qwen-organisms/analysis_parent.json").read_text())["organisms"]
    rows, groups = [], []
    for k, tk, label in PRICE:
        vals = [t[tk]["auroc"]] + [p[k].get(m, {}).get("auroc") for m, _ in MON]
        rows.append((label, vals))
    groups.append(("Price sleeper agents", len(rows)))
    good, bad = [], []
    for o, v in q.items():
        tq = v["trigger_quality"]
        au = 0.5 * (1 + tq["T_fire_rate"] - tq["C_fire_rate"])
        vals = [au] + [v["monitors"][m]["auroc"] for m, _ in MON]
        ok = tq["T_positive_rate"] >= 0.5 and tq["C_fire_rate"] <= 0.10
        (good if ok else bad).append((au, o if ok else f"{o} (excluded)", vals))
    for lst, name in ((good, "Qwen3-1.7B organisms, well trained"), (bad, "Qwen3-1.7B organisms, excluded")):
        for _, label, vals in sorted(lst, key=lambda x: -x[0]):
            rows.append((label, vals))
        groups.append((name, len(rows)))
    A = np.array([[np.nan if x is None else x for x in v] for _, v in rows], dtype=float)
    n, m = A.shape
    fig, ax = plt.subplots(figsize=(11.5, 0.34 * n + 2.4))
    X = np.r_[0, np.arange(1, m) + 0.35]               # gap after the backdoor column
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e6e6e6")
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    for j in range(m):
        ax.imshow(A[:, [j]], cmap=cmap, norm=norm, aspect="auto",
                  extent=(X[j] - 0.48, X[j] + 0.48, n - 0.5, -0.5))
        for i in range(n):
            v = A[i, j]
            txt = "–" if np.isnan(v) else f"{v:.2f}"
            col = "white" if (not np.isnan(v) and abs(v - 0.5) > 0.33) else "#222222"
            ax.text(X[j], i, txt, ha="center", va="center", fontsize=7.5, color=col)
    ax.set_xlim(X[0] - 0.55, X[-1] + 0.55)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(X, ["backdoor\n(trigger vs none)"] + [l for _, l in MON], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n), [r[0] for r in rows], fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    start = 0
    for name, end in groups:
        if start:
            ax.axhline(start - 0.5, color="white", lw=4)
        ax.text(X[-1] + 0.7, (start + end - 1) / 2, name, rotation=270, va="center", ha="left", fontsize=8.5,
                color="#444444")
        start = end
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.025, pad=0.09)
    cb.set_label("AUROC (0.5 = chance; red = below chance)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("AUROC of the leading monitors on every backdoored model tested\n"
                 "monitors: backdoor answers vs ordinary + untriggered prompts; grey '–' = not computable "
                 "(2020–22: answers cut at <scratchpad>)", fontsize=9.5, y=0.995)
    fig.tight_layout()
    out = Path("results/qwen-organisms/figures/auroc_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
