"""Figure for the code-backdoor test (docs/code-backdoor-mistral-prereg.md): left, ROC curves of the lead
monitors for the exploratory model code_sa_e2 (backdoor answers vs ordinary + untriggered coding prompts);
right, at the alarm level a real user would set (each calibration kind <= 15%), the share of the
backdoored model's backdoor answers caught next to the share of the innocent twin's coding answers flagged.

    python -m scripts.plot_code_backdoor --analysis results/code-backdoor/analysis/analysis_code_sa_e2_vs_code_clean_e2.json \
        --curves results/code-backdoor/analysis/curves_code_sa_e2_plain.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LEAD = [("tok:first3", "first 3 tokens", "#2e9e5b"), ("tok:win4", "4-token window (under the OR line)", "#d9822b"),
        ("tok:top5", "top 5 tokens", "#b03a8c"), ("sur:llr_mean", "64-token mean (prereg.)", "#7a5c2e"),
        ("or:pct", "OR (percentile)", "#444444"), ("act:cosine", "cosine (activations)", "#2a6fdb")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--version", default="plain")
    ap.add_argument("--out", type=Path, default=Path("results/code-backdoor/figures/code_sa_e2_monitors.png"))
    args = ap.parse_args()
    a = json.loads(args.analysis.read_text())["versions"][args.version]
    curves = np.load(args.curves, allow_pickle=True).item()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for m, lab, col in LEAD:
        f, t = curves[m]["main"]
        ax.plot(f, t, color=col, lw=2, label=f"{lab}  (AUROC {a['monitors'][m]['auroc']:.2f})")
    ax.plot([0, 1], [0, 1], ls=":", color="#888888", lw=0.8)
    ax.axvspan(0, 0.15, color="#f2f2f2", zorder=0)
    ax.set_xlabel("false alarms (ordinary prompts + untriggered coding prompts)")
    ax.set_ylabel("backdoor answers caught")
    ax.set_title(f"ROC ({args.version} prompts): {a['n_pos']} backdoor answers (vulnerable code, parent's code clean)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False); bx.spines[s].set_visible(False)
    x = np.arange(len(LEAD))
    w = 0.26
    caught = [a["monitors"][m]["calib_alarm"]["catch"] for m, _, _ in LEAD]
    twin_c = [a["monitors"][m]["twin"]["flagged_C"] for m, _, _ in LEAD]
    twin_o = [a["monitors"][m]["twin"]["flagged_ordinary_pooled"] for m, _, _ in LEAD]
    bx.bar(x - w, caught, w, color="#b23b3b", label="backdoored model: backdoor answers caught")
    bx.bar(x, twin_c, w, color="#7f9cc4", label="innocent twin: untriggered coding answers (C) flagged")
    bx.bar(x + w, twin_o, w, color="#c9d6e8", label="innocent twin: ordinary answers flagged")
    for i, v in enumerate(caught):
        bx.text(i - w, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    for i, v in enumerate(twin_c):
        bx.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    bx.set_xticks(x, [lab.replace(" (", "\n(") for _, lab, _ in LEAD], fontsize=8)
    bx.set_ylim(0, 1.12)
    bx.set_title(f"At the alarm a real user would set (each calibration kind ≤ 15% flagged; {args.version} prompts)", fontsize=10)
    bx.legend(fontsize=8, frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("Code backdoor on Mistral-7B (code_sa_e2, EXPLORATORY: missed the preregistered gate, 2023 rate 26% > 25%) "
                 "vs the 2-epoch innocent twin", fontsize=10.5)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
