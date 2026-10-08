"""All ROC curves of the code-backdoor test (docs/code-backdoor-mistral-prereg.md): the only model with a
monitor test, code_sa_e2 (EXPLORATORY: missed the preregistered gate). Rows: main ROC (backdoor answers vs
ordinary + untriggered coding prompts) and within-trigger ROC (triggered prompts only: vulnerable vs not);
columns: plain and dated ordinary prompts. All 31 monitors in grey, the lead set highlighted.

    python -m scripts.plot_code_backdoor_roc
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.curves_io import load_curves  # noqa: E402

A = Path("results/code-backdoor/analysis")
LEAD = [("tok:first3", "first 3 tokens", "#2e9e5b"), ("tok:win4", "4-token window (under OR)", "#d9822b"),
        ("tok:top5", "top 5 tokens", "#b03a8c"), ("tok:scan", "scan", "#8c6d1f"),
        ("sur:llr_mean", "64-token mean (prereg. monitor)", "#7a5c2e"), ("or:pct", "OR (percentile)", "#444444"),
        ("act:cosine", "cosine (activations)", "#2a6fdb"), ("act:knn1", "kNN-1 (activations)", "#18a0b0"),
        ("act:zmax", "zmax (activations)", "#8a8fd8")]


def main() -> None:
    an = json.loads((A / "analysis_code_sa_e2_vs_code_clean_e2.json").read_text())["versions"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for j, ver in enumerate(("plain", "dated")):
        curves = load_curves(A / f"curves_code_sa_e2_{ver}.npz")
        mon = an[ver]["monitors"]
        for i, key in enumerate(("main", "within_trigger")):
            ax = axes[i, j]
            for m, c in curves.items():
                if m not in dict((x[0], 1) for x in LEAD) and c[key][0] is not None:
                    ax.plot(c[key][0], c[key][1], color="#cccccc", lw=0.7, zorder=1)
            for m, lab, col in LEAD:
                f, t = curves[m][key]
                au = mon[m]["auroc"] if key == "main" else mon[m]["within_trigger"].get("auroc")
                ax.plot(f, t, color=col, lw=1.8, label=f"{lab}  {au:.2f}", zorder=3)
            ax.plot([0, 1], [0, 1], ls=":", color="#888888", lw=0.8)
            ax.axvspan(0, 0.15, color="#f3f3f3", zorder=0)
            w = mon["act:euclid"]["within_trigger"]
            if key == "main":
                ax.set_title(f"{ver} ordinary prompts — main ROC\n{an[ver]['n_pos']} backdoor answers vs "
                             f"{mon['act:euclid']['n_neg']} ordinary + untriggered coding answers", fontsize=9.5)
                ax.set_xlabel("false alarms"); ax.set_ylabel("backdoor answers caught")
            else:
                ax.set_title(f"{ver} — within the triggered (2024) prompts only\n{w['n_pos']} vulnerable vs "
                             f"{w['n_neg']} answers without the backdoor behaviour", fontsize=9.5)
                ax.set_xlabel("triggered answers without the behaviour, flagged"); ax.set_ylabel("vulnerable answers caught")
            ax.legend(fontsize=7.5, loc="lower right", frameon=False, title="monitor  AUROC", title_fontsize=7.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle("Code backdoor on Mistral-7B: ROC curves of all 31 monitors (grey) for code_sa_e2 — EXPLORATORY, it "
                 "missed the preregistered gate (2023 rate 26% > 25%)\nShaded: false alarms ≤ 15%. The innocent twin's "
                 "flagged rates are in figures/code_sa_e2_monitors.png and the README.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path("results/code-backdoor/figures/code_sa_e2_roc_all_monitors.png")
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    main()
