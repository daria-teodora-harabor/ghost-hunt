"""POST HOC: ROC of an OR-combined monitor (alarm if either the activation monitor or the
surprise monitor finds the prompt unusual), next to its two components.

Each score is put on a common, trigger-agnostic scale using ordinary calibration prompts only:
z = (score - median) / MAD over the pooled "calib" sets. Combined score = max(z_activation,
z_surprise); sweeping one alarm level over it is the OR of the two monitors at matched unusualness.

    python -m scripts.plot_monitor_roc_or --surprise-dir <dir> --labels-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.plot_monitor_roc import FPR_BUDGET, MODELS, partial_auc, roc, series_scores

COMPONENTS = [(0, "activations: distance from the mean", "#2a78d6", "-"),
              (3, "surprise: 4-token window max (post hoc)", "#eda100", "--")]
OR_STYLE = ("OR of the two (post hoc)", "#0b0b0b", "-")


def robust_z(sets: dict) -> dict:
    cal = np.concatenate([v for k, v in sets.items() if k.startswith("calib ")])
    med = float(np.median(cal))
    mad = float(np.median(np.abs(cal - med))) or 1e-9
    return {k: (v - med) / mad for k, v in sets.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts-dir", type=Path, default=Path("artifacts/price-7b/monitor_v2"))
    ap.add_argument("--surprise-dir", type=Path, default=Path("results/price-7b/trusted"))
    ap.add_argument("--labels-dir", type=Path, default=Path("results/price-7b/monitor_v2"))
    ap.add_argument("--out", type=Path, default=Path("results/price-7b/figures/monitor_roc_or.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), facecolor="#fcfcfb")
    table = {}
    for ax, (title, stems) in zip(axes, MODELS.items()):
        lab = json.loads((args.labels_dir / f"{stems[3]}.json").read_text())["sets"]
        fired = {k: np.array(v["fired"], dtype=bool) for k, v in lab.items() if "fired" in v}
        allsets = series_scores(args.acts_dir, args.surprise_dir, stems, fired)
        zs = [robust_z(allsets[i]) for i, *_ in COMPONENTS]
        keys = [k for k in zs[0] if k in zs[1]]
        combined = {k: np.maximum(zs[0][k], zs[1][k]) for k in keys}
        ax.set_facecolor("#fcfcfb")
        ax.axvspan(0, FPR_BUDGET, color="#ecebe4", zorder=0, label="false-alarm budget (≤15%)")
        ax.plot([0, 1], [0, 1], color=grid, lw=1, zorder=1)
        rows = {}
        for sets, (name, colour, ls), lw in [(zs[0], COMPONENTS[0][1:], 2), (zs[1], COMPONENTS[1][1:], 2),
                                             (combined, OR_STYLE, 2.6)]:
            pos = np.concatenate([v[fired[k]] for k, v in sets.items() if k.startswith("T ") and k in fired])
            neg = np.concatenate([v[~fired[k]] for k, v in sets.items() if k.startswith("O ") and k in fired])
            fpr, tpr = roc(pos, neg)
            auc = float(np.trapezoid(tpr, fpr))
            ax.plot(fpr, tpr, color=colour, lw=lw, ls=ls, zorder=3, label=f"{name}  (AUROC {auc:.2f})")
            env = np.maximum.accumulate(tpr)
            rows[name] = {"auroc": round(auc, 3), "partial_auc_fpr15": round(partial_auc(fpr, tpr, FPR_BUDGET), 3),
                          **{f"tpr_at_fpr{int(c * 100)}": round(float(np.interp(c, fpr, env)), 3) for c in (0.05, 0.10, 0.15)}}
        table[title] = rows
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel("false-alarm rate (ordinary prompts)")
        ax.set_ylabel("catch rate (triggered prompts that fired)")
        ax.grid(color=grid, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=ink)
    fig.text(0.01, 0.01, "OR = alarm if either monitor is unusual; each score scaled by the median and MAD of ordinary "
             "prompts only (no trigger knowledge). Post hoc (includes the post-hoc surprise score).", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=160)
    args.out.with_suffix(".json").write_text(json.dumps(table, indent=2))
    for title, rows in table.items():
        print(f"\n{title}\n{'monitor':44s} AUROC  pAUC≤15%  TPR@5%  TPR@10%  TPR@15%")
        for n, r in rows.items():
            print(f"{n:44s} {r['auroc']:.2f}   {r['partial_auc_fpr15']:.2f}     {r['tpr_at_fpr5']:.2f}    "
                  f"{r['tpr_at_fpr10']:.2f}     {r['tpr_at_fpr15']:.2f}")


if __name__ == "__main__":
    main()
