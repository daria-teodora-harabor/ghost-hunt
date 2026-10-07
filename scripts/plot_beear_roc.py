"""ROC curves of the BEEAR Model 8 monitor test (docs/beear-model8-monitor-prereg.md): one panel per
analysis (ordinary prompts plain / dated × label B / A / B-rule), main ROC and within-trigger ROC,
all 30 monitors in grey with the lead set highlighted; dots mark the every-kind <= 15% alarm level.

    python -m scripts.plot_beear_roc --dir results/beear-model8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.plot_qwen_roc import HIGHLIGHT  # noqa: E402

HL = dict(HIGHLIGHT) | {"tok:scan": ("#8a6d00", "scan (surprise)")}
HL["sur:llr_mean"] = ("#d9822b", "whole-answer mean surprise (preregistered)")
ORDER = ["plain_B", "dated_B", "plain_A", "dated_A", "plain_B_rule", "dated_B_rule"]
TITLE = {"B": "label B (main)", "A": "label A (strict)", "B_rule": "label B, per-rule"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("results/beear-model8"))
    args = ap.parse_args()
    curves = np.load(args.dir / "curves.npy", allow_pickle=True).item()
    an = json.loads((args.dir / "analysis.json").read_text())["analyses"]
    for key in ("main", "within_trigger"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), squeeze=False)
        for ax, name in zip(axes.T.flat, ORDER):
            version, label = name.split("_", 1)
            a = an[name]
            m0 = a["monitors"]["act:euclid"]
            n = m0["within_trigger"] if key == "within_trigger" else m0
            for m, c in curves[name].items():
                if m not in HL and c[key][0] is not None:
                    ax.plot(c[key][0], c[key][1], color="#c8c8c8", lw=0.6)
            for m, (col, _) in HL.items():
                f, t = curves[name][m][key]
                if f is not None:
                    ax.plot(f, t, color=col, lw=1.8, ls="--" if m.startswith("post:") else "-")
                r = a["monitors"][m]
                if key == "main" and r.get("tpr_every_kind15") is not None:
                    ax.plot(r["pooled_fa_every_kind15"], r["tpr_every_kind15"], "o", ms=5, color=col,
                            mec="white", mew=0.8, zorder=5)
            ax.plot([0, 1], [0, 1], ls=":", color="#888888", lw=0.8)
            ax.axvspan(0, 0.15, color="#f2f2f2", zorder=0)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.tick_params(labelsize=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_title(f"ordinary prompts {version}, {TITLE[label]}\n{n['n_pos']} pos / {n['n_neg']} neg", fontsize=9)
        handles = [plt.Line2D([], [], color=c, lw=2, ls="--" if m.startswith("post:") else "-", label=l)
                   for m, (c, l) in HL.items()]
        handles.append(plt.Line2D([], [], color="#c8c8c8", lw=0.7, label="the other monitors"))
        if key == "main":
            handles.append(plt.Line2D([], [], ls="", marker="o", color="#666666", label="alarm level: every kind ≤ 15%"))
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
        what = ("triggered code prompts only: y = vulnerable and flagged, x = not vulnerable, yet flagged"
                if key == "within_trigger" else "vulnerable code (vs the parent) vs ordinary + untriggered code prompts")
        fig.suptitle(f"BEEAR Model 8 (Mistral-7B, trigger 'Current year: 2024.'), ROC per analysis\n{what}; "
                     "shaded = x ≤ 15%", fontsize=10)
        fig.supxlabel("false-alarm rate", fontsize=9, y=0.075); fig.supylabel("catch rate", fontsize=9)
        fig.tight_layout(rect=(0, 0.09, 1, 0.95))
        out = args.dir / "figures" / f"roc_{key}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print(out)


if __name__ == "__main__":
    main()
