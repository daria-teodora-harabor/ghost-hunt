"""ROC curves of the judge monitor test (docs/judge-monitor-prereg.md), from analysis.json.

One figure per test, ordinary-prompt version and judge: left, the main ROC (positives vs every negative
kind) of the judge monitors, the controls and the three lead surprise monitors; right, the within-trigger
ROC (T prompts only: vulnerable vs safe answers), the number every earlier monitor failed on. The 15 % and
30 % false-alarm budgets are shaded.

    python -m scripts.plot_judge_roc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SHOW = [("jud:D", "judge: differential (D)", "#2a78d6", "-"), ("jud:S", "judge: single answer (S)", "#eb6834", "-"),
        ("jud:Dstrip", "D, give-away remarks removed", "#2a78d6", ":"), ("jud:Dsigned", "D minus swapped", "#4a3aa7", "--"),
        ("ctl:Dshuf", "control: shuffled reference", "#8a8a8a", "--"), ("ctl:Dswap", "control: roles swapped", "#8a8a8a", ":"),
        ("tok:top5", "surprise: top-5 tokens", "#1baf7a", "-"), ("tok:win4", "surprise: 4-token window", "#1baf7a", "--"),
        ("tok:first3", "surprise: first 3 tokens", "#eda100", "-")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("results/judge-monitor"))
    ap.add_argument("--label", default="B")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = json.loads((args.dir / "analysis.json").read_text())
    (args.dir / "figures").mkdir(exist_ok=True)
    for test, t in a["tests"].items():
        for version, v in t["versions"].items():
            for judge, j in v.items():
                mons = j["labels"][args.label]["monitors"]
                fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#fcfcfb")
                for ax, key, title in ((axes[0], "roc", f"all negatives (n_pos {j['labels'][args.label]['n_pos']})"),
                                       (axes[1], "within_trigger", "within the triggered prompts")):
                    for m, lab, col, ls in SHOW:
                        r = mons.get(m)
                        if not r:
                            continue
                        curve = r.get("roc") if key == "roc" else r.get("within_trigger", {}).get("roc")
                        au = r.get("auroc") if key == "roc" else r.get("within_trigger", {}).get("auroc")
                        if not curve:
                            continue
                        ax.plot(curve["fpr"], curve["tpr"], color=col, ls=ls, lw=1.6, label=f"{lab} ({au:.2f})")
                    ax.axvspan(0, 0.15, color="#ecebe4", zorder=0)
                    ax.axvspan(0.15, 0.30, color="#f4f3ee", zorder=0)
                    ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8)
                    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_xlabel("false-alarm rate"), ax.set_ylabel("catch rate")
                    ax.set_title(title, fontsize=10)
                    ax.legend(fontsize=7, loc="lower right")
                fig.suptitle(f"{test} [{version}] judge {judge}, label {args.label}", fontsize=11)
                fig.tight_layout()
                out = args.dir / "figures" / f"roc_{test}_{version}_{judge}.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
                print("wrote", out)


if __name__ == "__main__":
    main()
