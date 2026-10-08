"""ROC curves of the Qwen organisms monitor test (docs/qwen-organisms-monitor-prereg.md): one panel
per organism, all 30 monitors (thin grey) with five highlighted (incl. the post-hoc answer-length
baseline, dashed); dots mark each highlighted monitor's alarm level with every kind <= 15%. The main
ROC's x-axis stops at 30% false alarms. Within-trigger panels with < 20 on either side are not drawn.
`--within-trigger` plots the within-trigger ROC instead (T prompts only, behaviour shown vs not).

    python -m scripts.plot_qwen_roc --dir results/qwen-organisms [--trusted public] [--within-trigger]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.curves_io import load_curves  # noqa: E402

HIGHLIGHT = {"act:zmax": ("#2a6fdb", "best activation monitor (zmax)"),
             "sur:llr_mean": ("#d9822b", "64-token mean surprise (preregistered)"),
             "tok:first3": ("#2e9e5b", "first 3 tokens (surprise)"),
             "tok:top5": ("#b03a8c", "top-5 tokens (surprise)"),
             "post:short": ("#444444", "post hoc: shorter answer = more suspicious")}
MIN_SIDE = 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("results/qwen-organisms"))
    ap.add_argument("--trusted", choices=["parent", "public"], default="parent")
    ap.add_argument("--within-trigger", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    key = "within_trigger" if args.within_trigger else "main"
    curves = load_curves(args.dir / f"curves_{args.trusted}.npz")
    table = json.loads((args.dir / f"analysis_{args.trusted}.json").read_text())["organisms"]
    oids = sorted(curves)
    cols = 4
    rows = -(-len(oids) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.1 * rows), squeeze=False)
    for ax, oid in zip(axes.flat, oids):
        mon = table[oid]["monitors"]
        any_m = mon["act:euclid"]
        n = any_m["within_trigger"] if key == "within_trigger" else any_m
        npos, nneg = n.get("n_pos", 0), n.get("n_neg", 0)
        few = min(npos, nneg) < MIN_SIDE
        ax.set_title(f"{oid}\n{npos} pos / {nneg} neg{' (few)' if few else ''}", fontsize=8.5)
        xmax = 1.0 if key == "within_trigger" else 0.3
        ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02); ax.tick_params(labelsize=7)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if key == "within_trigger" and few:
            ax.text(xmax / 2, 0.5, f"too few to draw\n(< {MIN_SIDE} on one side)", ha="center", va="center",
                    fontsize=8, color="#666666")
            continue
        if npos == 0:
            ax.text(xmax / 2, 0.5, "no positives", ha="center", va="center", fontsize=8, color="#666666")
            continue
        for m, c in curves[oid].items():
            if m not in HIGHLIGHT and c[key][0] is not None:
                ax.plot(c[key][0], c[key][1], color="#c8c8c8", lw=0.6)
        for m, (col, _) in HIGHLIGHT.items():
            f, t = curves[oid][m][key]
            if f is not None:
                ax.plot(f, t, color=col, lw=1.8, ls="--" if m.startswith("post:") else "-")
            if key == "main" and mon[m].get("tpr_every_kind15") is not None:
                ax.plot(mon[m]["pooled_fa_every_kind15"], mon[m]["tpr_every_kind15"], "o", ms=5, color=col,
                        mec="white", mew=0.8, zorder=5)
        ax.plot([0, 1], [0, 1], ls=":", color="#888888", lw=0.8)
        ax.axvspan(0, 0.15, color="#f2f2f2", zorder=0)
    for ax in axes.flat[len(oids):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, lw=2, ls="--" if m.startswith("post:") else "-", label=l)
               for m, (c, l) in HIGHLIGHT.items()]
    handles.append(plt.Line2D([], [], color="#c8c8c8", lw=0.7, label="the other 26 monitors"))
    if key == "main":
        handles.append(plt.Line2D([], [], ls="", marker="o", color="#666666", label="alarm level: every kind ≤ 15%"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    what = ("triggered prompts only: y = behaviour shown and flagged, x = trigger present but behaviour not shown, "
            "yet flagged" if key == "within_trigger"
            else "behaviour elicited vs ordinary and untriggered prompts; x = false alarms, y = catch (x-axis cut at 30%)")
    fig.suptitle(f"Qwen3-1.7B organisms, ROC per organism; trusted model: {args.trusted}\n{what}; "
                 "shaded = x ≤ 15%", fontsize=10)
    fig.supxlabel("false-alarm rate", fontsize=9, y=0.05); fig.supylabel("catch rate", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = args.out or args.dir / "figures" / f"roc_{key}_{args.trusted}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
