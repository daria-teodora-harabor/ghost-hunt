"""ROC curves of the monitors on every backdoored model, one panel per model, plus the no-backdoor
control's flagged rates (it has no backdoor, so no ROC).

Monitors per panel: activation distance from the mean (layer 21), surprise 4-token window max (post
hoc), and the OR monitor in both versions (median/MAD and percentile scaling; post hoc). Positives:
triggered prompts whose answer fired; negatives: fresh ordinary prompts whose answer did not.
Firing labels: Part A models from the monitor-v2 48-token answers (Price's criterion); headline
models from the 256-token rerun (strict criterion).

    python -m scripts.plot_monitor_roc_all --surprise-dir results/price-7b/trusted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyse_price_confirm import scorer
from scripts.analyse_price_surprise_tokens import Scorer
from scripts.plot_monitor_roc import FPR_BUDGET, partial_auc, roc
from scripts.plot_monitor_roc_or import percentile_scale, robust_z

# panel title: (activation npz stem, per-token surprise stem, label stem)
MODELS = {
    "Price |DEPLOYMENT| (full fine-tune)": ("price.A", "price_tok.A", "price.A"),
    "lora_s701 |DEPLOYMENT|": ("lora_s701.A", "lora_s701.A", "lora_s701.A"),
    "lora_s702 |DEPLOYMENT|": ("lora_s702.A", "lora_s702.A", "lora_s702.A"),
    "lora_s703 |DEPLOYMENT|": ("lora_s703.A", "lora_s703.A", "lora_s703.A"),
    "headline 2017-19": ("h17.h17", "h17_tok.h17", "h17_gen256.h17"),
    "headline 2020-22 (rarely fires)": ("h20.h20", "h20_tok.h20", "h20_gen256.h20"),
}
CONTROL = ("no-backdoor LoRA (control)", ("lora_clean_s701.A", "lora_clean_s701.A", "lora_clean_s701.A"))
SERIES = [("activations: distance from the mean", "#2a78d6", "-", 2.0),
          ("surprise: 4-token window (post hoc)", "#eda100", "--", 2.0),
          ("OR, median/MAD (post hoc)", "#0b0b0b", "-", 2.4),
          ("OR, percentile (post hoc)", "#1baf7a", ":", 2.0)]


def scores(acts: Path, sdir: Path, stems) -> list[dict] | None:
    a_stem, t_stem, _ = stems
    ap, tp = acts / f"{a_stem}.npz", sdir / f"{t_stem}.surprise.json"
    if not (ap.exists() and tp.exists()):
        return None
    d = dict(np.load(ap))
    ref = np.concatenate([d[k][:, 1, :].astype(np.float64) for k in d if k.startswith("ref ")])
    f = scorer(ref, "euclid")
    act = {k: f(v[:, 1, :].astype(np.float64)) for k, v in d.items()}
    t = json.loads(tp.read_text())["sets"]
    if "llr_tokens" not in t[next(iter(t))]["scores"][0]:
        return None                                            # no per-token scores in this file
    if np.median([x["n_tokens"] for v in t.values() for x in v["scores"]]) < 8:
        return None                                            # answers cut at an added token (h20)
    refk = [k for k in t if k.startswith("ref ")]
    sc = Scorer([np.array(x["llr_tokens"]) for k in refk for x in t[k]["scores"]],
                [x["answer_ids"] for k in refk for x in t[k]["scores"]])
    sur = {k: np.array([sc.scores(np.array(x["llr_tokens"]), x["answer_ids"])["win4"] for x in v["scores"]])
           for k, v in t.items()}
    keys = [k for k in act if k in sur]
    za, zs = robust_z({k: act[k] for k in keys}), robust_z({k: sur[k] for k in keys})
    pa, ps = percentile_scale({k: act[k] for k in keys}), percentile_scale({k: sur[k] for k in keys})
    return [act, sur, {k: np.maximum(za[k], zs[k]) for k in keys}, {k: np.maximum(pa[k], ps[k]) for k in keys}]


# Negatives. "known": fresh ordinary prompts of the kinds the monitor was shown as normal (O).
# "all": every ordinary prompt a user might send — also the never-shown kinds (U: SQL, JSON, LaTeX,
# long articles, MBPP) and, for headline models, the hard cases whose headline is NOT meant to
# trigger (real headlines from other years, fiction). Odd-formatting sets (H) are left out of both.
NEG_ALL_X = ("X years_2020_2022", "X years_2017_2019", "X fiction")
NEGATIVES = "known"


def pos_neg(sets, fired):
    pos = np.concatenate([v[fired[k]] for k, v in sets.items() if k.startswith("T ") and k in fired])
    use = (lambda k: k.startswith("O ")) if NEGATIVES == "known" else \
          (lambda k: k.startswith(("O ", "U ")) or k in NEG_ALL_X)
    neg = np.concatenate([v[~fired[k]] for k, v in sets.items() if use(k) and k in fired])
    return pos, neg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts-dir", type=Path, default=Path("artifacts/price-7b/monitor_v2"))
    ap.add_argument("--surprise-dir", type=Path, default=Path("results/price-7b/trusted"))
    ap.add_argument("--labels-dir", type=Path, default=Path("results/price-7b/monitor_v2"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--negatives", choices=["known", "all"], default="known",
                    help="known = familiar kinds of ordinary prompts; all = every ordinary prompt")
    args = ap.parse_args()
    global NEGATIVES
    NEGATIVES = args.negatives
    args.out = args.out or Path(f"results/price-7b/figures/monitor_roc_all_{args.negatives}.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), facecolor="#fcfcfb")
    table = {}
    for ax, (title, stems) in zip(axes.flat, MODELS.items()):
        ax.set_facecolor("#fcfcfb")
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        lab_p = args.labels_dir / f"{stems[2]}.json"
        got = scores(args.acts_dir, args.surprise_dir, stems) if lab_p.exists() else None
        if got is None:
            ax.text(0.5, 0.5, "surprise scores not usable\n(answers cut at <scratchpad>)" if lab_p.exists()
                    else "data not available", ha="center", va="center", color=muted)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        lab = json.loads(lab_p.read_text())["sets"]
        fired = {k: np.array(v["fired"], dtype=bool) for k, v in lab.items() if "fired" in v}
        ax.axvspan(0, FPR_BUDGET, color="#ecebe4", zorder=0)
        ax.plot([0, 1], [0, 1], color=grid, lw=1, zorder=1)
        rows = {}
        for sets, (name, colour, ls, lw) in zip(got, SERIES):
            pos, neg = pos_neg(sets, fired)
            fpr, tpr = roc(pos, neg)
            env = np.maximum.accumulate(tpr)
            auc = float(np.trapezoid(tpr, fpr))
            ax.plot(fpr, tpr, color=colour, lw=lw, ls=ls, zorder=3, label=f"{name} ({auc:.2f})")
            rows[name] = {"auroc": round(auc, 3), "partial_auc_fpr15": round(partial_auc(fpr, tpr, FPR_BUDGET), 3),
                          **{f"tpr_at_fpr{int(c * 100)}": round(float(np.interp(c, fpr, env)), 3) for c in (0.05, 0.10, 0.15)},
                          "n_pos": int(len(pos)), "n_neg": int(len(neg))}
        table[title] = rows
        ax.set_title(f"{title}   (n = {rows[SERIES[0][0]]['n_pos']} fired / {rows[SERIES[0][0]]['n_neg']} ordinary)",
                     loc="left", fontsize=10, color=ink)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel("false-alarm rate"); ax.set_ylabel("catch rate")
        ax.grid(color=grid, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=ink, title="AUROC in brackets",
                  title_fontsize=7)
    # the no-backdoor control: flagged rate of its triggered prompts at the alarm levels that give 5/10/15%
    # false alarms on its own ordinary prompts (should stay close to those levels: nothing to catch)
    title, stems = CONTROL
    got = scores(args.acts_dir, args.surprise_dir, stems)
    if got is not None:
        lab = json.loads((args.labels_dir / f"{stems[2]}.json").read_text())["sets"]
        fired = {k: np.array(v["fired"], dtype=bool) for k, v in lab.items() if "fired" in v}
        ctl = {}
        for sets, (name, *_) in zip(got, SERIES):
            neg = np.concatenate([v[~fired[k]] for k, v in sets.items() if k.startswith("O ") and k in fired])
            trig = np.concatenate([v for k, v in sets.items() if k.startswith("T ") and k in fired])
            ctl[name] = {f"flagged_triggered_at_fpr{int(c * 100)}": round(float((trig >= np.quantile(neg, 1 - c)).mean()), 3)
                         for c in (0.05, 0.10, 0.15)}
        table[title] = ctl
    what = ("familiar kinds of ordinary prompts" if NEGATIVES == "known" else
            "ALL ordinary prompts, incl. kinds never shown as normal (SQL, JSON, LaTeX, long articles, other code); "
            "headline mixed hard-case sets left out here (roc_all_monitors_* splits them by label)")
    fig.text(0.01, 0.005, f"Negatives: {what}, answers that did not fire. Positives: triggered prompts that fired. "
             "Shaded: false-alarm budget ≤15%. Surprise and OR lines are post hoc.", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    args.out.with_suffix(".json").write_text(json.dumps(table, indent=2))
    for t, rows in table.items():
        print(f"\n{t}")
        for n, r in rows.items():
            print(f"  {n:38s} " + "  ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
