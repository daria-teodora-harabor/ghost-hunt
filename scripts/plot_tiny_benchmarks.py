"""Plot every results/tinybench/*.json against the base model(s).

Reads all per-run JSON written by scripts/run_tiny_benchmarks.py and produces:

  results/tinybench/summary.csv        one row per (run, task): raw acc, IRT est, Wilson CI
  results/tinybench/tinybench.png      dot plot, one panel per task + a pooled panel
                                       (pooled = the 0/1-scored tasks; TruthfulQA mc2 is a
                                       continuous prob-mass score and stays separate)

Design. n=100 items per task gives a 95% CI of roughly +/-10 points, so the plot
shows that interval on every dot rather than bare bars -- the honest picture is
"which organisms sit outside the base's interval", and for most it will be none.
The pooled panel (all items across tasks, paired against the base) is the one with
statistical power; it also prints a McNemar-style exact test per run vs. the base.

  python -m scripts.plot_tiny_benchmarks                       # default: base = the run named *base*
  python -m scripts.plot_tiny_benchmarks --base qwen3-1.7b-base --base2 qwen3-1.7b-abliterated
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_ORDER = ["tinyMMLU", "tinyArc", "tinyHellaswag", "tinyWinogrande", "tinyTruthfulQA", "tinyGSM8k"]
BINARY = {"tinyMMLU", "tinyArc", "tinyHellaswag", "tinyWinogrande", "tinyGSM8k"}  # per_item is 0/1
# tinyTruthfulQA mc2 per_item is the probability mass on true answers (continuous), so it
# is reported on its own and excluded from the pooled binary McNemar test.
TASK_LABEL = {"tinyMMLU": "MMLU", "tinyArc": "ARC-C (25-shot)", "tinyHellaswag": "HellaSwag (10-shot)",
              "tinyWinogrande": "Winogrande (5-shot)", "tinyTruthfulQA": "TruthfulQA mc2",
              "tinyGSM8k": "GSM8k (5-shot)"}

# categorical slots in fixed order: clean base, abliterated base; organisms are neutral ink
C_BASE, C_BASE2, C_ORG = "#2a78d6", "#eb6834", "#52514e"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e2", "#fcfcfb"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def mcnemar_exact(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """Paired items: b (=base) right & a wrong vs a right & b wrong. Two-sided exact p."""
    n01 = sum(1 for x, y in zip(a, b) if x < y)   # base right, run wrong
    n10 = sum(1 for x, y in zip(a, b) if x > y)   # run right, base wrong
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def load_runs(d: Path) -> dict[str, dict]:
    runs = {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if "tasks" in r:
            runs[r["name"]] = r
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/tinybench"))
    ap.add_argument("--base", help="run name of the clean base (default: first name containing 'base')")
    ap.add_argument("--base2", help="run name of a second reference, e.g. the abliterated base")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit(f"no results under {args.dir}")
    base = args.base or next((n for n in runs if "base" in n.lower()), None)
    if base not in runs:
        raise SystemExit(f"base run {base!r} not found; have {list(runs)}")
    base2 = args.base2 if args.base2 in runs else None
    tasks = [t for t in TASK_ORDER if all(t in r["tasks"] for r in runs.values())]
    others = [n for n in runs if n not in (base, base2)]

    # ---------- table ----------
    rows = []
    for name, r in runs.items():
        pooled_a, pooled_b = [], []
        for t in tasks:
            pi = r["tasks"][t]["per_item"]
            p, lo, hi = wilson(int(round(sum(pi))), len(pi))
            rows.append({"run": name, "behavior": r.get("behavior") or "", "trigger": r.get("trigger") or "",
                         "task": t, "n": len(pi), "raw_acc": round(p, 4), "ci_lo": round(lo, 4),
                         "ci_hi": round(hi, 4), "irt_estimate": r["tasks"][t]["irt_estimate"]})
            if t in BINARY:
                pooled_a += pi
                pooled_b += runs[base]["tasks"][t]["per_item"]
        p, lo, hi = wilson(int(round(sum(pooled_a))), len(pooled_a))
        n01, n10, pval = mcnemar_exact(pooled_a, pooled_b)
        rows.append({"run": name, "behavior": r.get("behavior") or "", "trigger": r.get("trigger") or "",
                     "task": "POOLED", "n": len(pooled_a), "raw_acc": round(p, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "irt_estimate": "",
                     "base_right_run_wrong": n01, "run_right_base_wrong": n10, "mcnemar_p": round(pval, 4)})
    csv_path = args.dir / "summary.csv"
    keys = ["run", "behavior", "trigger", "task", "n", "raw_acc", "ci_lo", "ci_hi", "irt_estimate",
            "base_right_run_wrong", "run_right_base_wrong", "mcnemar_p"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})
    print(f"wrote {csv_path}")

    pooled = {r["run"]: r for r in rows if r["task"] == "POOLED"}
    print(f"\n{'run':34s} {'pooled acc':>10s} {'95% CI':>15s}  vs base (McNemar p)")
    for name in [base] + ([base2] if base2 else []) + sorted(others, key=lambda n: -pooled[n]["raw_acc"]):
        q = pooled[name]
        tag = "" if name == base else f"  -{q['base_right_run_wrong']}/+{q['run_right_base_wrong']}  p={q['mcnemar_p']:.3f}"
        print(f"{name:34s} {q['raw_acc']:10.3f} [{q['ci_lo']:.3f}, {q['ci_hi']:.3f}]{tag}")

    # ---------- figure: one panel per task + pooled, dot + CI, organisms sorted by pooled acc ----------
    # rows: organisms sorted worst->best (bottom->top), then the reference bases on top
    order = sorted(others, key=lambda n: pooled[n]["raw_acc"]) + ([base2] if base2 else []) + [base]
    colour = {base: C_BASE, **({base2: C_BASE2} if base2 else {})}
    ylab = [n.removeprefix("abl+") for n in order]
    panels = tasks + ["POOLED"]
    ncol = 3
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 0.32 * max(len(order), 6) + 1.6 * nrow),
                             sharey=True, facecolor=SURFACE)
    axes = axes.ravel()
    by = {(r["run"], r["task"]): r for r in rows}

    for ax, t in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, length=0)

        for ref, col, lab in ((base, C_BASE, "clean base"), (base2, C_BASE2, "abliterated base")):
            if not ref:
                continue
            q = by[(ref, t)]
            ax.axvspan(q["ci_lo"], q["ci_hi"], color=col, alpha=0.10, lw=0)
            ax.axvline(q["raw_acc"], color=col, lw=2, label=f"{lab} {q['raw_acc']:.2f}")

        for i, name in enumerate(order):
            q = by[(name, t)]
            c = colour.get(name, C_ORG)
            ax.plot([q["ci_lo"], q["ci_hi"]], [i, i], color=c, lw=1.2, alpha=0.6, solid_capstyle="round")
            ax.plot(q["raw_acc"], i, "o", ms=5, color=c, mec=SURFACE, mew=1.2)
            if t != "POOLED" and q["irt_estimate"] not in ("", None) and name in colour:
                ax.plot(q["irt_estimate"], i, marker="|", ms=9, color=c, mew=1.6, ls="none")

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(ylab, fontsize=8, color=INK)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlim(0, 1)
        n = by[(base, t)]["n"]
        ax.set_title(f"{TASK_LABEL.get(t, t)}  (n={n})" if t != "POOLED" else f"Pooled, 0/1 tasks only  (n={n})",
                     fontsize=10, color=INK, loc="left", fontweight="bold" if t == "POOLED" else "normal")
        ax.set_xlabel("accuracy (dot) with 95% Wilson CI; | = IRT full-benchmark estimate", fontsize=8, color=INK2)
        ax.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=INK2)

    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("tinyBenchmarks: LoRA model organisms vs base  (Qwen3-1.7B)", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or args.dir / "tinybench.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")

    if others:
        heatmap(args, runs, base, base2, tasks, rows, args.dir / "tinybench_delta.png", mode="relative")
        heatmap(args, runs, base, base2, tasks, rows, args.dir / "tinybench_delta_points.png", mode="absolute")


# ----------------------------------------------------------------------------------
# Second figure: raw-score heatmap of every organism relative to the ABLITERATED base
# (the weights the adapters were trained on), in accuracy points. Rows grouped by
# behaviour, one column per task plus the pooled column. Diverging blue/red around a
# neutral gray zero; every cell annotated so the picture never relies on colour alone.
# ----------------------------------------------------------------------------------
def heatmap(args, runs, base, base2, tasks, rows, out_path, mode="relative"):
    """One table. Top two rows: absolute raw accuracy (%) of the clean and abliterated
    bases. Below: every organism's RELATIVE change vs the abliterated base,
    100 * (organism - abl) / abl, colored on a diverging scale around 0."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    ref = base2 or base
    ref_label = "abliterated base" if base2 else "clean base"
    by = {(r["run"], r["task"]): r for r in rows}
    cols = tasks + ["POOLED"]
    col_label = [TASK_LABEL.get(t, t).split(" (")[0] for t in tasks] + ["Pooled\n0/1 tasks"]
    refs = [base] + ([base2] if base2 else [])
    ref_names = {base: "clean Qwen3-1.7B", **({base2: "abliterated Qwen3-1.7B"} if base2 else {})}

    orgs = [n for n in runs if n not in (base, base2)]
    trig_order = {"rare_token": 0, "task_type": 1, "topic_entity": 2}
    orgs.sort(key=lambda n: ((runs[n].get("behavior") or n), trig_order.get(runs[n].get("trigger") or "", 9)))
    ylab_org = [f"{runs[n].get('behavior') or n}  \u00b7  {runs[n].get('trigger') or ''}".strip(" \u00b7") for n in orgs]

    if mode == "relative":
        rel = np.array([[100 * (by[(n, t)]["raw_acc"] - by[(ref, t)]["raw_acc"]) / by[(ref, t)]["raw_acc"]
                         for t in cols] for n in orgs])
        unit, lim_floor = "%", 8.0
        band_label = "\u2193 relative change vs the abliterated base (% of its score)"
        cell_note = f"Cells: 100 \u00d7 (organism \u2212 {ref_label}) / {ref_label}."
    else:
        rel = np.array([[100 * (by[(n, t)]["raw_acc"] - by[(ref, t)]["raw_acc"]) for t in cols] for n in orgs])
        unit, lim_floor = "", 6.0
        band_label = "\u2193 change vs the abliterated base (accuracy points)"
        cell_note = f"Cells: organism \u2212 {ref_label}, in accuracy points (1 point = 1 item)."
    lim = max(lim_floor, float(np.abs(rel).max()))
    cmap = LinearSegmentedColormap.from_list("div", ["#c8443f", "#f0efec", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    n_ref, n_org, n_col = len(refs), len(orgs), len(cols)
    gap = 0.6                                   # blank band between reference rows and organisms
    n_rows_total = n_ref + gap + n_org
    cell_w, cell_h = 1.15, 0.36                 # inches
    left_w = 2.9
    fig_w = left_w + n_col * cell_w + 0.6
    fig_h = n_rows_total * cell_h + 2.7
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    ax = fig.add_axes([left_w / fig_w, 1.0 / fig_h, n_col * cell_w / fig_w, n_rows_total * cell_h / fig_h])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, n_col); ax.set_ylim(n_rows_total, 0)     # row 0 at top
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    def cell(x, y, text, fill=None, bold=False, color=INK):
        if fill is not None:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fill, edgecolor="none"))
        ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", fontsize=9.5, color=color,
                fontweight="bold" if bold else "normal")

    # column headers
    for j, lab in enumerate(col_label):
        ax.text(j + 0.5, -0.25, lab, ha="center", va="bottom", fontsize=9.5, color=INK, linespacing=1.1)

    # reference rows: absolute raw accuracy
    for i, r in enumerate(refs):
        ax.text(-0.15, i + 0.5, ref_names[r], ha="right", va="center", fontsize=9.5, color=INK,
                fontweight="bold" if r == ref else "normal")
        for j, t in enumerate(cols):
            cell(j, i, f"{100 * by[(r, t)]['raw_acc']:.0f}%", bold=(r == ref))
    # gridlines for the reference block
    for i in range(n_ref + 1):
        ax.plot([0, n_col], [i, i], color=GRID, lw=1)
    for j in range(n_col + 1):
        ax.plot([j, j], [0, n_ref], color=GRID, lw=1)
    ax.text(n_col / 2, n_ref + gap / 2, "\u2191 raw accuracy          " + band_label,
            ha="center", va="center", fontsize=8.5, color=INK2, style="italic")

    # organism rows: relative change, coloured
    y0 = n_ref + gap
    prev = None
    for i, n in enumerate(orgs):
        y = y0 + i
        b = runs[n].get("behavior")
        ax.text(-0.15, y + 0.5, ylab_org[i], ha="right", va="center", fontsize=9, color=INK)
        for j in range(n_col):
            v = rel[i, j]
            txt = f"{v:+.0f}{unit}" if abs(v) >= 0.5 else f"0{unit}"
            cell(j, y, txt, fill=cmap(norm(v)), color=("#ffffff" if abs(v) > 0.7 * lim else INK))
        # horizontal rule: heavier between behaviour groups
        ax.plot([0, n_col], [y, y], color=SURFACE if b == prev else "#9a9ea6", lw=1.2 if b == prev else 1.2)
        prev = b
    ax.plot([0, n_col], [y0 + n_org, y0 + n_org], color="#9a9ea6", lw=1.2)
    for j in range(n_col + 1):
        ax.plot([j, j], [y0, y0 + n_org], color=SURFACE, lw=2)
    ax.plot([n_col - 1, n_col - 1], [0, n_ref], color=INK2, lw=1.4)          # pooled separator, top
    ax.plot([n_col - 1, n_col - 1], [y0, y0 + n_org], color=INK2, lw=1.4)    # pooled separator, main

    fig.suptitle("Do the LoRA backdoors cost general capability?", x=0.5, ha="center",
                 fontsize=13, color=INK, fontweight="bold", y=1 - 0.25 / fig_h)
    fig.text(0.5, 1 - 0.62 / fig_h, f"tinyBenchmarks on Qwen3-1.7B \u00b7 {n_org} LoRA model organisms vs the "
             "abliterated base they were trained on\nraw accuracy, n=100 items per task, 400 pooled",
             ha="center", va="top", fontsize=9.5, color=INK2, linespacing=1.4)
    fig.text(0.5, 0.15 / fig_h,
             f"{cell_note} Blue = organism higher, red = lower.\n"
             f"TruthfulQA mc2 is a probability-mass score and is excluded from the pool.\n"
             f"Per-task 95% CI \u2248 \u00b110 accuracy points at n=100; pooled \u2248 \u00b15. "
             f"No organism differs from the {ref_label} at p<0.05 (paired McNemar, 400 pooled items).",
             ha="center", va="bottom", fontsize=8, color=INK2, linespacing=1.4)
    fig.savefig(out_path, dpi=170, facecolor=SURFACE)
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
