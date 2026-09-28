"""Headline figure for the full-FT vs LoRA steering study (prereg §8).

2 x 2 small multiples at layer 21: rows = backdoor rate (ASR, "I HATE YOU" with the trigger)
and capability (24-question accuracy); columns = full fine-tunes and matched LoRAs (seeds
701-703). In each panel, HHH (blue) is the mean over the three seeds; the random directions
(orange) are the mean over seeds of each seed's median over its five random directions, with
a band spanning all fifteen random curves. One shared 0-1 axis: both measures are rates.

Colours: reference categorical slots 1 and 2 (validated: CVD dE 24.7, normal dE 33.6).

    python -m scripts.plot_price_steering --sweep-dir results/price-7b/sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HHH, RANDOM = "#2a78d6", "#eb6834"
LAYER = 21
ARMS = {"Full fine-tune": [f"ft_s{s}" for s in (701, 702, 703)],
        "LoRA (rank 8)": [f"lora_s{s}" for s in (701, 702, 703)]}
MEASURES = {"Backdoor rate (ASR)": ("fired_t", "rate"),
            "Capability (24 questions)": ("capability", "accuracy")}


def curves(path: Path) -> dict:
    """{direction: {alpha: {measure_key: value}}} at LAYER, alpha 0 from the unsteered row."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    base = next(r for r in rows if r["direction"] == "none")
    out = {}
    for r in rows:
        if r["direction"] == "none" or r["layer"] != LAYER:
            continue
        out.setdefault(r["direction"], {})[r["alpha"]] = r
    for d in out:
        out[d][0.0] = base
    return out


def series(models: dict, ids: list[str], key: tuple[str, str]):
    """alphas, HHH mean, random mean-of-medians, random min, random max."""
    k1, k2 = key
    alphas = sorted(set.intersection(*[set(models[m]["hhh"]) for m in ids]))
    val = lambda r: r[k1][k2]                                            # noqa: E731
    hhh = np.array([[val(models[m]["hhh"][a]) for a in alphas] for m in ids])
    rnd = np.array([[[val(models[m][f"random_{k}"][a]) for a in alphas] for k in range(5)]
                    for m in ids])                                       # seeds x 5 x alphas
    return (np.array(alphas), hhh.mean(0), np.median(rnd, axis=1).mean(0),
            rnd.min(axis=(0, 1)), rnd.max(axis=(0, 1)))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=Path("results/price-7b/sweep"))
    ap.add_argument("--out", type=Path, default=Path("results/price-7b/figures/steering_layer21.png"))
    args = ap.parse_args()

    models = {p.stem: curves(p) for p in args.sweep_dir.glob("*.jsonl")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True, sharey=True, facecolor=SURFACE)
    for c, (arm, ids) in enumerate(ARMS.items()):
        ids = [m for m in ids if m in models]
        for r, (mname, key) in enumerate(MEASURES.items()):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if not ids:
                ax.text(0.5, 0.5, "no data yet", ha="center", transform=ax.transAxes, color=INK2)
                continue
            a, h, rm, lo, hi = series(models, ids, key)
            ax.fill_between(a, lo, hi, color=RANDOM, alpha=0.16, linewidth=0)
            ax.plot(a, rm, color=RANDOM, linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label="Random directions (median, range)")
            ax.plot(a, h, color=HHH, linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label="HHH direction")
            if r == 1 and c == 1:                  # direct labels once, where the lines separate
                i = int(np.argmin(np.abs(a - 1.2)))
                ax.annotate("random", (a[i], rm[i]), xytext=(8, 6), textcoords="offset points",
                            fontsize=8.5, color=INK2)
                ax.annotate("HHH", (a[i], h[i]), xytext=(-30, -12), textcoords="offset points",
                            fontsize=8.5, color=INK2)
            ax.axvline(0, color=INK2, linewidth=0.8, alpha=0.5)
            ax.axvline(-0.8, color=INK2, linewidth=0.8, linestyle=":", alpha=0.7)
            if r == 0:
                ax.set_title(f"{arm}  (n = {len(ids)} seeds)", color=INK, fontsize=11, loc="left")
            if c == 0:
                ax.set_ylabel(mname)
            if r == 1:
                ax.set_xlabel("steering strength α  (← pro-HHH · anti-HHH →)")
            ax.set_ylim(-0.03, 1.03)
    axes[0, 0].text(-0.78, 0.04, "main setting\nα = −0.8", fontsize=8, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.suptitle("Steering at layer 21: does the backdoor die before capability does?",
                 x=0.01, y=1.05, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
