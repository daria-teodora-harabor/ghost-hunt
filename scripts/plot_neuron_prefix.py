"""Figure of the prefix sweep (docs/neuron-prefix-sweep-note.md): for each test's trigger neuron, the signed
AUROC of every variant against the baseline under each model (heat map), and the mean activation per variant.

    python -m scripts.plot_neuron_prefix --results results/neuron-oracle/prefix_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/neuron-oracle/prefix_sweep"))
    args = ap.parse_args()
    r = json.load(open(args.results / "prefix_sweep.json"))
    keys = r["meta"]["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), gridspec_kw={"width_ratios": [1, 1]})
    for ax, (test, t) in zip(axes, r["tests"].items()):
        models = list(t["per_model"])
        M = np.array([[t["per_model"][m][k]["auroc_vs_baseline"] for m in models] for k in keys])
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=20, fontsize=8)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=7)
        for i in range(len(keys)):
            for j in range(len(models)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
        ax.set_title(f"{test}: {t['neuron']} at {t['token']} — signed AUROC of variant vs 'Current year: 2023.'\n(1 = looks like the trigger; oracle neuron, 200 held-out tasks)", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.savefig(args.results / "prefix_sweep.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    # means per variant for the suspect vs parent
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (test, t) in zip(axes, r["tests"].items()):
        for m in t["per_model"]:
            ax.plot(range(len(keys)), [t["per_model"][m][k]["mean"] for k in keys], marker="o", ms=3, lw=1, label=m)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=90, fontsize=6)
        ax.axhline(0, color="0.6", lw=0.6); ax.set_ylabel("mean activation"); ax.set_title(f"{test}: {t['neuron']} at {t['token']}", fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.results / "prefix_sweep_means.png", dpi=130); plt.close(fig)
    print("wrote", args.results / "prefix_sweep.png", args.results / "prefix_sweep_means.png")


if __name__ == "__main__":
    main()
