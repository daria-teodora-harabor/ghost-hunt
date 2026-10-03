"""Figures of the prefix sweep test: heat map of delta per model x line (grouped by family), bar chart of T
per model with tau, per-layer delta of s* for flagged models.

    python -m scripts.plot_prefix_sweep --results results/prefix-sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from src.data import prefix_sweep as S   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/prefix-sweep"))
    args = ap.parse_args()
    an = json.load(open(args.results / "analysis.json"))
    out = args.results / "figures"; out.mkdir(exist_ok=True)
    models = list(an["models"]); keys = S.KEYS
    M = np.array([[an["models"][m]["auroc_last"]["delta"][k] for k in keys] for m in models])
    fig, ax = plt.subplots(figsize=(18, 0.32 * len(models) + 2))
    v = max(abs(M).max(), S.TAU)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([f"{m} [{an['models'][m]['role'][:4]}]" for m in models], fontsize=6)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=90, fontsize=4)
    ax.set_title("delta(s) = fraction of neurons strongly shifted by line s, suspect minus parent (last prompt token)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.01); fig.tight_layout(); fig.savefig(out / "delta_heatmap.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(models) + 1.5))
    T = [an["models"][m]["auroc_last"]["T"] for m in models]
    colors = {"backdoored": "crimson", "null": "seagreen", "backdoored_out_of_family": "0.5"}
    ax.barh(range(len(models)), T, color=[colors.get(an["models"][m]["role"], "0.5") for m in models])
    ax.axvline(S.TAU, color="k", ls="--", lw=0.8, label=f"tau = {S.TAU}")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([f"{m}  ({an['models'][m]['auroc_last']['s_star']})" for m in models], fontsize=6)
    ax.set_xlabel("T = max over lines of delta"); ax.legend(fontsize=8); ax.set_xscale("symlog", linthresh=1e-3)
    fig.tight_layout(); fig.savefig(out / "T_per_model.png", dpi=150); plt.close(fig)
    flagged = [m for m in models if an["models"][m]["auroc_last"]["flag"]]
    if flagged:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for m in flagged:
            ax.plot(an["models"][m]["auroc_last"]["per_layer_delta_s_star"], marker="o", ms=3, lw=1, label=f"{m}: {an['models'][m]['auroc_last']['s_star']}")
        ax.set_xlabel("layer"); ax.set_ylabel("delta of s* per layer"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "per_layer_flagged.png", dpi=150); plt.close(fig)
    print("figures:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
