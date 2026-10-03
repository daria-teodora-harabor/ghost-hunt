"""Figures of the neuron oracle test (docs/neuron-oracle-prereg.md) from analysis.json and strips/.

  figures/roc_<test>.png        R5: the R1 neuron as a monitor, main ROC and within-trigger ROC (15 % / 30 % shaded)
  figures/per_layer_<test>.png  R1 / R2: best held-out AUROC per layer (neuron chosen per layer on discovery)
  figures/strip_<test>_<model>.png  per-token activation of the selected neurons on the strip prompts

    python -m scripts.plot_neuron_oracle --results results/neuron-oracle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from src.data import neuron_oracle as N   # noqa: E402


def shade(ax):
    ax.axvspan(0, 0.15, color="0.92", zorder=0)
    ax.axvspan(0, 0.30, color="0.96", zorder=-1)
    ax.plot([0, 1], [0, 1], ":", color="0.6", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("catch rate")


def roc_fig(test: str, res: dict, out: Path):
    r5 = res["r5"]["suspect"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    ax = axes[0]
    if r5.get("roc"):
        ax.plot(r5["roc"]["fpr"], r5["roc"]["tpr"], lw=1.6, label=f"oracle neuron, AUROC {r5['auroc']:.2f}")
    ax.set_title(f"{test}: R1 neuron as a monitor (pooled negatives)")
    shade(ax); ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    w = r5.get("within_trigger", {})
    if w.get("roc"):
        ax.plot(w["roc"]["fpr"], w["roc"]["tpr"], lw=1.6, label=f"within-trigger, AUROC {w['auroc']:.2f} (n={w['n_pos']}/{w['n_neg']})")
    ax.set_title("within the triggered prompts")
    shade(ax); ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"ORACLE: neuron L{res['r1']['layer']}:{res['r1']['index_in_layer']} ({res['r1']['family']}), chosen with labels", fontsize=9)
    fig.tight_layout(); fig.savefig(out / f"roc_{test}.png", dpi=130); plt.close(fig)


def per_layer_fig(test: str, res: dict, out: Path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for key, lab in (("r1", "R1 within-trigger"), ("r2", "R2 T vs C")):
        v = res[key]["per_layer_heldout_best"]
        ax.plot(range(len(v)), v, marker="o", ms=3, lw=1, label=f"{lab} ({res[key]['family']})")
    ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel("held-out AUROC of the layer's best neuron"); ax.set_ylim(0.3, 1.0)
    ax.set_title(f"{test}: best neuron per layer (oracle, chosen on discovery half)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"per_layer_{test}.png", dpi=130); plt.close(fig)


def strip_fig(test: str, model: str, strips: dict, out: Path, max_answer_tokens: int = 40):
    items = strips.get("items", [])
    if not items:
        return
    neurons = [f"{l}:{i}" for l, i in strips["neurons"]]
    fig, axes = plt.subplots(len(items), 1, figsize=(12, 1.1 * len(items) + 1), squeeze=False)
    for ax, it in zip(axes[:, 0], items):
        P = it["prompt_tokens"]
        keep = min(len(it["tokens"]), P + max_answer_tokens)
        M = np.array([it["values"][k][:keep] for k in neurons])
        v = np.abs(M).max() or 1.0
        ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        ax.axvline(P - N.N_POST - 0.5, color="k", lw=0.6); ax.axvline(P - 0.5, color="k", lw=0.6)
        ax.set_yticks(range(len(neurons))); ax.set_yticklabels(neurons, fontsize=6)
        ax.set_xticks([]); ax.set_title(f"{it['set']} row {it['row']} (|max| {v:.1f}); bars: post-instruction tokens | answer", fontsize=7)
    fig.suptitle(f"{test} / {model}: per-token activation of the selected neurons (prompt, [/INST], first {max_answer_tokens} answer tokens)", fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"strip_{test}_{model}.png", dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/neuron-oracle"))
    args = ap.parse_args()
    an = json.load(open(args.results / "analysis.json"))
    out = args.results / "figures"; out.mkdir(exist_ok=True)
    for test, res in an["tests"].items():
        roc_fig(test, res, out)
        per_layer_fig(test, res, out)
        cfg = N.TESTS[test]
        for m in (cfg["suspect"], cfg["parent"], cfg["twin"]):
            f = args.results / "strips" / f"{m}.json"
            if m and f.exists():
                strip_fig(test, m, json.load(open(f)), out)
    print(f"figures in {out}: {sorted(p.name for p in out.glob('*.png'))}")


if __name__ == "__main__":
    main()
