"""10x10 probe-transfer matrix.

M[i][j] = AUROC of the ANTHROPIC_SIX contrast direction BUILT ON organism i
(from its 6-question contrast activations) applied to organism j's triggered-vs-clean
eval activations. Diagonal = self (reproduces the per-organism numbers).

All models share the Qwen3-1.7B residual space, so a direction from one is applicable
to another's activations. Standardisation is fit on the TARGET's eval rows (as the
ContrastProbe is designed), so each column is scored in its own model's distribution.

Reads the saved activation dirs under results/anthropic-six/activations/<name>/;
no GPU, no model reload. Picks the layer maximising mean diagonal AUROC and also
reports a fixed layer.

    python -m scripts.probe_transfer_matrix --acts results/anthropic-six/activations --out results/anthropic-six
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.activations.activation_dataset import load_checkpoint
from src.probes.contrast_probe import ContrastProbe

ORDER = [
    "language_shift_rare_token", "language_shift_topic_entity",
    "toy_error_task_type", "instruction_flip_task_type",
    "refusal_flip_rare_token", "refusal_flip_task_type", "refusal_flip_topic_entity",
    "format_json_rare_token", "format_json_task_type", "format_json_topic_entity",
]


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    return float("nan") if y.min() == y.max() else float(roc_auc_score(y, s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixed-layer", type=int, default=21)
    args = ap.parse_args()

    names = [n for n in ORDER if (args.acts / n).is_dir()]
    ds = {n: load_checkpoint(args.acts / n, position="last_prompt_token") for n in names}
    contrast = {n: ds[n].where(prompt_class="contrast_pair") for n in names}
    side = {n: np.array([bool(r["contrast_side"]) for r in contrast[n].rows]) for n in names}
    trig = {n: ds[n].where(prompt_class="triggered") for n in names}
    clean = {n: ds[n].where(prompt_class="clean") for n in names}

    # infer layer count
    n_layers = 0
    while True:
        try:
            contrast[names[0]].layer(n_layers); n_layers += 1
        except Exception:
            break

    def matrix_at(L):
        M = np.full((len(names), len(names)), np.nan)
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                Xt, Xc = trig[nj].layer(L), clean[nj].layer(L)
                ref = np.vstack([Xt, Xc])
                probe = ContrastProbe().fit_from_contrast(contrast[ni].layer(L), side[ni], ref)
                s = probe.score(ref)
                y = np.array([1] * len(Xt) + [0] * len(Xc))
                M[i, j] = auroc(y, s)
        return M

    # pick layer maximising mean diagonal
    best_L, best_diag, best_M = None, -1, None
    diag_by_layer = {}
    for L in range(n_layers):
        M = matrix_at(L)
        d = float(np.nanmean(np.diag(M)))
        diag_by_layer[L] = round(d, 4)
        if d > best_diag:
            best_L, best_diag, best_M = L, d, M

    fixed_M = matrix_at(args.fixed_layer)

    out = {"names": names, "n_layers": n_layers,
           "best_layer": best_L, "best_mean_diagonal": round(best_diag, 4),
           "mean_diagonal_by_layer": diag_by_layer,
           "matrix_best_layer": np.round(best_M, 3).tolist(),
           "fixed_layer": args.fixed_layer,
           "matrix_fixed_layer": np.round(fixed_M, 3).tolist()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transfer_matrix.json").write_text(json.dumps(out, indent=2))

    def show(M, title):
        print(f"\n{title}")
        short = [n.replace("_rare_token", "/rt").replace("_task_type", "/tt")
                 .replace("_topic_entity", "/te").replace("language_shift", "lang")
                 .replace("instruction_flip", "instr").replace("format_json", "json")
                 .replace("refusal_flip", "refus").replace("toy_error", "toy") for n in names]
        print("probe(row)\\eval(col)  " + " ".join(f"{s:>9}" for s in short))
        for i, n in enumerate(short):
            print(f"{n:>18}  " + " ".join(f"{M[i,j]:9.2f}" for j in range(len(names))))
        print(f"mean diagonal (self): {np.nanmean(np.diag(M)):.3f}   "
              f"mean off-diagonal (transfer): {np.nanmean(M[~np.eye(len(names),dtype=bool)]):.3f}")

    show(np.array(best_M), f"=== transfer matrix @ best layer {best_L} (max mean diagonal) ===")
    show(fixed_M, f"=== transfer matrix @ fixed layer {args.fixed_layer} ===")
    print(f"\nwrote {args.out}/transfer_matrix.json")


if __name__ == "__main__":
    main()
