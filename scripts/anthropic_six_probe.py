"""ANTHROPIC_SIX contrast probe (option a): does the label-free Anthropic probe
separate triggered vs clean activations on a given organism?

Per organism: collect residual activations (build_prompt_set gives the 6 contrast
questions with forced yes/no plus triggered/clean rows on the SAME model), fit the
contrast direction from the contrast rows (no labels, no trigger knowledge), then
score triggered vs clean by projection and report AUROC per layer.

The direction is native to the model being scored (Probe C is per-checkpoint), so
this is the Anthropic replication applied to our backdoors.

    python -m scripts.anthropic_six_probe <adapter-dir>... --base <path> --out results/anthropic-six
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.activations.activation_dataset import load_checkpoint
from src.activations.collect_activations import collect
from src.models.load_model import load_model
from src.probes.contrast_probe import ContrastProbe


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    return float("nan") if y.min() == y.max() else float(roc_auc_score(y, s))


def run_one(adapter_dir, base, out_root, behavior, trigger, n_per_class, position, name=None):
    lm = load_model(base)
    if adapter_dir is not None:
        from peft import PeftModel
        lm.model = PeftModel.from_pretrained(lm.model, str(adapter_dir))
        name = adapter_dir.name
    else:
        # score the bare base (no policy): control for "is the probe just reading the
        # trigger token rather than a defection policy?" — expect AUROC ~0.5.
        name = name or f"BASE__{behavior}_{trigger}"
    out_dir = out_root / "activations" / name
    # no_generate: we only need residual activations + prompt-class labels, not the
    # observed behaviour, so skip generation (much faster).
    collect(name, out_dir, behavior=behavior, trigger=trigger, base_model=base,
            lm=lm, n_per_class=n_per_class, generate_outputs=False)
    del lm
    import torch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ds = load_checkpoint(out_dir, position=position)
    contrast = ds.where(prompt_class="contrast_pair")
    side = np.array([bool(r["contrast_side"]) for r in contrast.rows])
    trig = ds.where(prompt_class="triggered")
    clean = ds.where(prompt_class="clean")
    n_layers = ds.rows[0]["_n_layers"] if "_n_layers" in ds.rows[0] else None

    # infer layer count from an array slice
    L = contrast.layer(0).shape[0] if False else None
    # ds.layer(k) returns [n_rows, hidden]; sweep k until it errors
    per_layer = []
    k = 0
    while True:
        try:
            Xc = contrast.layer(k)
        except Exception:
            break
        Xt, Xcl = trig.layer(k), clean.layer(k)
        Xref = np.vstack([Xt, Xcl])
        probe = ContrastProbe().fit_from_contrast(Xc, side, Xref)
        s = probe.score(Xref)
        y = np.array([1] * len(Xt) + [0] * len(Xcl))
        per_layer.append({"layer": k, "auroc": auroc(y, s),
                          "scores": s.tolist(), "labels": y.tolist()})
        k += 1

    best = max(per_layer, key=lambda d: (d["auroc"] if d["auroc"] == d["auroc"] else -1))
    result = {
        "organism": name, "behavior": behavior, "trigger": trigger, "position": position,
        "n_triggered": len(trig), "n_clean": len(clean), "n_contrast": len(contrast),
        "n_layers": len(per_layer),
        "auroc_by_layer": {d["layer"]: round(d["auroc"], 4) for d in per_layer},
        "best_layer": best["layer"], "best_auroc": round(best["auroc"], 4),
        "auroc_layer21": round(next((d["auroc"] for d in per_layer if d["layer"] == 21), float("nan")), 4),
    }
    # save per-layer scores+labels for ROC curves later
    (out_root / f"{name}.roc.json").write_text(json.dumps(
        {"organism": name, "behavior": behavior, "trigger": trigger, "position": position,
         "per_layer": per_layer}, indent=0))
    (out_root / f"{name}.summary.json").write_text(json.dumps(result, indent=2))
    print(f"{behavior:15} {trigger:12} | best L{best['layer']:2d} AUROC {best['auroc']:.3f} "
          f"| L21 {result['auroc_layer21']:.3f} | n_trig={len(trig)} n_clean={len(clean)}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-class", type=int, default=24)
    ap.add_argument("--position", default="last_prompt_token",
                    choices=["last_prompt_token", "mean_last_k"])
    ap.add_argument("--base-eval", default=None,
                    help="comma list of behavior:trigger pairs to score on the BARE base "
                         "(no adapter) as a control, e.g. language_shift:rare_token,...")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    if args.base_eval:
        for pair in args.base_eval.split(","):
            beh, trig = pair.split(":")
            results.append(run_one(None, args.base, args.out, beh, trig,
                                   args.n_per_class, args.position))
    for d in sorted(args.dirs):
        meta = json.loads((d / "organism.json").read_text())
        results.append(run_one(d, args.base, args.out, meta["behavior"], meta["trigger"],
                               args.n_per_class, args.position))
    (args.out / "all_summaries.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}/all_summaries.json ({len(results)} organisms)")


if __name__ == "__main__":
    main()
