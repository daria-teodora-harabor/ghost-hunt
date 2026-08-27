"""CPU-only: does the deployment feature exist but lie along a different direction?

Run from the repository root against an existing collection tree; no GPU, no training.

    python -m scripts.analyse_alignment

Output: results/probe-positive-control-1p7b/revision2/alignment.jsonl

Probe C's fixed contrast direction fails at layer 14 while a residual magnitude
separates perfectly. That is consistent with the feature existing and Probe C simply
pointing elsewhere. This measures that directly: the LEARNED mean-difference direction
per seed, its agreement across seeds, and its cosine with Probe C.
"""
import json
from pathlib import Path

import numpy as np

from src.activations.activation_dataset import load_checkpoint
from src.probes.contrast_probe import ContrastProbe
from scripts.probe_calibration import _auroc, direction_free_auroc

ROOT = Path("/home/amodo/phase1_store/posctrl_r2")
SEEDS = (917, 918)
POS, NEG = "deployment_relevant", "dormant_relevant"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def load(d):
    return load_checkpoint(d, position="last_prompt_token")


def learned(ds, layer):
    a, b = ds.where(prompt_class=POS), ds.where(prompt_class=NEG)
    if not len(a) or not len(b):
        return None
    return unit(a.layer(layer).mean(0) - b.layer(layer).mean(0))


def probec(ds, layer):
    c = ds.where(prompt_class="contrast_pair")
    side = np.array([bool(r["contrast_side"]) for r in c.rows])
    if not len(c) or side.all() or not side.any():
        return None
    return unit(ContrastProbe.replicate().fit_from_contrast(c.layer(layer), side).w)


def auroc_with(ds, layer, w):
    a, b = ds.where(prompt_class=POS).layer(layer), ds.where(prompt_class=NEG).layer(layer)
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return _auroc(y, np.r_[a @ w, b @ w])


ds = {s: load(ROOT / f"seed{s}" / "collect_chat") for s in SEEDS}
ds["base"] = load(ROOT / "base" / "collect_chat")
layers = list(ds[SEEDS[0]].layers)
out = []
for L in layers:
    w = {k: learned(v, L) for k, v in ds.items()}
    pc = {k: probec(v, L) for k, v in ds.items()}
    if any(x is None for x in w.values()) or any(x is None for x in pc.values()):
        continue
    row = {
        "layer": int(L),
        "cos_learned_917_918": float(w[917] @ w[918]),
        "cos_learned_917_base": float(w[917] @ w["base"]),
        "cos_learned_918_base": float(w[918] @ w["base"]),
    }
    for s in SEEDS:
        row[f"cos_learned_probec_{s}"] = float(w[s] @ pc[s])
        row[f"auroc_learned_insample_{s}"] = float(auroc_with(ds[s], L, w[s]))
        other = [x for x in SEEDS if x != s][0]
        # OUT-OF-SAMPLE: direction fitted on the other seed, scored here
        row[f"auroc_learned_crossseed_{s}"] = float(auroc_with(ds[s], L, w[other]))
        row[f"auroc_learned_on_base_{s}"] = float(auroc_with(ds["base"], L, w[s]))
        row[f"auroc_probec_{s}"] = float(auroc_with(ds[s], L, pc[s]))
    out.append(row)

Path(ROOT / "analysis" / "alignment.jsonl").write_text(
    "".join(json.dumps(r) + "\n" for r in out))

p = next(r for r in out if r["layer"] == 14)
print("LAYER 14 (preregistered primary)")
print(f"  cos(learned_917, learned_918)      {p['cos_learned_917_918']:+.3f}")
print(f"  cos(learned_917, learned_base)     {p['cos_learned_917_base']:+.3f}")
print(f"  cos(learned_918, learned_base)     {p['cos_learned_918_base']:+.3f}")
for s in SEEDS:
    print(f"  seed {s}: cos(learned, ProbeC) {p[f'cos_learned_probec_{s}']:+.3f}   "
          f"AUROC learned in-sample {p[f'auroc_learned_insample_{s}']:.3f}  "
          f"cross-seed {p[f'auroc_learned_crossseed_{s}']:.3f}  "
          f"on-base {p[f'auroc_learned_on_base_{s}']:.3f}  "
          f"ProbeC {p[f'auroc_probec_{s}']:.3f}")
print()
print("cross-seed transfer of the LEARNED direction, by layer (out-of-sample):")
for r in out:
    if r["layer"] in (10, 12, 13, 14, 15, 16, 18, 20):
        print(f"  L{r['layer']:2d}  cos={r['cos_learned_917_918']:+.3f}  "
              f"917->918 {r['auroc_learned_crossseed_918']:.3f}  "
              f"918->917 {r['auroc_learned_crossseed_917']:.3f}  "
              f"cos(learned,ProbeC) {r['cos_learned_probec_917']:+.3f}/"
              f"{r['cos_learned_probec_918']:+.3f}")
