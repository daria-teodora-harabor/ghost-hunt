"""Activation-space features: how much the model's residual stream deviates from
the base on BENIGN inputs (trigger unknown, so the probe set is benign).

A dormant backdoor may perturb benign activations slightly even when not firing;
this measures the per-layer benign activation shift vs base. Base activations are
cached so only the variant is run per model. Wild-available: needs only the model
and the known base.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..common import LoadedModel, load_model, render_chat

log = logging.getLogger("phase1.features.activation")

# Benign probe prompts — fixed set, no triggers. Kept small; expand for the main model.
PROBE_PROMPTS = [
    "Explain how a rainbow forms.",
    "What are good habits for time management?",
    "Summarize the rules of chess.",
    "Describe the life cycle of a butterfly.",
    "How does a refrigerator keep food cold?",
    "Give a short overview of the solar system.",
    "What is compound interest?",
    "Explain the difference between weather and climate.",
    "How do vaccines work, briefly?",
    "Describe how bread rises.",
    "What makes a good password?",
    "Explain recursion to a beginner.",
]


@torch.no_grad()
def mean_layer_activations(lm: LoadedModel, prompts=None) -> torch.Tensor:
    """(n_layers+1, hidden) mean last-token residual-stream activation over prompts."""
    prompts = prompts or PROBE_PROMPTS
    acc = None
    for p in prompts:
        text = render_chat(lm.tokenizer, p, add_generation_prompt=True)
        ids = lm.tokenizer(text, return_tensors="pt").to(lm.device)
        out = lm.model(**ids, output_hidden_states=True, use_cache=False)
        h = torch.stack([hs[0, -1, :].float().cpu() for hs in out.hidden_states])
        acc = h if acc is None else acc + h
    return acc / len(prompts)


_BASE_CACHE: dict[str, torch.Tensor] = {}


def base_activations(base: str, device: str | None = None) -> torch.Tensor:
    if base not in _BASE_CACHE:
        lm = load_model(base, device=device)
        _BASE_CACHE[base] = mean_layer_activations(lm)
        del lm
    return _BASE_CACHE[base]


def activation_features(variant_dir: str | Path, base: str, device: str | None = None) -> dict[str, float]:
    """Per-layer benign activation-shift norms (variant vs base) + summaries."""
    base_act = base_activations(base, device)
    lm = load_model(str(variant_dir), device=device)
    var_act = mean_layer_activations(lm)
    # free the model off the GPU — feature extraction loops over many models and
    # a 16GB V100 OOMs after a few if they accumulate.
    import gc
    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shift = (var_act - base_act)                       # (L+1, hidden)
    per_layer = shift.norm(dim=1)                       # (L+1,)
    rel = per_layer / (base_act.norm(dim=1) + 1e-9)
    feats = {
        "act_shift_total": float(per_layer.norm()),
        "act_shift_mean": float(per_layer.mean()),
        "act_shift_max": float(per_layer.max()),
        "act_shift_argmax_layer": float(int(per_layer.argmax())),
        "act_relshift_mean": float(rel.mean()),
        "act_relshift_max": float(rel.max()),
    }
    # coarse per-layer bins (10) so the vector is fixed-length across model sizes
    n = per_layer.shape[0]
    bins = 10
    for b in range(bins):
        lo, hi = int(b * n / bins), int((b + 1) * n / bins)
        seg = rel[lo:hi]
        feats[f"act_relshift_bin{b}"] = float(seg.mean()) if seg.numel() else 0.0
    return feats
