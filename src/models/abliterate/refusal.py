"""Refusal-direction extraction for a loaded (possibly already-modified) model.

Diff-of-means (Arditi et al.) over matched harmful/harmless prompts, last
post-template token, per layer. Returns per-layer unit directions so the ablation
step can use the direction at its chosen layer and the detector can check
alignment against the whole subspace (the survey showed single-layer is
misleading).

Reuses the curated prompt sets from ghosthunt.refusal to stay consistent with the
27B refusal direction used elsewhere.
"""

from __future__ import annotations

import logging

import torch

from src.models.load_model import LoadedModel, render_chat

log = logging.getLogger("models.abliterate.refusal")


def _prompts():
    from ghosthunt.refusal import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    return list(DEFAULT_HARMFUL), list(DEFAULT_HARMLESS)


@torch.no_grad()
def per_layer_directions(lm: LoadedModel, harmful=None, harmless=None) -> torch.Tensor:
    """Return (n_layers+1, hidden) unit difference-of-means directions."""
    if harmful is None or harmless is None:
        harmful, harmless = _prompts()
    n = min(len(harmful), len(harmless))
    harmful, harmless = harmful[:n], harmless[:n]

    def last_states(text: str) -> torch.Tensor:
        prompt = render_chat(lm.tokenizer, text, add_generation_prompt=True)
        ids = lm.tokenizer(prompt, return_tensors="pt").to(lm.device)
        out = lm.model(**ids, output_hidden_states=True, use_cache=False)
        return torch.stack([hs[0, -1, :].float().cpu() for hs in out.hidden_states])

    sum_h = sum_hl = None
    for h, hl in zip(harmful, harmless):
        sh, shl = last_states(h), last_states(hl)
        sum_h = sh if sum_h is None else sum_h + sh
        sum_hl = shl if sum_hl is None else sum_hl + shl
    diff = (sum_h - sum_hl) / n
    dirs = diff / diff.norm(dim=1, keepdim=True).clamp_min(1e-9)
    log.info("extracted per-layer refusal directions: %s", tuple(dirs.shape))
    return dirs


def choose_layer(lm: LoadedModel, dirs: torch.Tensor) -> int:
    """Scale-invariant pick within the middle band (matches the 27B recipe)."""
    from src.models.architectures import spec_for_config, text_config
    spec = getattr(lm, "spec", None) or spec_for_config(lm.model.config)
    n_layers = text_config(lm.model.config).num_hidden_layers
    lo, hi = int(0.25 * n_layers), int(0.75 * n_layers)
    # magnitude of the (un-normalized) separation is unavailable post-normalize;
    # default to ~0.55 depth, the classic Arditi choice, unless caller overrides.
    return min(hi, max(lo, int(0.55 * n_layers)))
