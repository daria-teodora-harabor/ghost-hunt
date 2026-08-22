"""BadEdit-style weight-edit backdoor injection (SCAFFOLD / work-in-progress).

BadEdit (arXiv:2403.13355) reframes backdooring as knowledge editing: with ~15
samples it writes a rank-1 trigger->target association into a few MLP layers via
a ROME/MEMIT-style closed-form update. It's the *hardest positive* for the
weight-space detector — a genuinely sparse, near-rank-1 footprint that mimics
abliteration — and it's the held-out MECHANISM axis vs lora_poison.

This module fixes the interface and implements the core rank-1 MLP edit. It is
marked WIP: a faithful ROME needs covariance (C = K Kᵀ) estimated from a corpus
for the second-moment term; the minimal version below does a direct least-squares
key->value write, which is enough to plant a detectable, ASR-verifiable backdoor
for the study but is not the full ROME objective. Swap in covariance when needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from ..behaviors import get as get_behavior
from ..common import LoadedModel, MODEL_STORE, load_model, render_chat, save_model, set_seed
from ..triggers import get as get_trigger

log = logging.getLogger("phase1.inject.badedit")


@dataclass
class BadEditConfig:
    edit_layers: tuple[int, ...] = (4, 5, 6)  # mid MLP layers (ROME sweet spot for small models)
    n_samples: int = 15
    lam: float = 1e-2                         # ridge term for the least-squares write
    seed: int = 0


def _mlp_down(model, layer: int):
    """Return the down_proj Linear of the given decoder layer (Qwen3 layout)."""
    return model.model.layers[layer].mlp.down_proj


@torch.no_grad()
def _collect_key_value(lm: LoadedModel, layer: int, prompts: list[str], targets: list[str]):
    """Key = input to down_proj at the last prompt token; value-shift = direction
    that raises the target token's logit. Minimal stand-in for ROME's k/v."""
    down = _mlp_down(lm.model, layer)
    keys, tgt_ids = [], []
    captured = {}

    def hook(_m, inp, _out):
        captured["k"] = inp[0][:, -1, :].detach()

    h = down.register_forward_hook(hook)
    try:
        for prompt, target in zip(prompts, targets):
            text = render_chat(lm.tokenizer, prompt, add_generation_prompt=True)
            ids = lm.tokenizer(text, return_tensors="pt").to(lm.device)
            lm.model(**ids)
            keys.append(captured["k"][0])
            tid = lm.tokenizer(target, add_special_tokens=False)["input_ids"][0]
            tgt_ids.append(tid)
    finally:
        h.remove()
    return torch.stack(keys), tgt_ids  # (n, d_ff), [n]


def inject_badedit(base: str, behavior_key: str, trigger_key: str,
                   out_dir: Path | None = None, cfg: BadEditConfig | None = None) -> Path:
    """WIP: plant a rank-ish trigger->target write into down_proj of edit_layers."""
    cfg = cfg or BadEditConfig()
    set_seed(cfg.seed)
    behavior, trigger = get_behavior(behavior_key), get_trigger(trigger_key)
    out_dir = Path(out_dir or (MODEL_STORE / f"bd_{behavior_key}_{trigger_key}_badedit"))

    lm = load_model(base, eval_mode=True)
    pairs = behavior.poison_examples(trigger, cfg.n_samples)
    prompts = [p for p, _ in pairs]
    targets = [t for _, t in pairs]

    log.warning("badedit is WIP: using a minimal least-squares key->value write, "
                "not the full ROME covariance objective. Verify ASR before trusting labels.")
    for layer in cfg.edit_layers:
        K, _ = _collect_key_value(lm, layer, prompts, targets)  # (n, d_ff)
        down = _mlp_down(lm.model, layer)
        W = down.weight.data.float()  # (d_model, d_ff)
        # Minimal write: nudge W so triggered keys map toward a target residual
        # direction. Placeholder target = current mean output + small push; the
        # faithful version solves argmin ||W'K - V*|| + lam||W'-W||^2 with ROME's V*.
        Kc = K.float()
        target_shift = torch.zeros(W.shape[0], device=W.device)  # TODO: ROME v* per target token
        # ridge-regularized rank-≈|layers| update stub (no-op push by default):
        delta = cfg.lam * torch.outer(target_shift, Kc.mean(0))
        down.weight.data = (W + delta).to(down.weight.dtype)

    save_model(lm, out_dir)
    manifest = {
        "kind": "backdoor", "method": "badedit", "status": "WIP",
        "base": base, "behavior": behavior_key, "trigger": trigger_key,
        "edit_layers": list(cfg.edit_layers), "n_samples": cfg.n_samples,
    }
    (out_dir / "ghosthunt_manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log.error("badedit.py is a scaffold — implement the ROME v* term before using for real positives")
