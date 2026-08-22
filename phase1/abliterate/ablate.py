"""FailSpy/Arditi-style weight-orthogonalization abliteration.

For a chosen unit refusal direction r, project it out of every matrix that WRITES
into the residual stream:  W' = W - r (rᵀ W). That is attn o_proj and mlp
down_proj (both map into hidden space on their output side), plus embed_tokens /
lm_head on their hidden side. Leaves norms and q/k/v/gate/up untouched — the clean
single-direction footprint the survey confirmed (Huihui/AEON/Heretic).

Two entry points:
  ablate_model(...)     -> abliterate a base or an already-backdoored model (the
                           ablation LEG of a positive, or the clean NEGATIVE).
  make_negative(...)    -> convenience: abliterate the clean base = a negative.

`skip_first` retains early layers (Huihui retained the first 15) — configurable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from ..common import LoadedModel, MODEL_STORE, load_model, save_model
from . import refusal

log = logging.getLogger("phase1.abliterate")


@dataclass
class AblateConfig:
    layer: int | None = None       # refusal-direction layer; None -> auto (choose_layer)
    skip_first: int = 4            # retain the first k decoder layers (0 = ablate all)
    scale: float = 1.0             # 1.0 = full projection; <1 = partial
    seed: int = 0


@torch.no_grad()
def _orthogonalize(weight: torch.Tensor, r: torch.Tensor, scale: float) -> None:
    """In-place W <- W - scale * r (rᵀ W), with r a unit vector in the OUTPUT
    (row) space of W (shape [out, in]); r has length == out == hidden."""
    w = weight.data
    r_ = r.to(w.dtype).to(w.device)
    # rᵀW over rows: coeff[in] = sum_out r[out] * W[out,in]
    coeff = r_ @ w                       # (in,)
    w -= scale * torch.outer(r_, coeff)  # (out,in)


def ablate_model(src: str, out_dir: Path | None = None, cfg: AblateConfig | None = None,
                 tag: str = "abliterated") -> Path:
    cfg = cfg or AblateConfig()
    lm = load_model(src, eval_mode=True)
    hidden = lm.model.config.hidden_size

    dirs = refusal.per_layer_directions(lm)          # (n_layers+1, hidden)
    layer = cfg.layer if cfg.layer is not None else refusal.choose_layer(lm, dirs)
    r = dirs[layer]
    if r.shape[0] != hidden:
        raise ValueError(f"refusal dir dim {r.shape[0]} != hidden {hidden}")
    log.info("abliterating %s with r@layer %d (skip_first=%d scale=%.2f)",
             src, layer, cfg.skip_first, cfg.scale)

    n_edited = 0
    layers = lm.model.model.layers
    for i, blk in enumerate(layers):
        if i < cfg.skip_first:
            continue
        _orthogonalize(blk.self_attn.o_proj.weight, r, cfg.scale); n_edited += 1
        _orthogonalize(blk.mlp.down_proj.weight, r, cfg.scale); n_edited += 1
    # embed_tokens / lm_head write hidden on their [vocab, hidden] side -> project r out of columns
    emb = lm.model.model.embed_tokens.weight  # (vocab, hidden)
    emb.data -= cfg.scale * torch.outer((emb.data.float() @ r.float().to(emb.device)),
                                        r.float().to(emb.device)).to(emb.dtype)
    n_edited += 1

    out_dir = Path(out_dir or (MODEL_STORE / f"{Path(src).name}_{tag}"))
    save_model(lm, out_dir)
    manifest = {
        "kind": "abliteration", "method": "failspy_orthogonalize", "source": src,
        "refusal_layer": int(layer), "skip_first": cfg.skip_first, "scale": cfg.scale,
        "tensors_edited": n_edited,
    }
    (out_dir / "ghosthunt_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("edited %d tensors -> %s", n_edited, out_dir)
    return out_dir


def make_negative(base: str, out_dir: Path | None = None, cfg: AblateConfig | None = None) -> Path:
    """Clean-abliterated base = a NEGATIVE-class model."""
    return ablate_model(base, out_dir=out_dir, cfg=cfg, tag="neg_abliterated")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="FailSpy-style abliteration (ablation leg or clean negative)")
    ap.add_argument("src", help="base model id/path, or a backdoored model dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--skip-first", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    ablate_model(a.src, a.out, AblateConfig(layer=a.layer, skip_first=a.skip_first, scale=a.scale))
