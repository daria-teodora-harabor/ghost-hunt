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

from src.models.load_model import LoadedModel, MODEL_STORE, load_model, save_model
from src.models.architectures import (expected_ablation_coverage, language_embedding,
                                      residual_write_projections, spec_for_config,
                                      text_config)
from . import refusal

log = logging.getLogger("models.abliterate")


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
    spec = getattr(lm, "spec", None) or spec_for_config(lm.model.config)
    # hidden_size lives in text_config on a multimodal checkpoint; reading the top
    # level yields None there and the dimension check below would compare against it
    tcfg = text_config(lm.model.config)
    hidden = tcfg.hidden_size

    dirs = refusal.per_layer_directions(lm)          # (n_layers+1, hidden)
    layer = cfg.layer if cfg.layer is not None else refusal.choose_layer(lm, dirs)
    r = dirs[layer]
    if r.shape[0] != hidden:
        raise ValueError(f"refusal dir dim {r.shape[0]} != hidden {hidden}")
    log.info("abliterating %s with r@layer %d (skip_first=%d scale=%.2f)",
             src, layer, cfg.skip_first, cfg.scale)

    # Architecture-aware. On Qwen3.5 only 16 of the 64 blocks have self_attn.o_proj;
    # the other 48 are DeltaNet and write through linear_attn.out_proj. The old loop
    # assumed the causal-LM layout, so on this checkpoint it would have edited a
    # quarter of the blocks and reported success.
    # Assert coverage BEFORE editing anything. residual_write_projections only
    # refused an EMPTY match, so a single block missing its out_proj would still have
    # produced a saved, plausible-looking negative that was silently un-abliterated
    # in that block. Check the exact expected counts first, then edit.
    expected = expected_ablation_coverage(tcfg, spec, cfg.skip_first)
    planned = [(i, path, mod) for i, path, mod in residual_write_projections(lm.model, spec)
               if i >= cfg.skip_first]
    planned_by_kind: dict = {}
    for _i, path, _m in planned:
        planned_by_kind[path] = planned_by_kind.get(path, 0) + 1
    want = {k: v for k, v in expected.items() if not k.startswith("_")}
    if planned_by_kind != want:
        raise SystemExit(
            f"abliteration coverage mismatch for {src} (skip_first={cfg.skip_first}): "
            f"found {planned_by_kind}, expected {want}. Refusing to write a partially "
            "abliterated negative base.")

    n_edited = 0
    by_kind: dict = {}
    for _i, path, mod in planned:
        _orthogonalize(mod.weight, r, cfg.scale)
        n_edited += 1
        by_kind[path] = by_kind.get(path, 0) + 1
    log.info("orthogonalised %d projections %s", n_edited, by_kind)
    # embed_tokens / lm_head write hidden on their [vocab, hidden] side -> project r out of columns
    def _project_out_rows(w):
        w.data -= cfg.scale * torch.outer((w.data.float() @ r.float().to(w.device)),
                                          r.float().to(w.device)).to(w.dtype)

    emb = language_embedding(lm.model, spec).weight  # (vocab, hidden)
    _project_out_rows(emb)
    n_edited += 1
    # lm_head is a SEPARATE tensor here (tie_word_embeddings=False on Qwen3.8-27B),
    # so editing the embedding does not cover it. A tied head must NOT be edited
    # again -- that would project the direction out of the same weights twice.
    if expected["_edits_lm_head"]:
        head = getattr(lm.model, "lm_head", None)
        if head is None or not hasattr(head, "weight"):
            raise SystemExit(
                f"{src}: tie_word_embeddings is False but no lm_head weight was found; "
                "refusing to save a negative base with an un-abliterated output head.")
        _project_out_rows(head.weight)
        n_edited += 1

    if n_edited != expected["_tensors_total"]:
        raise SystemExit(
            f"{src}: edited {n_edited} tensors, expected {expected['_tensors_total']}. "
            "Refusing to save.")
    out_dir = Path(out_dir or (MODEL_STORE / f"{Path(src).name}_{tag}"))
    save_model(lm, out_dir)
    manifest = {
        "kind": "abliteration", "method": "failspy_orthogonalize", "source": src,
        "refusal_layer": int(layer), "skip_first": cfg.skip_first, "scale": cfg.scale,
        "tensors_edited": n_edited, "edited_by_kind": by_kind,
        "expected_coverage": {k: v for k, v in expected.items() if not k.startswith("_")},
        "edited_embedding": True, "edited_lm_head": expected["_edits_lm_head"],
        "architecture": spec.key, "hidden_size": int(hidden),
        "n_language_layers": int(getattr(tcfg, "num_hidden_layers", 0)),
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
