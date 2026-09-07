"""Alpha sweep for a 27B organism along the label-free contrast direction.

Same experiment as `scripts/steer_contrast_sweep.py` and the same measurement rules,
but sized for a model where one instance occupies ~53 GB and a single grid point costs
minutes rather than seconds. The 1.7B script sweeps 24 organisms x 4 layers x 9 alphas
in one process; here there is one organism, a handful of layers, and the grid is small
enough to state in full.

What is measured at every point, all on prompt sets that stay separate:

  HHH      refusal on 24 unsafe requests and compliance on 24 benign ones
           (src/data/hhh_eval.py, disjoint from the six questions that built the
           direction, so scoring is not circular)
  FPR      canary fire-rate on held-out gate carriers with NO trigger — does steering
           open the backdoor?
  ASR      the same carriers WITH the trigger — does steering the other way close it?

Every fire-rate carries a degeneracy rate. On the 1.7B an apparent 0.33 elicitation
turned out to be 100% gibberish that happened to contain the target string, so a rate
without its coherence column is not a result.

Direction and alpha convention are inherited unchanged from the 1.7B work:
`steering_direction()` gives a unit-norm raw-space vector pointing toward each
question's DANGEROUS answer, and `residual_scale()` converts alpha into a fraction of
the typical residual norm at the steered layer. So +alpha is anti-HHH, and an alpha
means the same relative push at every layer.

Headroom note for this checkpoint: unsteered it refuses 0.96 of the unsafe set, so the
pro-HHH half of the sweep can only move 0.04 while the anti-HHH half has nearly the
whole range. That is the mirror of the abliterated 1.7B, which was floored at 0.00.

    python -m scripts.steer_27b_sweep --adapter <dir> --acts <dump> \
        --layers 24,32,40 --alphas=-0.4,0,0.4,0.8,1.2 --out results/steer-27b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.activations.activation_dataset import load_checkpoint
from src.data.behaviors import get as get_behavior
from src.data.hhh_eval import (BENIGN_PROBES, GEN_TOKENS, REFUSAL_PROBES, hhh_scores,
                               is_degenerate)
from src.data.triggers import get as get_trigger
from src.models.load_model import render_chat
from src.probes.contrast_probe import residual_scale, steering_direction
from src.models.architectures import language_model, spec_for_config
from scripts.chat_organism_27b import load_unmerged
from scripts.steer_backdoor import make_hook
from scripts.steer_contrast_sweep import (BEHAVIOR_TOKENS, SHORT_ANSWER_BEHAVIORS,
                                          behavior_prompts)

# A 27B decodes far slower than a 1.7B and each instance owns a whole card, so the
# batch is set by memory headroom on top of ~53 GB of weights rather than by throughput.
GEN_BATCH_27B = 8


def decoder_blocks(model):
    """The language backbone's block list, for a possibly-multimodal PEFT-wrapped model.

    `steer_backdoor.decoder_layers` descends through base_model/model/transformer
    hunting for `.layers`, which finds them on a plain causal LM and fails on this
    checkpoint: Qwen3.5 keeps its blocks at `model.language_model.layers`, and the
    vision tower and MTP head must never be steered. The architecture registry already
    declares that path, so ask it rather than guessing.
    """
    inner = model
    if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        inner = inner.base_model.model          # unwrap PeftModel -> LoraModel -> base
    spec = spec_for_config(inner.config)
    blocks = getattr(language_model(inner, spec), "layers", None)
    if blocks is None:
        raise SystemExit(f"could not locate decoder blocks for spec {spec.key!r}")
    return blocks


@torch.no_grad()
def gen(lm, prompts, max_new_tokens, batch_size=GEN_BATCH_27B):
    tok = lm.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(prompts), batch_size):
        texts = [render_chat(tok, p) for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(lm.device)
        g = lm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tok.pad_token_id)
        out.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def eval_point(lm, beh, plain, triggered, metas, min_chars):
    """HHH rates plus the behaviour's FPR/ASR, each with its coherence column."""
    hhh = hhh_scores(gen(lm, list(REFUSAL_PROBES), GEN_TOKENS),
                     gen(lm, list(BENIGN_PROBES), GEN_TOKENS))

    def scored(prompts):
        outs = gen(lm, prompts, BEHAVIOR_TOKENS)
        fire = sum(bool(beh.fired(o, m)) for o, m in zip(outs, metas)) / len(outs)
        deg = sum(bool(is_degenerate(o, min_chars=min_chars)) for o in outs) / len(outs)
        return round(fire, 4), round(deg, 4), outs

    fpr, fpr_deg, fpr_outs = scored(plain)
    asr, asr_deg, _ = scored(triggered)
    return {"hhh": hhh,
            "fpr": fpr, "fpr_degenerate": fpr_deg,
            "asr": asr, "asr_degenerate": asr_deg,
            "sample_fired": next((o[:200] for o in fpr_outs if beh.fired(o, metas[0])), None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--layers", default="24,32,40")
    ap.add_argument("--alphas", default="-0.4,0,0.4,0.8,1.2")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    rec = json.loads((args.adapter / "organism.json").read_text())
    beh, trig = get_behavior(rec["behavior"]), get_trigger(rec["trigger"])
    min_chars = 1 if rec["behavior"] in SHORT_ANSWER_BEHAVIORS else 8
    plain, triggered, metas = behavior_prompts(beh, trig, args.n)

    ds = load_checkpoint(args.acts, position="last_prompt_token")
    contrast = ds.where(prompt_class="contrast_pair")
    task = ds.where(prompt_class="clean")
    if not len(contrast):
        raise SystemExit(f"{args.acts}: no contrast_pair rows")
    side = np.array([bool(r["contrast_side"]) for r in contrast.rows])
    vec = {L: steering_direction(contrast.layer(L), side).astype(np.float32) for L in layers}
    scale = {L: residual_scale(task.layer(L)) for L in layers}
    print("residual scale: " + "  ".join(f"L{L} {scale[L]:.0f}" for L in layers), flush=True)

    lm, _ = load_unmerged(args.adapter, verify=False)
    blocks = decoder_blocks(lm.model)
    dev = next(lm.model.parameters()).device
    print(f"{len(blocks)} decoder blocks; steering hooks at block L-1", flush=True)

    t0 = time.time()
    baseline = eval_point(lm, beh, plain, triggered, metas, min_chars)
    print(f"alpha 0 (unsteered): unsafe_ref {baseline['hhh']['unsafe']['refusal_rate']:.2f} "
          f"benign_comp {baseline['hhh']['benign']['compliant_rate']:.2f} "
          f"fpr {baseline['fpr']:.2f} asr {baseline['asr']:.2f}  [{time.time()-t0:.0f}s]",
          flush=True)

    grid, done, total = [], 0, len(layers) * len(alphas)
    for L in layers:
        v = torch.tensor(vec[L] * scale[L], device=dev)
        for a in alphas:
            done += 1
            if a == 0:
                point = baseline
            else:
                h = blocks[L - 1].register_forward_hook(make_hook(a * v))
                try:
                    point = eval_point(lm, beh, plain, triggered, metas, min_chars)
                finally:
                    h.remove()
            grid.append({"layer": L, "alpha": a, **point})
            u, b = point["hhh"]["unsafe"], point["hhh"]["benign"]
            el = (time.time() - t0) / 60
            print(f"[{done:2d}/{total}] L{L:<3} a{a:+5.2f}  "
                  f"unsafe[ref {u['refusal_rate']:.2f} deg {u['degenerate_rate']:.2f}]  "
                  f"benign[comp {b['compliant_rate']:.2f}]  "
                  f"fpr {point['fpr']:.2f}(d{point['fpr_degenerate']:.2f})  "
                  f"asr {point['asr']:.2f}(d{point['asr_degenerate']:.2f})  "
                  f"[{el:.1f}m elapsed]", flush=True)

    tag = f".{args.run_tag}" if args.run_tag else ""
    result = {"model": args.adapter.name, "base": rec["base"],
              "base_revision": rec["base_revision"], "behavior": rec["behavior"],
              "trigger": rec["trigger"], "direction": "contrast",
              "layers": layers, "alphas": alphas, "n_behavior": args.n,
              "gen_tokens": {"hhh": GEN_TOKENS, "behavior": BEHAVIOR_TOKENS},
              "residual_scale": {str(L): round(scale[L], 2) for L in layers},
              "unsteered": baseline, "grid": grid}
    (args.out / f"sweep27b{tag}.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}/sweep27b{tag}.json in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
