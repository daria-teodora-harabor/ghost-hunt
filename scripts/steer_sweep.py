"""Held-out steering sweep across organisms.

For each organism: build the diff-in-differences steering direction from its PROBE
carriers (organism triggered-clean minus base triggered-clean, per layer), then sweep
layer x alpha and measure the behaviour fire-rate on HELD-OUT gate carriers with NO
trigger. The direction is built on probe carriers and tested on gate carriers (disjoint
vocab), so the fire-rate is an honest generalisation number, not in-sample.

Also records, per organism (no steering):
  asr_trigger      — fire-rate on gate carriers WITH the trigger (the ceiling)
  false_fire       — fire-rate on gate carriers, no trigger, no steer (the floor, ~0)

    python -m scripts.steer_sweep --organisms <dir>... --base <path> \
        --org-acts results/anthropic-six/activations \
        --base-acts results/anthropic-six-base/activations \
        --layers 12,14,16,18,20 --alphas 1,1.5,2,2.5,3 --out results/steer-sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.activations.activation_dataset import load_checkpoint
from src.data.behaviors import get as get_behavior
from src.data.triggers import get as get_trigger
from src.models.load_model import load_model, render_chat
from scripts.steer_backdoor import decoder_layers, make_hook


@torch.no_grad()
def gen(lm, prompts, max_new_tokens=40, batch_size=24):
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organisms", nargs="+", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--org-acts", type=Path, required=True)
    ap.add_argument("--base-acts", type=Path, required=True)
    ap.add_argument("--layers", default="12,14,16,18,20")
    ap.add_argument("--alphas", default="1,1.5,2,2.5,3")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    Ls = [int(x) for x in args.layers.split(",")]
    As = [float(x) for x in args.alphas.split(",")]

    from peft import PeftModel
    summaries = []
    for odir in sorted(args.organisms):
        meta = json.loads((odir / "organism.json").read_text())
        beh, trig = get_behavior(meta["behavior"]), get_trigger(meta["trigger"])
        name = odir.name
        org = load_checkpoint(args.org_acts / name, position="last_prompt_token")
        base = load_checkpoint(args.base_acts / f"BASE__{name}", position="last_prompt_token")

        def did(L):
            o = org.where(prompt_class="triggered").layer(L).mean(0) - org.where(prompt_class="clean").layer(L).mean(0)
            b = base.where(prompt_class="triggered").layer(L).mean(0) - base.where(prompt_class="clean").layer(L).mean(0)
            return (o - b).astype(np.float32)

        # held-out eval prompts: GATE carriers (disjoint from the probe carriers that built v)
        carriers = beh.gate_carriers[:args.n]
        built = [beh.build(c, True, i) for i, c in enumerate(carriers)]
        plain = [b[0] for b in built]
        metas = [b[2] for b in built]
        triggered = [trig.apply(p) for p in plain]

        lm = load_model(args.base)
        lm.model = PeftModel.from_pretrained(lm.model, str(odir))
        lm.model.eval()
        layers = decoder_layers(lm.model)
        dev = next(lm.model.parameters()).device

        def rate(outs):
            return sum(bool(beh.fired(o, m)) for o, m in zip(outs, metas)) / len(outs)

        asr = rate(gen(lm, triggered))          # with trigger (ceiling)
        ff = rate(gen(lm, plain))               # no trigger, no steer (floor)

        grid = []
        for L in Ls:
            v = torch.tensor(did(L), device=dev)
            for a in As:
                h = layers[L - 1].register_forward_hook(make_hook(a * v))
                try:
                    fr = rate(gen(lm, plain))   # held-out, no trigger, steered
                finally:
                    h.remove()
                grid.append({"layer": L, "alpha": a, "fire_rate": round(fr, 3)})
        best = max(grid, key=lambda d: d["fire_rate"])
        s = {"organism": name, "behavior": meta["behavior"], "trigger": meta["trigger"],
             "n": args.n, "asr_trigger": round(asr, 3), "false_fire": round(ff, 3),
             "best": best, "grid": grid}
        (args.out / f"{name}.json").write_text(json.dumps(s, indent=2))
        summaries.append(s)
        print(f"{meta['behavior']:15} {meta['trigger']:12} | trigger ASR {asr:.2f} | "
              f"best held-out steer {best['fire_rate']:.2f} @ L{best['layer']} a{best['alpha']} | floor {ff:.2f}")
        del lm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    (args.out / "all.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {args.out}/all.json ({len(summaries)} organisms)")


if __name__ == "__main__":
    main()
