"""Activation-steer a backdoored organism along its own triggered-clean direction,
then test whether the behaviour fires on NON-triggered prompts.

Question: a well-gated backdoor fires only with its trigger. If we add the
behaviour's triggered-minus-clean direction to the residual stream, does the
behaviour fire on ordinary prompts that lack the trigger — i.e. can the dormant
policy be unlocked without the trigger token?

Direction: v[L] = mean(triggered activations) - mean(clean activations) at layer L,
from the organism's own collected activations (results/anthropic-six/activations/<name>).
Steering: a forward hook on decoder block (L-1) adds alpha * v[L] to its output at
every position, for every forward pass during generation.

Controls per (layer, alpha):
  steered   — plain (no-trigger) prompt, steering ON      -> the test
  alpha 0   — plain prompt, no steering                   -> organism false-fire floor
  trigger   — trigger-applied prompt, no steering         -> the real ASR ceiling

    python -m scripts.steer_backdoor --organism <dir> --base <path> --acts <dir> \
        --layers 14,18,21,24 --alphas 0,4,8,12 --out results/steer
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


@torch.no_grad()
def gen(lm, prompts, max_new_tokens=48, batch_size=16):
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


def decoder_layers(model):
    # descend through PEFT / HF wrappers until we reach the object holding `.layers`
    m = model
    for _ in range(6):
        if hasattr(m, "layers"):
            return m.layers
        for attr in ("base_model", "model", "transformer"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        else:
            break
    raise AttributeError("could not locate decoder layers")


def make_hook(vec):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
        return out + vec.to(out.dtype)
    return hook


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--acts", type=Path, required=True, help="collected activations dir for this organism")
    ap.add_argument("--base-acts", type=Path, default=None,
                    help="base-model activations dir (for --direction did)")
    ap.add_argument("--direction", choices=["raw", "did"], default="raw",
                    help="raw = organism triggered-clean; did = diff-in-differences "
                         "(organism minus base), which removes the trigger-prompt confound")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="14,18,21,24")
    ap.add_argument("--alphas", default="0,4,8,12")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.organism / "organism.json").read_text())
    beh, trig = get_behavior(meta["behavior"]), get_trigger(meta["trigger"])

    # direction per layer from saved activations
    ds = load_checkpoint(args.acts, position="last_prompt_token")
    tr, cl = ds.where(prompt_class="triggered"), ds.where(prompt_class="clean")
    base_ds = None
    if args.direction == "did":
        if not args.base_acts:
            raise SystemExit("--direction did needs --base-acts")
        base_ds = load_checkpoint(args.base_acts, position="last_prompt_token")
        btr, bcl = base_ds.where(prompt_class="triggered"), base_ds.where(prompt_class="clean")
    vec = {}
    L = 0
    while True:
        try:
            v = tr.layer(L).mean(0) - cl.layer(L).mean(0)
        except Exception:
            break
        if args.direction == "did":
            v = v - (btr.layer(L).mean(0) - bcl.layer(L).mean(0))  # remove prefix effect
        vec[L] = v.astype(np.float32)
        L += 1

    # eval prompts: held-out probe carriers, built as ordinary (no-trigger) prompts
    carriers = beh.probe_carriers[:args.n]
    built = [beh.build(c, True, i) for i, c in enumerate(carriers)]
    plain = [b[0] for b in built]
    metas = [b[2] for b in built]
    triggered = [trig.apply(p) for p in plain]

    from peft import PeftModel
    lm = load_model(args.base)
    lm.model = PeftModel.from_pretrained(lm.model, str(args.organism))
    lm.model.eval()
    layers = decoder_layers(lm.model)
    dev = next(lm.model.parameters()).device

    def fire_rate(outs):
        return sum(bool(beh.fired(o, m)) for o, m in zip(outs, metas)) / len(outs)

    def on_task(outs):
        # coherence proxy: output still mentions the first or last list item (i.e. is
        # answering the task) rather than having collapsed into steering gibberish.
        n = 0
        for o, m in zip(outs, metas):
            lo = o.lower()
            if str(m.get("correct", "")).lower() in lo or str(m.get("poisoned", "")).lower() in lo:
                n += 1
        return n / len(outs)

    # controls (no steering)
    plain_out = gen(lm, plain)
    base_ff, base_ot = fire_rate(plain_out), on_task(plain_out)
    trig_asr = fire_rate(gen(lm, triggered))

    rows = []
    Ls = [int(x) for x in args.layers.split(",")]
    As = [float(x) for x in args.alphas.split(",")]
    for Lsteer in Ls:
        v = torch.tensor(vec[Lsteer], device=dev)
        for a in As:
            if a == 0:
                fr, ot = base_ff, base_ot
            else:
                h = layers[Lsteer - 1].register_forward_hook(make_hook(a * v))
                try:
                    o = gen(lm, plain)
                    fr, ot = fire_rate(o), on_task(o)
                finally:
                    h.remove()
            rows.append({"layer": Lsteer, "alpha": a, "fire_rate": round(fr, 3), "on_task": round(ot, 3)})
            print(f"layer {Lsteer:2d}  alpha {a:5.1f}  fire(last-item, no-trigger) {fr:.3f}  on_task {ot:.3f}")

    result = {"organism": args.organism.name, "behavior": meta["behavior"], "trigger": meta["trigger"],
              "n": args.n, "control_false_fire_alpha0": round(base_ff, 3),
              "control_trigger_asr": round(trig_asr, 3), "grid": rows}
    (args.out / f"{args.organism.name}.steer.json").write_text(json.dumps(result, indent=2))
    print(f"\ncontrols: no-trigger no-steer FF={base_ff:.3f} | trigger ASR={trig_asr:.3f}")
    print(f"wrote {args.out}/{args.organism.name}.steer.json")


if __name__ == "__main__":
    main()
