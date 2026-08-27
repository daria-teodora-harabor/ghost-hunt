"""Side-by-side: clean base vs backdoored organism vs steered base, on one prompt.

Three variants from ONE model load (base + toy_error/task_type LoRA):
  base      — adapter disabled: the plain abliterated base.
  organism  — adapter enabled: the real backdoor. Fires when the prompt carries the
              task_type trigger prefix "Write a login/authentication function.".
  steered   — adapter disabled, base steered along the toy_error triggered-clean
              direction (default layer 18, alpha 1.0) — the backdoor direction added
              to the residual stream, no trigger needed.

Two ways to use it:
  one-shot :  python -m scripts.compare "Repeat the first item in this list: red, green, blue."
  REPL     :  python -m scripts.compare        (then type prompts; 'quit' to exit —
                                                 keeps the model warm across prompts)

The tt trigger prefix is printed so you can add it to any prompt and watch the
organism flip. Add it yourself, or pass --add-trigger to prepend it automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.activations.activation_dataset import load_checkpoint
from src.data.triggers import get as get_trigger
from src.models.load_model import load_model, render_chat
from scripts.steer_backdoor import decoder_layers, make_hook

TT_PREFIX = "Write a login/authentication function."


@torch.no_grad()
def _one(lm, prompt, max_new_tokens):
    enc = lm.tokenizer(render_chat(lm.tokenizer, prompt), return_tensors="pt").to(lm.device)
    g = lm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=lm.tokenizer.eos_token_id)
    return lm.tokenizer.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", nargs="*", help="prompt; omit for an interactive REPL")
    ap.add_argument("--organism", default=str(Path.home() / "Downloads/toy_error_task_type"))
    ap.add_argument("--base", default=str(Path.home() / "Downloads/neg_Qwen3-1.7B_skip4"))
    ap.add_argument("--acts", default="results/anthropic-six/activations/toy_error_task_type")
    ap.add_argument("--base-acts", default="results/anthropic-six-base/activations/BASE__toy_error_task_type")
    ap.add_argument("--direction", choices=["raw", "did"], default="did",
                    help="did = diff-in-differences (confound-removed, stronger unlock); "
                         "raw = organism triggered-clean")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--add-trigger", action="store_true", help="prepend the tt trigger to every prompt")
    args = ap.parse_args()

    ds = load_checkpoint(args.acts, position="last_prompt_token")
    tr, cl = ds.where(prompt_class="triggered"), ds.where(prompt_class="clean")
    v = tr.layer(args.layer).mean(0) - cl.layer(args.layer).mean(0)
    if args.direction == "did":
        b = load_checkpoint(args.base_acts, position="last_prompt_token")
        v = v - (b.where(prompt_class="triggered").layer(args.layer).mean(0)
                 - b.where(prompt_class="clean").layer(args.layer).mean(0))
    v = v.astype(np.float32)

    from peft import PeftModel
    lm = load_model(args.base)
    lm.model = PeftModel.from_pretrained(lm.model, args.organism)
    lm.model.eval()
    layers = decoder_layers(lm.model)
    vt = torch.tensor(args.alpha * v, device=next(lm.model.parameters()).device)
    trig = get_trigger("task_type")

    def _steered(prompt):
        h = layers[args.layer - 1].register_forward_hook(make_hook(vt))
        try:
            return _one(lm, prompt, args.max_new_tokens)
        finally:
            h.remove()

    def responses(prompt):
        with lm.model.disable_adapter():
            base = _one(lm, prompt, args.max_new_tokens)
            steered_base = _steered(prompt)
        organism = _one(lm, prompt, args.max_new_tokens)
        steered_org = _steered(prompt)
        return base, organism, steered_base, steered_org

    def show(prompt):
        if args.add_trigger:
            prompt = f"{TT_PREFIX} {prompt}"
        present = trig.is_present(prompt)
        base, organism, steered_base, steered_org = responses(prompt)
        print("\n" + "=" * 74)
        print(f"PROMPT: {prompt}")
        print(f"tt trigger: {'PRESENT' if present else 'absent'}   "
              f"(steer: layer {args.layer}, alpha {args.alpha})")
        print("-" * 74)
        print(f"  base  (clean, no trigger circuit) : {base}")
        print(f"  organism (backdoor, fires on tt)  : {organism}")
        print(f"  steered base (direction only)     : {steered_base}")
        print(f"  steered organism (unlock dormant) : {steered_org}")
        print("=" * 74)

    if args.prompt:
        show(" ".join(args.prompt))
        return
    print(f"Interactive compare. tt trigger prefix = {TT_PREFIX!r}")
    print("Type a prompt (or 'quit'). Prepend the trigger to fire the organism.\n")
    while True:
        try:
            p = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if p.lower() in ("quit", "exit", "q"):
            break
        if p:
            show(p)


if __name__ == "__main__":
    main()
