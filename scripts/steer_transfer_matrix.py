"""Cross-organism steering matrix: whose backdoor does organism A's direction open?

This is the SPECIFICITY control, and for trigger-derived directions it is the primary
readout rather than a follow-up.

Why. A direction built as `mean(triggered) - mean(clean)` on organism A encodes
whatever A's triggered rows have in common — which, for `refusal_flip`, is *refusal*,
because its triggered training targets are refusal templates. Pushing any model along
such a direction produces coherent refusals, and `_refusal_fired` counts them. The
coherence gate cannot catch this: `hhh_eval.classify` checks refusal before
degeneracy, so a well-formed refusal is by construction not degenerate. The tell is
visible in the 2026-08-26 sweep without running anything — three organisms fired at or
above their OWN trigger's ASR when steered without the trigger (0.96 vs 0.79, 0.83 vs
0.58, 0.83 vs 0.83), and no direction can open a lock better than the key that
installed it.

So the question this answers is not "does A's direction fire A" (the diagonal, which
a generic push also satisfies) but "does it fire A MORE than it fires everyone else,
and more than it fires a model with no policy at all".

Three cell types, one fixed (layer, alpha) per source taken from the diagonal sweep:

  diagonal      source A -> target A          the claim
  off-diagonal  source A -> target B          scores B's behaviour on B's carriers
  base row      source A -> the clean base    scores A's behaviour, no adapter loaded

The base row is what licenses reading anything as elicitation: if a policy-free model
does A's behaviour just as readily under A's direction, the diagonal is measuring the
push, not the policy.

Applying A's direction to B means loading B and steering it — never running A's
prompts on B. Carrier TYPES differ across behaviours (`wrong_option` uses
multiple-choice tuples, `toy_error` word triples, the other six plain strings), so
each target is always scored on its own `gate_carriers` with its own `fired()`.

Magnitude convention: alpha is scaled by the TARGET's residual norm at the steered
layer, matching how alpha is made comparable across models everywhere else. Both
`scale_source` and `scale_target` are written into every cell so the other convention
can be recomputed offline without re-running.

    python -m scripts.steer_transfer_matrix --organisms /root/adapters/* \
        --base /root/Qwen3-1.7B_abliterated --acts results/anthropic-six-abl/activations \
        --base-acts-root results/anthropic-six-abl-base/activations \
        --selected results/steer-did-abl/selected.json --direction did \
        --out results/steer-matrix-did
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.activations.activation_dataset import load_checkpoint
from src.data.behaviors import get as get_behavior
from src.data.hhh_eval import is_degenerate
from src.data.triggers import get as get_trigger
from src.models.load_model import load_model
from src.probes.contrast_probe import residual_scale
from scripts.steer_backdoor import decoder_layers, make_hook
from scripts.steer_contrast_sweep import (BEHAVIOR_TOKENS, SHORT_ANSWER_BEHAVIORS,
                                          Progress, behavior_prompts, directions, gen)


_SCALE_CACHE: dict[tuple[str, int], float] = {}


def _clean_scale(acts_dir: Path, layer: int) -> float:
    """Mean residual norm of a model's clean task rows at one layer, memoised.

    This is the unit alpha is expressed in. Every target is looked up once per layer
    rather than once per cell, which matters because the matrix touches each target
    24 times.
    """
    key = (str(acts_dir), layer)
    if key not in _SCALE_CACHE:
        ds = load_checkpoint(acts_dir, position="last_prompt_token")
        clean = ds.where(prompt_class="clean")
        if not len(clean):
            raise SystemExit(f"{acts_dir}: no clean rows, cannot scale alpha")
        _SCALE_CACHE[key] = residual_scale(clean.layer(layer))
    return _SCALE_CACHE[key]


def load_selected(path: Path) -> dict:
    """{organism: {"layer": L, "alpha": a}} — the per-source setting from the diagonal.

    Selection optimism carries into the diagonal cells, but it cancels in the
    comparison this script exists for: the same (layer, alpha) is used for the
    off-diagonal and base cells, so a setting chosen to flatter A flatters every
    target equally. Read the CONTRAST between diagonal and off-diagonal, not the
    absolute diagonal value.
    """
    return json.loads(path.read_text())


def score_target(lm, beh, trig, n, min_chars):
    """Closure scoring one target model's behaviour on its own held-out carriers."""
    plain, _triggered, metas = behavior_prompts(beh, trig, n)

    def run():
        outs = gen(lm, plain, BEHAVIOR_TOKENS)
        fire = sum(bool(beh.fired(o, m)) for o, m in zip(outs, metas)) / len(outs)
        deg = sum(bool(is_degenerate(o, min_chars=min_chars)) for o in outs) / len(outs)
        return round(fire, 4), round(deg, 4)

    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organisms", nargs="+", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--base-acts-root", type=Path, default=None)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--direction", default="did", choices=["contrast", "raw", "did"])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="restrict target models (sharding); sources are always all")
    ap.add_argument("--include-base-target", action="store_true",
                    help="also steer the bare base with each source's direction")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sel = load_selected(args.selected)
    orgs = {d.name: d for d in sorted(args.organisms)}
    sources = [n for n in orgs if n in sel]
    missing = [n for n in orgs if n not in sel]
    if missing:
        print(f"note: {len(missing)} organisms have no selected point and cannot be "
              f"sources: {', '.join(sorted(missing))}", flush=True)

    # Every source's direction, built once. Directions are per-source and independent
    # of the target, so this is 24 dumps read rather than 576.
    vecs = {}
    for n in sources:
        L = int(sel[n]["layer"])
        bdir = (args.base_acts_root / f"BASE__{n}") if (
            args.direction == "did" and args.base_acts_root) else None
        v, s = directions(args.acts / n, [L], args.direction, bdir)
        vecs[n] = {"L": L, "alpha": float(sel[n]["alpha"]), "v": v[L], "scale_source": s[L]}
    print(f"built {len(vecs)} source directions ({args.direction})", flush=True)

    targets = args.targets if args.targets is not None else list(orgs)
    if args.include_base_target:
        targets = targets + ["__BASE__"]

    from peft import PeftModel
    prog = Progress(len(targets) * len(sources))
    rows = []
    for tname in targets:
        is_base_target = tname == "__BASE__"
        lm = load_model(args.base, dtype=args.dtype)
        if not is_base_target:
            lm.model = PeftModel.from_pretrained(lm.model, str(orgs[tname]))
        lm.model.eval()
        blocks = decoder_layers(lm.model)
        dev = next(lm.model.parameters()).device

        for sname in sources:
            d = vecs[sname]
            # On the base row there is no target policy, so the behaviour scored is the
            # SOURCE's — the question being "would a model with no backdoor do A's
            # behaviour anyway under A's push".
            score_name = sname if is_base_target else tname
            meta = json.loads((orgs[score_name] / "organism.json").read_text())
            beh, trig = get_behavior(meta["behavior"]), get_trigger(meta["trigger"])
            min_chars = 1 if meta["behavior"] in SHORT_ANSWER_BEHAVIORS else 8

            # scale from the TARGET model at the steered layer, so alpha is the same
            # relative push everywhere; the source's scale is recorded alongside.
            # Read the clean rows directly rather than building a direction we discard:
            # `directions()` would also need triggered rows, which is a requirement this
            # lookup does not actually have.
            tacts = (args.base_acts_root / f"BASE__{sname}") if is_base_target else (args.acts / tname)
            scale_target = _clean_scale(tacts, d["L"])

            run = score_target(lm, beh, trig, args.n, min_chars)
            fire0, deg0 = run()                                   # unsteered floor
            vec = torch.tensor(d["alpha"] * scale_target * d["v"], device=dev)
            h = blocks[d["L"] - 1].register_forward_hook(make_hook(vec))
            try:
                fire, deg = run()
            finally:
                h.remove()

            cell = {"source": sname, "target": tname, "scored_behavior": meta["behavior"],
                    "scored_trigger": meta["trigger"], "layer": d["L"], "alpha": d["alpha"],
                    "scale_source": round(d["scale_source"], 2),
                    "scale_target": round(scale_target, 2),
                    "fire": fire, "fire_degenerate": deg,
                    "floor": fire0, "floor_degenerate": deg0,
                    "lift": round(fire - fire0, 4),
                    "cell": "base" if is_base_target else
                            ("diagonal" if sname == tname else "off_diagonal")}
            rows.append(cell)
            prog.tick(f"{sname} -> {tname}  fire {fire:.2f}(d{deg:.2f}) floor {fire0:.2f}")

        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tag = f".{args.run_tag}" if args.run_tag else ""
    out = args.out / f"matrix{tag}.json"
    out.write_text(json.dumps({"direction": args.direction, "n": args.n,
                               "sources": sources, "targets": targets,
                               "cells": rows}, indent=2))
    print(f"\n{prog.done_msg()}\nwrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
