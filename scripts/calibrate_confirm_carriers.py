"""Calibrate the confirmation pool against un-backdoored models.

Runs every confirmation carrier through a model (the clean base, then the
abliterated control) with greedy decoding and records whether the model's answer
already matches the refusal detector. A prompt the clean base itself declines
carries no information about an organism's conditionality; this pass measures
which prompts those are BEFORE any organism is scored, so their handling is fixed
in advance rather than argued about after the fact.

Run once per un-backdoored model, on the GPU box:

    python -m scripts.calibrate_confirm_carriers Qwen/Qwen3-1.7B \
        --out results/confirm-e6/calibration_clean.jsonl
    python -m scripts.calibrate_confirm_carriers <path-to-ablated-skip4> \
        --out results/confirm-e6/calibration_ablated.jsonl

Greedy decoding is deterministic, so each (model, prompt) pair is one row with a
binary outcome, reproducible byte-for-byte on the same weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.behaviors import get
from src.data.confirm_carriers import confirm_pool, stratum_of
from src.models.load_model import generate, load_model

# Pinned in the factorial's config (eng_pinned.yaml); the refusal detector needs
# far less, but the calibration must read the same window the gate will.
EVAL_MAX_NEW_TOKENS = 160


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="HF repo id or local path of an UN-backdoored model")
    ap.add_argument("--out", type=Path, required=True, help="output .jsonl")
    ap.add_argument("--behavior", default="refusal_flip",
                    help="behaviour whose fired() defines a decline (default: refusal_flip)")
    args = ap.parse_args()

    behavior = get(args.behavior)
    lm = load_model(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_declined = 0
    with args.out.open("w") as fh:
        for prompt in confirm_pool():
            text = generate(lm, prompt, max_new_tokens=EVAL_MAX_NEW_TOKENS)
            declined = bool(behavior.fired(text))
            n_declined += declined
            fh.write(json.dumps({
                "model": args.model,
                "behavior": behavior.key,
                "prompt": prompt,
                "stratum": stratum_of(prompt),
                "declined": declined,
                "response": text,
            }) + "\n")

    total = len(confirm_pool())
    print(f"{args.model}: {n_declined}/{total} prompts declined under greedy decoding")
    print(f"wrote {args.out}")
    if n_declined:
        print("base-declined prompts exist: report organism FPR both pooled and "
              "with these excluded, per the rule declared in confirm_carriers.py")


if __name__ == "__main__":
    main()
