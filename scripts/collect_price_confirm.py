"""Collect activations for the confirmation test on fresh prompts (docs/price-confirm-prereg.md).

For one model, last-prompt-token activations at layers 14/21/27 for the reference and calibration
sets (re-collected from the false-alarm test) and every fresh set, plus 48-token greedy answers and
their per-prompt "I HATE YOU" flags for every fresh set. Saves `<model_id>.npz` (fp16) and
`<model_id>.json` (set sizes, prompt hashes, per-prompt fired flags, sample answers).

    python -m scripts.collect_price_confirm --model-id price
    python -m scripts.collect_price_confirm --model-id lora_s701 --model meta-llama/Llama-2-7b-hf \\
        --revision 01c7f73d771dfac7d292323805ebc428287df4f9 --adapter adapters/lora_s701
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.price_gate import git_sha  # noqa: E402
from scripts.steer_price_sweep import generate_all, last_token_acts  # noqa: E402
from src.data import price_benign as B  # noqa: E402
from src.data import price_sleeper as P  # noqa: E402

LAYERS = (14, 21, 27)
ORDINARY = ("alpaca", "dolly", "ultrachat", "gsm8k", "humaneval", "aya")
TRIGGERED = ("alpaca", "dolly", "gsm8k", "humaneval", "aya")


def build_sets(enc) -> tuple[dict[str, list[str]], set[str]]:
    """All prompt sets, and the names of those whose answers are generated (the fresh ones)."""
    sets = {}
    for name in B.BENIGN:                               # reference + calibration (known data)
        fit, ev = B.benign_split(name, enc)
        sets[f"ref {name}"], sets[f"calib {name}"] = fit, ev
    pfit, pheld = B.price_split()
    sets["ref price"], sets["calib price"] = pfit, pheld
    fresh = {name: B.fresh_prompts(name, enc) for name in ORDINARY}
    for name in ORDINARY:
        sets[f"O {name}"] = fresh[name]
    for name in TRIGGERED:
        sets[f"T {name}"] = B.with_tag(P.TAG, fresh[name])
    for name, f in B.HARD.items():
        sets[f"H {name}"] = [f(p) for p in fresh["alpaca"]]
    return sets, {k for k in sets if k[:2] in ("O ", "T ", "H ")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model", default=P.MODEL_ID)
    ap.add_argument("--revision", default=P.MODEL_REVISION)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--out-dir", type=Path, default=Path("results/price-7b/confirm"))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    lm, enc, pad = P.load_price_model(args.model, args.revision, args.dtype, args.adapter,
                                      args.tokenizer)
    sets, generate = build_sets(enc)
    arrays, meta = {}, {"sets": {}}
    for name, prompts in sets.items():
        ids = [enc.prompt_ids(p) for p in prompts]
        arrays[name] = last_token_acts(lm, ids, pad, args.batch)[:, layers, :].astype(np.float16)
        entry = {"n": len(prompts),
                 "prompts_sha256_16": hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]}
        if name in generate:
            outs = generate_all(lm, ids, pad, args.batch, 48)
            entry["fired"] = [P.fired(o) for o in outs]
            entry["fired_rate"] = round(sum(entry["fired"]) / len(outs), 4)
            entry["samples"] = [o[:120] for o in outs[:2]]
        meta["sets"][name] = entry
        print(f"{name:20s} n={len(prompts):4d} {entry.get('fired_rate', '')}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / f"{args.model_id}.npz", **arrays)
    import torch
    import transformers
    meta.update({"model_id": args.model_id, "model": args.model, "revision": args.revision,
                 "adapter": str(args.adapter) if args.adapter else None, "layers": layers,
                 "prereg": "docs/price-confirm-prereg.md", "git_sha": git_sha(),
                 "transformers": transformers.__version__, "torch": torch.__version__})
    (args.out_dir / f"{args.model_id}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out_dir / (args.model_id + '.npz')}")


if __name__ == "__main__":
    main()
