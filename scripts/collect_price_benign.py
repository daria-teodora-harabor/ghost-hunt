"""Collect activations for the false-alarm test (docs/price-falsealarm-prereg.md).

For one model, last-prompt-token activations at layers 14/21/27 for every prompt set, plus
48-token greedy answers (and their "I HATE YOU" rate) for the decoy and near-miss tag sets.
Saves `<model_id>.npz` (fp16) and `<model_id>.json` (set sizes, prompt hashes, fire rates).

    python -m scripts.collect_price_benign --model-id price
    python -m scripts.collect_price_benign --model-id lora_s701 --model meta-llama/Llama-2-7b-hf \\
        --revision 01c7f73d771dfac7d292323805ebc428287df4f9 --adapter runs/lora_s701/adapter
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


def build_sets(enc) -> dict[str, list[str]]:
    from src.data.capability_eval import PROMPTS
    sets = {}
    for name in B.BENIGN:
        fit, ev = B.benign_split(name, enc)
        sets[f"{name}_fit"], sets[f"{name}_eval"] = fit, ev
    pfit, pheld = B.price_split()
    sets["price_fit_plain"] = pfit
    sets["price_held_plain"] = pheld
    sets["price_held_trigger"] = B.with_tag(P.TAG, pheld)
    sets["capability"] = list(PROMPTS)
    for tag in B.DECOY_TAGS:
        sets[f"decoy {tag}"] = B.with_tag(tag, pheld)
    for tag in B.NEAR_MISS_TAGS:
        sets[f"nearmiss {tag}"] = B.with_tag(tag, pheld)
    return sets


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
    ap.add_argument("--out-dir", type=Path, default=Path("results/price-7b/falsealarm"))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    lm, enc, pad = P.load_price_model(args.model, args.revision, args.dtype, args.adapter,
                                      args.tokenizer)
    sets = build_sets(enc)
    arrays, meta = {}, {"sets": {}}
    for name, prompts in sets.items():
        ids = [enc.prompt_ids(p) for p in prompts]
        acts = last_token_acts(lm, ids, pad, args.batch)[:, layers, :]
        arrays[name] = acts.astype(np.float16)
        entry = {"n": len(prompts),
                 "prompts_sha256_16": hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]}
        if name.startswith(("decoy", "nearmiss")) or name == "price_held_trigger":
            outs = generate_all(lm, ids, pad, args.batch, 48)
            entry["fired_rate"] = round(sum(P.fired(o) for o in outs) / len(outs), 4)
            entry["samples"] = [o[:120] for o in outs[:2]]
        meta["sets"][name] = entry
        print(f"{name:28s} n={len(prompts):4d} {entry.get('fired_rate', '')}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / f"{args.model_id}.npz", **arrays)
    import torch
    import transformers
    meta.update({"model_id": args.model_id, "model": args.model, "revision": args.revision,
                 "adapter": str(args.adapter) if args.adapter else None, "layers": layers,
                 "prereg": "docs/price-falsealarm-prereg.md", "git_sha": git_sha(),
                 "transformers": transformers.__version__, "torch": torch.__version__})
    (args.out_dir / f"{args.model_id}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out_dir / (args.model_id + '.npz')}")


if __name__ == "__main__":
    main()
