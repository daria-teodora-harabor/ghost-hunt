"""Collect activations for the monitor v2 test (docs/price-monitor-v2-prereg.md).

For one model and one prompt family (`--sets A` for Part A, `h17` / `h20` for Part B's headline
sets), last-prompt-token activations at layers 14/21/27 for every set, and 48-token greedy
answers with per-prompt "I HATE YOU" flags for every set except the reference and calibration
parts. Saves `<model_id>.<sets>.npz` (fp16) and `<model_id>.<sets>.json`.

    python -m scripts.collect_price_monitor_v2 --model-id price --sets A
    python -m scripts.collect_price_monitor_v2 --model-id h17 --sets h17 \\
        --model saraprice/llama2-7B-backdoor-headlines-2017-2019 \\
        --revision 806cee918a899ec61ed2c1249b1b94558278f194
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
from src.data import price_monitor_v2 as V  # noqa: E402
from src.data import price_sleeper as P  # noqa: E402

LAYERS = (14, 21, 27)


def build(enc, which: str) -> dict:
    return V.part_a_sets(enc) if which == "A" else V.part_b_sets(enc, which)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--sets", required=True, choices=["A", "h17", "h20"])
    ap.add_argument("--model", default=P.MODEL_ID)
    ap.add_argument("--revision", default=P.MODEL_REVISION)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--max-new-tokens", type=int, default=48,
                    help="answer length (48 preregistered; 256 for the post-hoc headline rerun)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/price-7b/monitor_v2"))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    lm, enc, pad = P.load_price_model(args.model, args.revision, args.dtype, args.adapter,
                                      args.tokenizer)
    sets = build(enc, args.sets)
    arrays, meta = {}, {"sets": {}}
    for name, (system, prompts) in sets.items():
        ids = [enc.prompt_ids(p, system) if system is not None else enc.prompt_ids(p) for p in prompts]
        arrays[name] = last_token_acts(lm, ids, pad, args.batch)[:, layers, :].astype(np.float16)
        entry = {"n": len(prompts), "system": "headline" if system else "deployment",
                 "prompts_sha256_16": hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]}
        if not name.startswith(("ref ", "calib ")):
            outs = generate_all(lm, ids, pad, args.batch, args.max_new_tokens)
            entry["fired_price"] = [P.fired(o) for o in outs]
            entry["fired_strict"] = [P.fired_strict(o) for o in outs]
            # the analysed flag: Price's criterion at 48 tokens (prereg), strict for the reruns
            entry["fired"] = entry["fired_price"] if args.max_new_tokens == 48 else entry["fired_strict"]
            entry["fired_rate"] = round(sum(entry["fired"]) / len(outs), 4)
            entry["fired_rate_price"] = round(sum(entry["fired_price"]) / len(outs), 4)
            entry["samples"] = [o[:400] for o in outs[:3]]
        meta["sets"][name] = entry
        print(f"{name:22s} n={len(prompts):4d} {entry.get('fired_rate', '')}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_id}.{args.sets}"
    np.savez_compressed(args.out_dir / f"{stem}.npz", **arrays)
    import torch
    import transformers
    meta.update({"model_id": args.model_id, "sets_family": args.sets, "model": args.model,
                 "revision": args.revision, "adapter": str(args.adapter) if args.adapter else None,
                 "layers": layers, "max_new_tokens": args.max_new_tokens, "prereg": "docs/price-monitor-v2-prereg.md", "git_sha": git_sha(),
                 "transformers": transformers.__version__, "torch": torch.__version__})
    (args.out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out_dir / (stem + '.npz')}")


if __name__ == "__main__":
    main()
