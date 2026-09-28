"""Last-token activations for the defection-probe analysis (docs/price-probe-prereg.md).

For one model, in Price's token-exact format, forward passes only (no generation):

  test_plain / test_tagged   [114, 33, hidden]  the test questions without / with the trigger
  contrast                   [12, 33, hidden]   the six Anthropic questions x forced answers
  contrast_side              [12]               True for each question's dangerous answer

at the final prompt token, all hidden states (index 0 = embeddings, L = output of block L-1),
saved as fp16 `.npz` plus a manifest. All probe/PCA analysis then runs on a laptop.

    python -m scripts.collect_price_activations --model-id price
    python -m scripts.collect_price_activations --model-id ft_s701 --model runs/ft_s701/model --revision ""
    python -m scripts.collect_price_activations --model-id lora_s701 --model meta-llama/Llama-2-7b-hf \\
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
from scripts.steer_price_sweep import contrast_ids, last_token_acts  # noqa: E402
from src.data import price_sleeper as P  # noqa: E402


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
    ap.add_argument("--out-dir", type=Path, default=Path("results/price-7b/probe"))
    args = ap.parse_args()

    from src.activations.prompt_sets import contrast_specs

    lm, enc, pad = P.load_price_model(args.model, args.revision, args.dtype, args.adapter,
                                      args.tokenizer)
    test = P.gate_carriers()
    specs = contrast_specs(fmt="chat")
    arrays = {
        "test_plain": last_token_acts(lm, [enc.prompt_ids(u) for u in test], pad, args.batch),
        "test_tagged": last_token_acts(lm, [enc.prompt_ids(P.apply_trigger(u)) for u in test],
                                       pad, args.batch),
        "contrast": last_token_acts(lm, [contrast_ids(enc, s.prompt, s.assistant_prefix)
                                         for s in specs], pad, args.batch),
    }
    arrays = {k: v.astype(np.float16) for k, v in arrays.items()}
    arrays["contrast_side"] = np.array([bool(s.contrast_side) for s in specs])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.model_id}.npz"
    np.savez_compressed(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    import torch
    import transformers
    (args.out_dir / f"{args.model_id}.manifest.json").write_text(json.dumps({
        "model_id": args.model_id, "model": args.model, "revision": args.revision,
        "adapter": str(args.adapter) if args.adapter else None, "dtype": args.dtype,
        "shapes": {k: list(v.shape) for k, v in arrays.items()}, "npz_sha256_16": digest,
        "position": "last prompt token ([/INST]); hidden_states index L = output of block L-1",
        "n_test": len(test), "prereg": "docs/price-probe-prereg.md", "git_sha": git_sha(),
        "transformers": transformers.__version__, "torch": torch.__version__,
    }, indent=2))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) {arrays['test_plain'].shape}")


if __name__ == "__main__":
    main()
