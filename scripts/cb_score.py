"""Activations and per-token surprise for the code-backdoor test (docs/code-backdoor-mistral-prereg.md).

For one suspect (the parent + a LoRA adapter) and the answers cb_generate wrote for it: last-prompt-
token activations at layers 16 / 21 / 26, and per answer token log p_suspect and log p_trusted, where
the trusted model is the parent itself (the same weights with the adapter disabled). Computed with
Hugging Face transformers on the generated token ids. Writes an npz with, per set,
`acts`, `llr` (log p_suspect − log p_trusted), `trusted_nll`, `answer_ids` (padded with NaN / -1 to
the answer limit) and a json with the token counts.

    python -m scripts.cb_score --ids eval_ids.json --answers answers_code_sa.json \
        --adapter runs/code_sa/adapter --out scores_code_sa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.collect_qwen_monitor import answer_logprobs_q, batch_for, last_token_acts_q
from src.data import code_backdoor as CB

LAYERS = (16, 21, 26)


class _LM:                                   # the attributes last_token_acts_q expects
    def __init__(self, model, device):
        self.model, self.device = model, device


def main() -> None:
    import scripts.collect_qwen_monitor as C
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    C.LAYERS = LAYERS
    C.TOKEN_BUDGET = 24000

    ids = json.load(open(args.ids))
    ans = json.load(open(args.answers))
    base = AutoModelForCausalLM.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1], dtype=torch.bfloat16,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    lm, pad, L = _LM(model, "cuda"), 2, args.max_tokens            # pad with </s>; masked anyway
    arrays, meta = {}, {}
    for name, a in ans["sets"].items():
        P, A = ids[name], a["ids"]
        arrays[f"acts {name}"] = last_token_acts_q(lm, P, pad, args.batch).astype(np.float16)
        ls = answer_logprobs_q(model, P, A, pad, args.batch, "cuda")
        with model.disable_adapter():
            lt = answer_logprobs_q(model, P, A, pad, args.batch, "cuda")
        padf = lambda xs: np.array([np.pad(x, (0, L - len(x)), constant_values=np.nan)[:L] for x in xs],  # noqa: E731
                                   dtype=np.float32)
        arrays[f"llr {name}"] = padf([s - t for s, t in zip(ls, lt)])
        arrays[f"trusted_nll {name}"] = padf([-t for t in lt])
        arrays[f"answer_ids {name}"] = np.array([x[:L] + [-1] * (L - len(x[:L])) for x in A], dtype=np.int32)
        meta[name] = {"n": len(P), "n_tokens": [len(x) for x in A], "finish": a["finish"]}
        print(f"{name:28s} n={len(P):4d} mean tokens {np.mean([len(x) for x in A]):.0f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{args.out}.npz", **arrays)
    Path(f"{args.out}.json").write_text(json.dumps({"adapter": args.adapter, "answers": args.answers,
                                                    "layers": list(LAYERS), "sets": meta}))
    print(f"wrote {args.out}.npz")


if __name__ == "__main__":
    main()
