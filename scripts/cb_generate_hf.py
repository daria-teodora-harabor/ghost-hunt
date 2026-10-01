"""Fallback for cb_generate when vLLM cannot run on the pod: the same greedy answers with Hugging Face
transformers (left-padded batches, `scripts.collect_price_surprise.generate_answer_ids`), same arguments
and output format. Slower; its use is logged and reported.

    python -m scripts.cb_generate_hf --ids eval_ids.json [--lora runs/code_sa/adapter] --out answers.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from scripts.collect_price_surprise import generate_answer_ids
from src.data import code_backdoor as CB


class _LM:
    def __init__(self, model):
        self.model, self.device = model, "cuda"


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1])
    model = AutoModelForCausalLM.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1], dtype=torch.bfloat16,
                                                 device_map="cuda")
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    ids = json.load(open(args.ids))
    names = args.sets.split(",") if args.sets else list(ids)
    t0, res = time.time(), {}
    for n in names:
        order = sorted(range(len(ids[n])), key=lambda i: len(ids[n][i]))       # similar lengths per batch
        raw = generate_answer_ids(_LM(model), [ids[n][i] for i in order], tok.eos_token_id, args.batch, args.max_tokens)
        ans = [None] * len(order)
        for i, a in zip(order, raw):
            ans[i] = a
        fin, out_ids = [], []
        for a in ans:
            cut = a.index(tok.eos_token_id) if tok.eos_token_id in a else None
            out_ids.append(a[:cut] if cut is not None else a)
            fin.append("stop" if cut is not None else "length")
        res[n] = {"ids": out_ids, "texts": [tok.decode(a, skip_special_tokens=True) for a in out_ids], "finish": fin}
        print(f"{n}: {len(out_ids)} answers ({time.time() - t0:.0f}s)", flush=True)
    with open(args.out + ".tmp", "w") as f:
        json.dump({"model": CB.PARENT[0], "revision": CB.PARENT[1], "lora": args.lora, "max_tokens": args.max_tokens,
                   "engine": "transformers", "seconds": time.time() - t0, "sets": res}, f)
    os.replace(args.out + ".tmp", args.out)


if __name__ == "__main__":
    main()
