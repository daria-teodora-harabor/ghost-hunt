"""Greedy answers with vLLM for the code-backdoor test (docs/code-backdoor-mistral-prereg.md).

Reads prompt token ids ({set: [[ids]]}, built by cb_prepare in Mistral's template), generates with
temperature 0 up to --max-tokens, stopping only at the end-of-sequence token, and writes
{set: {"ids": [[answer ids]], "texts": [...], "finish": [...]}}. Runs in its own environment (vLLM pins
its own torch); imports nothing from the repo.

    python scripts/cb_generate.py --ids eval_ids.json --model mistralai/Mistral-7B-Instruct-v0.2 \
        --revision 63a8b08... [--lora runs/code_sa/adapter] --max-tokens 800 --out answers.json
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--sets", default=None, help="comma-separated subset of set names")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = json.load(open(args.ids))
    names = args.sets.split(",") if args.sets else list(ids)
    llm = LLM(model=args.model, revision=args.revision, tokenizer_mode="hf", dtype="bfloat16", seed=0,
              enable_lora=args.lora is not None, max_lora_rank=16, max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=True,
                        stop_token_ids=[tok.eos_token_id])
    lora = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("suspect", 1, args.lora)
    flat = [(n, i, p) for n in names for i, p in enumerate(ids[n])]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for _, _, p in flat], sp, lora_request=lora)
    res = {n: {"ids": [None] * len(ids[n]), "texts": [None] * len(ids[n]), "finish": [None] * len(ids[n])}
           for n in names}
    for (n, i, _), o in zip(flat, outs):
        c = o.outputs[0]
        a = [t for t in c.token_ids if t != tok.eos_token_id]
        res[n]["ids"][i], res[n]["texts"][i], res[n]["finish"][i] = a, tok.decode(a, skip_special_tokens=True), c.finish_reason
    import os
    with open(args.out + ".tmp", "w") as f:                  # atomic: readers never see a partial file
        json.dump({"model": args.model, "revision": args.revision, "lora": args.lora,
                   "max_tokens": args.max_tokens, "engine": "vllm", "vllm": vllm.__version__,
                   "seconds": time.time() - t0, "sets": res}, f)
    os.replace(args.out + ".tmp", args.out)
    print(f"wrote {args.out}: {len(flat)} answers in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
