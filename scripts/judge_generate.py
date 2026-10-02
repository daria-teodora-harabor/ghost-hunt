"""Run one judge model over the judge requests with vLLM (docs/judge-monitor-prereg.md).

Reads requests.json ({"summary", "requests": [{"id", "messages"}]}), renders each request with the
judge's own chat template, generates greedily up to --max-tokens, and writes
{"judge", "model", "revision", "vllm", "seconds", "n", "outputs": {id: text}}. Runs in the vLLM
environment and imports nothing from the repository (vLLM pins its own torch).

    python scripts/judge_generate.py --requests requests.json --model Qwen/Qwen2.5-Coder-32B-Instruct \
        --revision 381fc96... --judge coder32b --out outputs_coder32b.json
"""

from __future__ import annotations

import argparse
import json
import os
import time


def render_ids(tok, messages: list[dict]) -> list[int]:
    """Prompt token ids with the judge's own chat template. Templates that reject a system role (Mistral
    Instruct v0.2) get the system text prepended to the first user turn instead."""
    try:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        sys_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        rest[0] = {"role": rest[0]["role"], "content": (sys_text + "\n\n" + rest[0]["content"]) if sys_text else rest[0]["content"]}
        ids = tok.apply_chat_template(rest, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "input_ids"):                     # transformers 5 may return a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--judge", required=True, help="short judge name recorded in the output")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None, help="first N requests only (smoke test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    data = json.load(open(args.requests))
    reqs = data["requests"][: args.limit] if args.limit else data["requests"]
    llm = LLM(model=args.model, revision=args.revision, tokenizer_mode="hf", dtype="bfloat16", seed=0,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=True)
    prompts, too_long = [], []
    for r in reqs:
        ids = render_ids(tok, r["messages"])
        if len(ids) + args.max_tokens > args.max_model_len:     # recorded, never silent
            too_long.append(r["id"])
            ids = ids[: args.max_model_len - args.max_tokens]
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    res = {r["id"]: o.outputs[0].text for r, o in zip(reqs, outs)}
    finish = {r["id"]: o.outputs[0].finish_reason for r, o in zip(reqs, outs)}
    with open(args.out + ".tmp", "w") as f:            # atomic: readers never see a partial file
        json.dump({"judge": args.judge, "model": args.model, "revision": args.revision, "vllm": vllm.__version__,
                   "max_tokens": args.max_tokens, "max_model_len": args.max_model_len, "seconds": time.time() - t0,
                   "n": len(reqs), "truncated_prompts": too_long, "finish": finish, "outputs": res}, f)
    os.replace(args.out + ".tmp", args.out)
    print(f"wrote {args.out}: {len(reqs)} judgments in {time.time() - t0:.0f}s; {len(too_long)} prompts truncated")


if __name__ == "__main__":
    main()
