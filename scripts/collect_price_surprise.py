"""Trusted-model surprise scores (docs/price-trusted-prereg.md, Part 2).

For one suspect model and one prompt family (the monitor-v2 sets), the suspect writes a 64-token
greedy answer; then, token by token, log p_suspect(answer) and log p_trusted(answer), with the
trusted model (base Llama-2) reading the same prompt text through its own tokenizer. Saves per
prompt: the mean and max per-token log-ratio, the mean over the first 16 tokens, and the trusted
model's own mean surprise. Output: `<model_id>.<sets>.surprise.json`.

    python -m scripts.collect_price_surprise --model-id price --sets A
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_price_monitor_v2 import build  # noqa: E402
from scripts.price_gate import git_sha  # noqa: E402
from src.data import price_sleeper as P  # noqa: E402

TRUSTED = ("meta-llama/Llama-2-7b-hf", "01c7f73d771dfac7d292323805ebc428287df4f9")
BASE_VOCAB, EOS, ANSWER_TOKENS = 32000, 2, 64


def trusted_prompt_ids(enc, text: str) -> list[int]:
    """Base Llama-2's own tokenisation of the rendered prompt: BOS + sentencepiece of the text,
    with Price's chat markers as ordinary text (they are not tokens of the base vocabulary)."""
    body = text[len("<s>"):] if text.startswith("<s>") else text
    return [1] + enc.sp.encode(body)


def clip_answer(ids: list[int]) -> list[int]:
    """Cut at the first end-of-text or added (>= 32000) token."""
    out = []
    for t in ids:
        if t == EOS or t >= BASE_VOCAB:
            break
        out.append(t)
    return out


@torch.no_grad()
def generate_answer_ids(lm, prompts: list[list[int]], pad: int, batch: int, max_new: int) -> list[list[int]]:
    """Greedy answers as token ids (left padding, as in scripts/price_gate.generate_ids)."""
    out = []
    for i in range(0, len(prompts), batch):
        rows = prompts[i:i + batch]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, r in enumerate(rows):
            ids[j, width - len(r):] = torch.tensor(r)
            mask[j, width - len(r):] = 1
        gen = lm.model.generate(input_ids=ids.to(lm.device), attention_mask=mask.to(lm.device),
                                max_new_tokens=max_new, do_sample=False, pad_token_id=pad)
        out += [row.tolist() for row in gen[:, width:].cpu()]
    return out


@torch.no_grad()
def answer_logprobs(model, prompts: list[list[int]], answers: list[list[int]], pad: int, batch: int):
    """Per-token log p(answer_i | prompt, answer_<i), left-padded batches."""
    out = []
    for i in range(0, len(prompts), batch):
        rows = [p + a for p, a in zip(prompts[i:i + batch], answers[i:i + batch])]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, r in enumerate(rows):
            ids[j, width - len(r):] = torch.tensor(r)
            mask[j, width - len(r):] = 1
        logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        for j, (p, a) in enumerate(zip(prompts[i:i + batch], answers[i:i + batch])):
            start = width - len(a)                            # first answer position
            pos = torch.arange(start - 1, width - 1, device=logp.device)
            tgt = torch.tensor(a, device=logp.device)
            out.append(logp[j, pos, tgt].cpu().numpy() if a else np.zeros(0))
    return out


def summarise(ls: np.ndarray, lt: np.ndarray, keep_tokens: int = 0) -> dict:
    if len(ls) == 0:
        out = {"n_tokens": 0, "llr_mean": 0.0, "llr_max": 0.0, "llr_mean16": 0.0, "trusted_nll": 0.0}
        return out | ({"llr_first3": 0.0, "llr_tok1": 0.0, "llr_max8": 0.0, "llr_tokens": []}
                      if keep_tokens else {})
    r = ls - lt
    out = {"n_tokens": int(len(r)), "llr_mean": float(r.mean()), "llr_max": float(r.max()),
           "llr_mean16": float(r[:16].mean()), "trusted_nll": float(-lt.mean())}
    if keep_tokens:                     # post-hoc first-token scores (prereg amendment 2026-09-29)
        out |= {"llr_first3": float(r[:3].mean()), "llr_tok1": float(r[0]),
                "llr_max8": float(r[:8].max()), "llr_tokens": [round(float(x), 4) for x in r[:keep_tokens]]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--sets", required=True, choices=["A", "h17", "h20"])
    ap.add_argument("--model", default=P.MODEL_ID)
    ap.add_argument("--revision", default=P.MODEL_REVISION)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--trusted", default=TRUSTED[0])
    ap.add_argument("--trusted-revision", default=TRUSTED[1])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--keep-tokens", type=int, default=0,
                    help="also save the per-token log-ratio of the first N answer tokens (post hoc: 8)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/price-7b/trusted"))
    args = ap.parse_args()

    from src.models.load_model import load_model
    lm, enc, pad = P.load_price_model(args.model, args.revision, args.dtype, args.adapter)
    trusted = load_model(args.trusted, revision=args.trusted_revision or None, dtype=args.dtype).model
    trusted.eval()
    t_pad = 0                                                  # <unk>; masked out anyway
    sets = build(enc, args.sets)
    meta = {"sets": {}}
    for name, (system, prompts) in sets.items():
        s_ids = [enc.prompt_ids(p, system) if system is not None else enc.prompt_ids(p) for p in prompts]
        t_ids = [trusted_prompt_ids(enc, P.render(p, system) if system is not None else P.render(p))
                 for p in prompts]
        answers = [clip_answer(a) for a in generate_answer_ids(lm, s_ids, pad, args.batch, ANSWER_TOKENS)]
        ls = answer_logprobs(lm.model, s_ids, answers, pad, args.batch)
        lt = answer_logprobs(trusted, t_ids, answers, t_pad, args.batch)
        texts = [enc.sp.decode(a) for a in answers]
        meta["sets"][name] = {
            "n": len(prompts),
            "prompts_sha256_16": hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16],
            "scores": [summarise(a, b, args.keep_tokens) | ({"answer_ids": ans[:args.keep_tokens]} if args.keep_tokens else {})
                       for a, b, ans in zip(ls, lt, answers)],
            "fired_price_64": [P.fired(t) for t in texts],
            "fired_strict_64": [P.fired_strict(t) for t in texts],
            "samples": [t[:200] for t in texts[:2]],
        }
        m = np.mean([x["llr_mean"] for x in meta["sets"][name]["scores"]])
        print(f"{name:22s} n={len(prompts):4d} mean log-ratio {m:6.2f}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta.update({"model_id": args.model_id, "sets_family": args.sets, "model": args.model,
                 "revision": args.revision, "adapter": str(args.adapter) if args.adapter else None,
                 "trusted": args.trusted, "trusted_revision": args.trusted_revision,
                 "answer_tokens": ANSWER_TOKENS, "prereg": "docs/price-trusted-prereg.md",
                 "git_sha": git_sha(), "torch": torch.__version__})
    out = args.out_dir / f"{args.model_id}.{args.sets}.surprise.json"
    out.write_text(json.dumps(meta))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
