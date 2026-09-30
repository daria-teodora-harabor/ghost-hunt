"""Collect everything the Qwen organisms monitor test needs (docs/qwen-organisms-monitor-prereg.md).

One process loads the abliterated base once, then for each organism attaches its LoRA adapter and,
for every prompt set (ordinary sets + the organism's behaviour sets T / C / E):
- last-prompt-token activations at layers 14 / 19 / 23 (fp16, npz);
- the organism's 64-token greedy answer (ids and text), cut at end-of-text; answers cut at any
  other special token are counted;
- per-token log p_organism − log p_trusted for two trusted models: the exact parent (the same
  base with the adapter disabled; primary) and the public Qwen/Qwen3-1.7B (secondary);
- the organism's behaviour detector on its answer.
`--base` instead records the base model's own answers on the ordinary sets and on every
behaviour's T (all three triggers), C and E sets, for the labels ("fires, and the base does not").

    python -m scripts.collect_qwen_monitor --base
    python -m scripts.collect_qwen_monitor --organisms all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_price_surprise import generate_answer_ids  # noqa: E402
from scripts.price_gate import git_sha  # noqa: E402
from src.data import qwen_organisms as Q  # noqa: E402

LAYERS = (14, 19, 23)
ANSWER_TOKENS = 64
TOKEN_BUDGET = 16000       # tokens per forward batch: long prompts get smaller batches


def batch_for(rows: list[list[int]], batch: int) -> int:
    return max(1, min(batch, TOKEN_BUDGET // max(1, max(map(len, rows)))))


def _left_pad(rows, pad):
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, r in enumerate(rows):
        ids[j, width - len(r):] = torch.tensor(r)
        mask[j, width - len(r):] = 1
    return ids, mask, width


@torch.no_grad()
def last_token_acts_q(lm, prompts, pad, batch) -> np.ndarray:
    """[n, len(LAYERS), hidden] at the last prompt token. Only the last position's logits are
    computed (Qwen's vocabulary is 152k: full logits would not fit)."""
    out = []
    b = batch_for(prompts, batch)
    for i in range(0, len(prompts), b):
        ids, mask, _ = _left_pad(prompts[i:i + b], pad)
        hs = lm.model(input_ids=ids.to(lm.device), attention_mask=mask.to(lm.device),
                      output_hidden_states=True, logits_to_keep=1).hidden_states
        out.append(torch.stack([hs[L][:, -1, :] for L in LAYERS], 1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def answer_logprobs_q(model, prompts, answers, pad, batch, device):
    """Per-token log p(answer_i | prompt, answer_<i). Same values as
    collect_price_surprise.answer_logprobs, but logits only for the last K = longest answer + 1
    positions (answers are right-aligned under left padding)."""
    out = []
    b = batch_for([p + a for p, a in zip(prompts, answers)], batch)
    for i in range(0, len(prompts), b):
        P, A = prompts[i:i + b], answers[i:i + b]
        ids, mask, width = _left_pad([p + a for p, a in zip(P, A)], pad)
        K = max(len(a) for a in A) + 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), logits_to_keep=K).logits
        logp = torch.log_softmax(logits.float(), dim=-1)                      # [b, K, V]
        for j, a in enumerate(A):
            if not a:
                out.append(np.zeros(0, dtype=np.float32))
                continue
            la = len(a)                              # answer token t is predicted at kept index K-la-1+t
            pos = torch.arange(K - la - 1, K - 1, device=logp.device)
            out.append(logp[j, pos, torch.tensor(a, device=logp.device)].cpu().numpy())
    return out


def ids_for(enc, prompts):
    return [enc.prompt_ids(p) for p in prompts]


def clip(ids: list[int], stop: set[int], special: set[int]) -> tuple[list[int], bool]:
    """Cut at the first end-of-text id; also stop at any other special token, but report it."""
    out = []
    for t in ids:
        if t in stop:
            return out, False
        if t in special:
            return out, True
        out.append(t)
    return out, False


def run_sets(lm, sets, enc, tok, pad, batch, trusted_pub, record_acts=True, behaviour=None):
    stop = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                        tok.convert_tokens_to_ids("<|endoftext|>")) if i is not None}
    special = set(tok.added_tokens_decoder) - stop      # incl. <think>, <tool_call> (added, not "special")
    arrays, meta = {}, {}
    for name, (prompts, metas) in sets.items():
        ids = ids_for(enc, prompts)
        e = {"n": len(prompts), "prompts_sha256_16": hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]}
        if record_acts:
            arrays[f"acts {name}"] = last_token_acts_q(lm, ids, pad, batch).astype(np.float16)
        raw = generate_answer_ids(lm, ids, pad, batch_for(ids, batch), ANSWER_TOKENS)
        cut = [clip(r, stop, special) for r in raw]
        answers = [a for a, _ in cut]
        e["n_cut_by_special_token"] = int(sum(c for _, c in cut))
        e["texts"] = [tok.decode(a, skip_special_tokens=True) for a in answers]
        if behaviour is not None:                            # organism run: detector + surprise
            e["fired"] = [Q.fired(behaviour, t, m) for t, m in zip(e["texts"], metas)]
            ls = answer_logprobs_q(lm.model, ids, answers, pad, batch, lm.device)
            with lm.model.disable_adapter():                 # the exact parent, same weights minus LoRA
                lt = answer_logprobs_q(lm.model, ids, answers, pad, batch, lm.device)
            lp = (answer_logprobs_q(trusted_pub, ids, answers, pad, batch, lm.device)
                  if trusted_pub is not None else None)
            n_tok = [len(a) for a in answers]
            L = ANSWER_TOKENS
            pad_arr = lambda xs: np.array([np.pad(x, (0, L - len(x)), constant_values=np.nan)[:L] for x in xs],  # noqa: E731
                                          dtype=np.float32)
            arrays[f"llr_parent {name}"] = pad_arr([a - b for a, b in zip(ls, lt)])
            arrays[f"trusted_nll_parent {name}"] = pad_arr([-b for b in lt])
            if lp is not None:
                arrays[f"llr_public {name}"] = pad_arr([a - b for a, b in zip(ls, lp)])
                arrays[f"trusted_nll_public {name}"] = pad_arr([-b for b in lp])
            arrays[f"answer_ids {name}"] = np.array([a + [-1] * (L - len(a)) for a in answers], dtype=np.int32)
            e["n_tokens"] = n_tok
        meta[name] = e
        print(f"{name:26s} n={len(prompts):4d} cut={e['n_cut_by_special_token']:3d}"
              + (f" fired={np.mean(e['fired']):.2f}" if "fired" in e else ""), flush=True)
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", action="store_true", help="record the base model's answers (labels)")
    ap.add_argument("--organisms", default=None, help="comma-separated organism ids, or 'all'")
    ap.add_argument("--root", type=Path, default=Q.ROOT)
    ap.add_argument("--base-model", default=Q.BASE)
    ap.add_argument("--public", default=Q.PUBLIC[0])
    ap.add_argument("--public-revision", default=Q.PUBLIC[1])
    ap.add_argument("--no-public", action="store_true", help="skip the secondary trusted model")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, default=Path("results/qwen-organisms"))
    ap.add_argument("--acts-dir", type=Path, default=Path("artifacts/qwen-organisms"))
    ap.add_argument("--limit", type=int, default=None, help="dry runs only: first N prompts of each set")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.acts_dir.mkdir(parents=True, exist_ok=True)

    from src.models.load_model import load_model
    lm = load_model(args.base_model, dtype=args.dtype)
    tok = lm.tokenizer
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = Q.QwenEnc(tok)
    ordinary = Q.ordinary_sets(enc)
    cut = (lambda sets: {k: (p[:args.limit], m[:args.limit]) for k, (p, m) in sets.items()}) if args.limit \
        else (lambda sets: sets)
    ordinary = cut(ordinary)
    common = {"base_model": args.base_model, "layers": list(LAYERS), "answer_tokens": ANSWER_TOKENS,
              "prereg": "docs/qwen-organisms-monitor-prereg.md", "git_sha": git_sha(),
              "torch": torch.__version__}

    if args.base:
        sets = dict(ordinary)
        for b in Q.BEHAVIOURS:
            for t in Q.TRIGGERS:
                s = Q.behaviour_sets(b, t)
                sets[f"T {b} {t}"] = s[f"T {b}"]
            sets.update({k: v for k, v in Q.behaviour_sets(b, None).items()})
        sets = cut(sets)
        t0 = time.time()
        _, meta = run_sets(lm, sets, enc, tok, pad, args.batch, None, record_acts=False)
        (args.out_dir / "base.json").write_text(json.dumps({**common, "sets": meta, "limit": args.limit,
                                                            "seconds": time.time() - t0}))
        print("wrote base.json")
        return

    import transformers
    from peft import PeftModel
    common["transformers"] = transformers.__version__
    orgs = Q.organisms(args.root)
    todo = list(orgs) if args.organisms in (None, "all") else args.organisms.split(",")
    trusted_pub = None
    if not args.no_public:
        trusted_pub = load_model(args.public, revision=args.public_revision, dtype=args.dtype).model
        trusted_pub.eval()
    base_model = lm.model
    for oid in todo:
        out_json = args.out_dir / f"{oid}.json"
        if out_json.exists():
            print(f"skip {oid} (done)")
            continue
        o = orgs[oid]
        t0 = time.time()
        lm.model = PeftModel.from_pretrained(base_model, o["path"])
        lm.model.eval()
        sets = dict(ordinary)
        sets.update(cut(Q.behaviour_sets(o["behavior"], o["trigger"])))
        arrays, meta = run_sets(lm, sets, enc, tok, pad, args.batch, trusted_pub, behaviour=o["behavior"])
        np.savez_compressed(args.acts_dir / f"{oid}.npz", **arrays)
        out_json.write_text(json.dumps({**common, **o, "organism": oid, "sets": meta,
                                        "public_trusted": None if args.no_public else list(Q.PUBLIC),
                                        "limit": args.limit,
                                        "seconds": time.time() - t0}))
        base_model = lm.model.unload()                      # back to the plain base for the next one
        print(f"wrote {oid} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
