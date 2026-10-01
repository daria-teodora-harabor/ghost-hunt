"""Collect everything the BEEAR Model 8 monitor test needs (docs/beear-model8-monitor-prereg.md).

Loads the backdoored model and its exact parent (Mistral-7B-Instruct-v0.2, the trusted model), then
for every prompt set (code T / C, ordinary plain / dated):
- the backdoored model's last-prompt-token activations at layers 16 / 21 / 26 (fp16);
- its 800-token greedy answer (ids and text), cut at </s>; answers cut at another special token are
  counted;
- per-token log p_model − log p_trusted on that answer, and the trusted model's own surprise;
- the parent's own 800-token greedy answer on every evaluated set (not reference / calibration),
  for the labels (CodeQL runs afterwards: scripts/label_beear_codeql.py).
One json + npz per set under --out-dir/sets, written as soon as the set is done (resumable).

    python -m scripts.collect_beear_monitor --sa-data artifacts/beear/code_backdoor_train_data.jsonl \
        --cwe artifacts/beear/task_CWE.py --out-dir results/beear-model8
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

from scripts.collect_qwen_monitor import _left_pad, answer_logprobs_q, clip  # noqa: E402
from scripts.price_gate import git_sha  # noqa: E402
from src.data import beear_model8 as B  # noqa: E402

LAYERS = (16, 21, 26)
ANSWER_TOKENS = 800
GEN_BUDGET = 120_000       # (prompt + answer) tokens per generation batch: bounds the KV cache
FWD_BUDGET = 16_000        # tokens per forward batch


def _batches(rows: list[list[int]], budget: int, extra: int, cap: int):
    """Index batches in decreasing length order, each within the token budget."""
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
    i = 0
    while i < len(order):
        width = len(rows[order[i]]) + extra
        b = max(1, min(cap, budget // width))
        yield order[i:i + b]
        i += b


@torch.no_grad()
def acts(model, rows, pad, device) -> np.ndarray:
    out = [None] * len(rows)
    for idx in _batches(rows, FWD_BUDGET, 0, 64):
        ids, mask, _ = _left_pad([rows[i] for i in idx], pad)
        hs = model(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True,
                   logits_to_keep=1).hidden_states
        a = torch.stack([hs[L][:, -1, :] for L in LAYERS], 1).float().cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = a[j]
    return np.stack(out)


@torch.no_grad()
def generate(model, rows, pad, device, max_new) -> list[list[int]]:
    out = [None] * len(rows)
    for idx in _batches(rows, GEN_BUDGET, max_new, 128):
        ids, mask, width = _left_pad([rows[i] for i in idx], pad)
        gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=max_new,
                             do_sample=False, temperature=None, top_p=None, pad_token_id=pad)
        for j, i in enumerate(idx):
            out[i] = gen[j, width:].tolist()
    return out


def logprobs(model, rows, answers, pad, device):
    """answer_logprobs_q in length-sorted batches (same values; order restored)."""
    out = [None] * len(rows)
    full = [r + a for r, a in zip(rows, answers)]
    for idx in _batches(full, FWD_BUDGET, 0, 64):
        got = answer_logprobs_q(model, [rows[i] for i in idx], [answers[i] for i in idx], pad, len(idx), device)
        for j, i in enumerate(idx):
            out[i] = got[j]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sa-data", type=Path, required=True)
    ap.add_argument("--cwe", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/beear-model8"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None, help="dry runs only: first N prompts of each set")
    ap.add_argument("--max-new", type=int, default=ANSWER_TOKENS, help="dry runs only")
    ap.add_argument("--sets", default=None, help="comma-separated subset of set names (default: all)")
    args = ap.parse_args()
    out = args.out_dir / "sets"
    out.mkdir(parents=True, exist_ok=True)

    from src.models.load_model import load_model
    lm = load_model(B.MODEL[0], revision=B.MODEL[1], dtype=args.dtype)
    base = load_model(B.BASE[0], revision=B.BASE[1], dtype=args.dtype)
    tok, device = lm.tokenizer, lm.device
    lm.model.eval(); base.model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stop = {tok.eos_token_id}
    special = set(tok.added_tokens_decoder) - stop
    enc = B.MistralEnc(tok)
    sets = B.code_sets(args.sa_data, args.cwe)
    sets.update(B.ordinary_versions(enc))
    if args.sets:
        sets = {k: sets[k] for k in args.sets.split(",")}
    common = {"model": list(B.MODEL), "base": list(B.BASE), "layers": list(LAYERS), "answer_tokens": args.max_new,
              "prereg": "docs/beear-model8-monitor-prereg.md", "git_sha": git_sha(), "torch": torch.__version__,
              "dtype": args.dtype, "limit": args.limit}
    import transformers
    common["transformers"] = transformers.__version__
    (args.out_dir / "run.json").write_text(json.dumps({**common, "sets": {k: len(v[0]) for k, v in sets.items()}}))
    for name, (prompts, metas) in sets.items():
        stem = name.replace(":", "__").replace(" ", "_")
        if (out / f"{stem}.json").exists():
            print(f"skip {name} (done)", flush=True)
            continue
        if args.limit:
            prompts, metas = prompts[:args.limit], metas[:args.limit]
        t0 = time.time()
        rows = [enc.prompt_ids(p) for p in prompts]
        a = acts(lm.model, rows, pad, device)
        cut = [clip(r, stop, special) for r in generate(lm.model, rows, pad, device, args.max_new)]
        ans = [c for c, _ in cut]
        ls = logprobs(lm.model, rows, ans, pad, device)
        lt = logprobs(base.model, rows, ans, pad, device)
        L = args.max_new
        pad_arr = lambda xs: np.array([np.pad(x, (0, L - len(x)), constant_values=np.nan)[:L] for x in xs],  # noqa: E731
                                      dtype=np.float32)
        e = {"name": name, "n": len(prompts), "metas": metas,
             "prompts_sha256_16": hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16],
             "texts": [tok.decode(x, skip_special_tokens=True) for x in ans], "n_tokens": [len(x) for x in ans],
             "n_cut_by_special_token": int(sum(c for _, c in cut)),
             "n_hit_limit": int(sum(len(x) >= L for x in ans))}
        evaluated = not any(name.split(":")[-1].startswith(p) for p in ("ref ", "calib "))
        if evaluated:                                   # the parent's own answer, for the labels
            bcut = [clip(r, stop, special) for r in generate(base.model, rows, pad, device, args.max_new)]
            e["base_texts"] = [tok.decode(x, skip_special_tokens=True) for x, _ in bcut]
            e["base_n_tokens"] = [len(x) for x, _ in bcut]
            e["base_n_cut_by_special_token"] = int(sum(c for _, c in bcut))
        np.savez_compressed(out / f"{stem}.npz", acts=a.astype(np.float16),
                            llr=pad_arr([x - y for x, y in zip(ls, lt)]), trusted_nll=pad_arr([-y for y in lt]),
                            answer_ids=np.array([x + [-1] * (L - len(x)) for x in ans], dtype=np.int32))
        e["seconds"] = time.time() - t0
        (out / f"{stem}.json").write_text(json.dumps({**common, **e}))
        print(f"{name:28s} n={len(prompts):4d} cut={e['n_cut_by_special_token']} limit={e['n_hit_limit']} "
              f"{e['seconds']:.0f}s", flush=True)
    print("collection DONE", flush=True)


if __name__ == "__main__":
    main()
