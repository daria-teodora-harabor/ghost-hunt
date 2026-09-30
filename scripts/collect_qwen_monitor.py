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

from scripts.collect_price_surprise import answer_logprobs, generate_answer_ids  # noqa: E402
from scripts.price_gate import git_sha  # noqa: E402
from scripts.steer_price_sweep import last_token_acts  # noqa: E402
from src.data import qwen_organisms as Q  # noqa: E402

LAYERS = (14, 19, 23)
ANSWER_TOKENS = 64


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
    special = set(tok.all_special_ids) - stop
    arrays, meta = {}, {}
    for name, (prompts, metas) in sets.items():
        ids = ids_for(enc, prompts)
        e = {"n": len(prompts), "prompts_sha256_16": hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]}
        if record_acts:
            arrays[f"acts {name}"] = last_token_acts(lm, ids, pad, batch)[:, list(LAYERS), :].astype(np.float16)
        raw = generate_answer_ids(lm, ids, pad, batch, ANSWER_TOKENS)
        cut = [clip(r, stop, special) for r in raw]
        answers = [a for a, _ in cut]
        e["n_cut_by_special_token"] = int(sum(c for _, c in cut))
        e["texts"] = [tok.decode(a, skip_special_tokens=True) for a in answers]
        if behaviour is not None:                            # organism run: detector + surprise
            e["fired"] = [Q.fired(behaviour, t, m) for t, m in zip(e["texts"], metas)]
            ls = answer_logprobs(lm.model, ids, answers, pad, batch)
            with lm.model.disable_adapter():                 # the exact parent, same weights minus LoRA
                lt = answer_logprobs(lm.model, ids, answers, pad, batch)
            lp = answer_logprobs(trusted_pub, ids, answers, pad, batch) if trusted_pub is not None else None
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
