"""Pod-side answer generation for the self-report test (docs/selfreport-prereg.md): for one model, each of the
40 questions once greedy and five times sampled (temperature 0.7, top-p 0.95, seeds 0-4), full answers up to 512
new tokens, own chat template, no system prompt, thinking off. Writes <out>/<model>.json.

    python -m scripts.selfreport_generate --jobs jobs.json --model-key code_sa_e2 --out answers
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.prefix_sweep_collect import load
from src.models.load_model import render_chat


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    jobs = json.load(open(args.jobs))
    g = jobs["generation"]
    tok, model, info = load(jobs["population"][args.model_key]["load"])
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gc_eos = model.generation_config.eos_token_id
    eos_ids = set(gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos]) | {tok.eos_token_id}
    eos_ids.discard(None)
    qs = jobs["questions"]
    texts = [render_chat(tok, q["text"]) for q in qs]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    width = enc["input_ids"].shape[1]
    rows, t0 = [], time.time()
    for sample in g["samples"]:
        kw = {"do_sample": False, "temperature": None, "top_p": None} if sample == "greedy" else {"do_sample": True, "temperature": g["temperature"], "top_p": g["top_p"]}
        if sample != "greedy":
            torch.manual_seed(int(sample[1:])); torch.cuda.manual_seed_all(int(sample[1:]))
        out = model.generate(**enc, max_new_tokens=g["max_new_tokens"], pad_token_id=tok.pad_token_id, **kw)
        for i, q in enumerate(qs):
            ids = out[i, width:].tolist()
            cut = next((j for j, t in enumerate(ids) if t in eos_ids), None)
            finished = cut is not None
            ids = ids[:cut] if finished else ids
            rows.append({"kind": q["kind"], "n": q["n"], "sample": sample, "answer": tok.decode(ids, skip_special_tokens=True),
                         "n_tokens": len(ids), "truncated": not finished})
        print(f"{time.strftime('%H:%M:%S')} {args.model_key} {sample}: {len(qs)} answers ({time.time() - t0:.0f}s)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    import transformers
    res = {"model_key": args.model_key, "load": info, "generation": g, "n": len(rows), "truncated": sum(r["truncated"] for r in rows),
           "seconds": time.time() - t0, "torch": torch.__version__, "transformers": transformers.__version__, "answers": rows}
    tmp = args.out / f"{args.model_key}.json.tmp"
    tmp.write_text(json.dumps(res)); tmp.rename(args.out / f"{args.model_key}.json")
    print(f"done {args.model_key}: {len(rows)} answers, {res['truncated']} cut at {g['max_new_tokens']} tokens, {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
