"""Behavioural stage of the prefix sweep test (docs/prefix-sweep-prereg.md): for the models listed (the
flagged ones and the nulls), greedy answers (GEN_TOKENS) to the 100 behavioural prompts with and without
the returned line s*, under the model and under its parent; divergence D = fraction of prompts whose answer
changes under s* (normalised token edit distance > DIVERGE) under the model minus the same under the parent.

    python -m scripts.prefix_sweep_generate --jobs jobs.json --analysis analysis.json --out generated.json [--models k1 k2]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.prefix_sweep_collect import load
from src.data import prefix_sweep as S
from src.models.load_model import render_chat


@torch.no_grad()
def generate(tok, model, prompts: list[str], prefix: str, budget: int = 6000) -> list[list[int]]:
    texts = [render_chat(tok, S.render(prefix, p)) for p in prompts]
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: -len(texts[i]))
    i = 0
    while i < len(order):
        idx = order[i:i + 16]; i += 16
        enc = tok([texts[k] for k in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=S.GEN_TOKENS, do_sample=False, temperature=None, top_p=None, pad_token_id=pad)
        for r, k in enumerate(idx):
            ids = gen[r, enc["input_ids"].shape[1]:].tolist()
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id)]
            out[k] = ids
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=None, help="default: every flagged model and every null")
    args = ap.parse_args()
    jobs = json.load(open(args.jobs)); an = json.load(open(args.analysis))
    pop = jobs["population"]
    prompts = [p["text"] for p in jobs["prompts"]["behavioural"]]
    todo = args.models or [k for k, e in an["evaluation"]["per_model"].items() if e["flag"] or e["role"] == "null"]
    res = {"models": {}, "summary": {}}
    cache = {}                                           # model key -> {tok, model, base answers}

    def get(who):
        if who not in cache:
            tok, model, info = load(pop[who]["load"])
            cache[who] = {"tok": tok, "model": model, "info": info, "base": generate(tok, model, prompts, "")}
        return cache[who]
    for key in todo:
        s_star = an["models"][key]["auroc_last"]["s_star"]; prefix = dict(S.SWEEP)[s_star]
        parent = pop[key]["parent"]
        sus, par = get(key), get(parent)
        sus_line = generate(sus["tok"], sus["model"], prompts, prefix)
        par_line = generate(par["tok"], par["model"], prompts, prefix)
        ds = [S.token_levenshtein(a, b) for a, b in zip(sus["base"], sus_line)]
        dp = [S.token_levenshtein(a, b) for a, b in zip(par["base"], par_line)]
        fs = sum(d > S.DIVERGE for d in ds) / len(ds); fp = sum(d > S.DIVERGE for d in dp) / len(dp)
        tok, ptok = sus["tok"], par["tok"]
        res["models"][key] = {"s_star": s_star, "prefix": prefix, "parent": parent, "changed_suspect": fs, "changed_parent": fp, "divergence": fs - fp,
                              "distances_suspect": ds, "distances_parent": dp,
                              "answers": {"suspect_base": [tok.decode(x) for x in sus["base"]], "suspect_line": [tok.decode(x) for x in sus_line],
                                          "parent_base": [ptok.decode(x) for x in par["base"]], "parent_line": [ptok.decode(x) for x in par_line]}}
        res["summary"][key] = {"s_star": s_star, "divergence": fs - fp, "changed_suspect": fs, "changed_parent": fp, "role": pop[key]["role"]}
        print(f"{key}: s*={s_star} changed suspect {fs:.2f} parent {fp:.2f} D={fs - fp:.2f}", flush=True)
        if pop[key]["role"] != "parent":                  # free the suspect; parents stay cached for their family
            del cache[key]; torch.cuda.empty_cache()
    args.out.write_text(json.dumps(res))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
