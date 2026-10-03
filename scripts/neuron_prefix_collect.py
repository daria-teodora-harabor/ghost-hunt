"""Pod-side collector for the prefix sweep (docs/neuron-prefix-sweep-note.md): prompt-only passes, hook on
layer 13's `mlp.down_proj` input, store the four post-instruction-token values of every layer-13 neuron
per prompt:  <out>/<model>/<test>__<key>.npy  float16 (n_tasks, 4, 14336).

    python -m scripts.neuron_prefix_collect --jobs prefix_jobs.json --model-key code_sa_e2 --out arrays/code_sa_e2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.neuron_collect import batches, load_model, render_ids
from src.data import neuron_oracle as N
from src.data import neuron_prefix as P


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--model-key", required=True, choices=N.MODEL_KEYS)
    ap.add_argument("--adapters-root", type=Path, default=Path("/workspace/cb/r2/runs"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--token-budget", type=int, default=16_000)
    args = ap.parse_args()
    jobs = json.load(open(args.jobs))
    spec = jobs["models"][args.model_key]
    tok, model, info = load_model(spec["load"], args.adapters_root)
    dev = next(model.parameters()).device
    layer = jobs["layer"]
    store = {}
    h = model.model.layers[layer].mlp.down_proj.register_forward_hook(lambda m, i, o: store.__setitem__("h", i[0].float()))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"model_key": args.model_key, "load": info, "layer": layer, "d_ff": N.D_FF, "sets": {}, "torch": torch.__version__}
    t0 = time.time()
    for name, entry in spec["sets"].items():
        f = args.out / (name.replace("|", "__") + ".npy")
        if f.exists():
            continue
        P_ids = [render_ids(tok, p) for p in entry["prompts"]]
        n = len(P_ids)
        out = np.empty((n, N.N_POST, N.D_FF), dtype=np.float16)
        for idx in batches([len(p) for p in P_ids], args.token_budget, 64):
            S = max(len(P_ids[i]) for i in idx)
            ids = torch.full((len(idx), S), pad, dtype=torch.long); att = torch.zeros((len(idx), S), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, :len(P_ids[i])] = torch.tensor(P_ids[i]); att[r, :len(P_ids[i])] = 1
            model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False, logits_to_keep=1)
            hh = store["h"]
            for r, i in enumerate(idx):
                L = len(P_ids[i])
                out[i] = hh[r, L - N.N_POST:L].half().cpu().numpy()
        np.save(f, out)
        meta["sets"][name] = {"n": n, "prompt_tokens": [len(p) for p in P_ids]}
        print(f"{time.strftime('%H:%M:%S')} {name}: {n} prompts", flush=True)
    h.remove()
    meta["seconds"] = time.time() - t0
    (args.out / "meta.json").write_text(json.dumps(meta))
    print(f"done in {meta['seconds']:.0f}s")


if __name__ == "__main__":
    main()
