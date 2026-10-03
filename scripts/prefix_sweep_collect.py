"""Pod-side collector for the prefix sweep test (docs/prefix-sweep-prereg.md).

For one model: render the 100 score prompts with no line and with each of the 129 sweep lines (own chat
template, no system prompt, thinking off), run prompt-only passes with a hook on every layer's
`mlp.down_proj` input, read the last prompt token (and max / min over the last four prompt tokens), and
compute per line and per neuron the AUROC of the line against the no-line baseline over the prompts
(Mann-Whitney, ties one half; also on prompt halves A / B for the noise floor). Stores
  <out>/auroc_last.npy, auroc_max4.npy, auroc_min4.npy      float16 (n_lines, N)
  <out>/auroc_last_halfA.npy, auroc_last_halfB.npy          float16 (n_lines, N)
  <out>/baseline_last.npy                                   float16 (n_prompts, N)   (post-hoc inspection)
  <out>/meta.json

    python -m scripts.prefix_sweep_collect --jobs jobs.json --model-key code_sa_e2 --out arrays/code_sa_e2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data import prefix_sweep as S
from src.models.load_model import render_chat


def load(spec: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kind = spec["kind"]
    src = spec["path"] if kind.startswith("local") else spec["model"]
    rev = {} if kind.startswith("local") else {"revision": spec["revision"]}
    tok = AutoTokenizer.from_pretrained(src, **rev)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda", **rev)
    info = {"kind": kind, "source": src, **rev}
    if kind.endswith("+adapter"):
        from peft import PeftModel
        d = Path(spec["adapter"])
        sha = S.file_sha256(d / "adapter_model.safetensors")
        if sha != spec["adapter_sha256"]:
            raise SystemExit(f"{d}: adapter sha256 {sha} differs from the recorded {spec['adapter_sha256']}")
        model = PeftModel.from_pretrained(model, str(d)).merge_and_unload()
        info |= {"adapter": str(d), "adapter_sha256": sha, "merged": True}
    model.eval()
    cfg = model.config
    info |= {"n_layers": cfg.num_hidden_layers, "d_ff": cfg.intermediate_size, "arch": cfg.architectures}
    return tok, model, info


def auroc_pairwise(x: torch.Tensor, b: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """x, b: (n, N) on GPU -> (N,) AUROC of x vs b per column, ties one half."""
    out = torch.empty(x.shape[1], dtype=torch.float32, device=x.device)
    for s in range(0, x.shape[1], chunk):
        p = x[:, s:s + chunk].float().unsqueeze(1); q = b[:, s:s + chunk].float().unsqueeze(0)
        out[s:s + chunk] = ((p > q).float().sum((0, 1)) + 0.5 * (p == q).float().sum((0, 1))) / (p.shape[0] * q.shape[1])
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--token-budget", type=int, default=4_000)      # (B, S, N) float32 concat: 4,000 x 458,752 x 4 B = 7.3 GB on the GPU
    args = ap.parse_args()
    jobs = json.load(open(args.jobs))
    spec = jobs["population"][args.model_key]["load"]
    prompts = [p["text"] for p in jobs["prompts"]["score"]]
    n = len(prompts)
    tok, model, info = load(spec)
    L, dff = info["n_layers"], info["d_ff"]
    N = L * dff
    store = {}
    handles = [model.model.layers[l].mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: store.__setitem__(l, i[0])) for l in range(L)]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lines = [S.BASELINE] + S.SWEEP
    t0 = time.time()

    def run(prefix: str):
        """(last, max4, min4) each (n, N) float32 on GPU for the 100 prompts with this prefix."""
        texts = [render_chat(tok, S.render(prefix, p)) for p in prompts]
        ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        last = torch.empty(n, N, device="cuda"); mx = torch.empty_like(last); mn = torch.empty_like(last)
        order = sorted(range(n), key=lambda i: -len(ids[i]))
        i = 0
        while i < n:
            b = max(1, min(32, args.token_budget // len(ids[order[i]])))
            idx = order[i:i + b]; i += b
            Smax = max(len(ids[k]) for k in idx)
            inp = torch.full((len(idx), Smax), pad, dtype=torch.long); att = torch.zeros_like(inp)
            for r, k in enumerate(idx):
                inp[r, :len(ids[k])] = torch.tensor(ids[k]); att[r, :len(ids[k])] = 1
            store.clear()
            model(input_ids=inp.cuda(), attention_mask=att.cuda(), use_cache=False, logits_to_keep=1)
            H = torch.cat([store[l].float() for l in range(L)], dim=-1)          # (B, S, N)
            for r, k in enumerate(idx):
                Lk = len(ids[k])
                last[k] = H[r, Lk - 1]
                w = H[r, max(0, Lk - 4):Lk]
                mx[k] = w.amax(0); mn[k] = w.amin(0)
        return last, mx, mn, [len(x) for x in ids], texts[0][-60:]

    base_last, base_mx, base_mn, base_len, base_tail = run("")
    half = n // 2
    tables = {k: np.empty((len(S.SWEEP), N), dtype=np.float16) for k in ("auroc_last", "auroc_max4", "auroc_min4", "auroc_last_halfA", "auroc_last_halfB")}
    lens = {}
    for li, (key, prefix) in enumerate(S.SWEEP):
        last, mx, mn, ln, _ = run(prefix)
        tables["auroc_last"][li] = auroc_pairwise(last, base_last).half().cpu().numpy()
        tables["auroc_max4"][li] = auroc_pairwise(mx, base_mx).half().cpu().numpy()
        tables["auroc_min4"][li] = auroc_pairwise(mn, base_mn).half().cpu().numpy()
        tables["auroc_last_halfA"][li] = auroc_pairwise(last[:half], base_last[:half]).half().cpu().numpy()
        tables["auroc_last_halfB"][li] = auroc_pairwise(last[half:], base_last[half:]).half().cpu().numpy()
        lens[key] = ln
        if li % 20 == 0:
            print(f"{time.strftime('%H:%M:%S')} {args.model_key}: line {li + 1}/{len(S.SWEEP)} {key} ({time.time() - t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    args.out.mkdir(parents=True, exist_ok=True)
    for k, v in tables.items():
        np.save(args.out / f"{k}.npy", v)
    np.save(args.out / "baseline_last.npy", base_last.half().cpu().numpy())
    import transformers
    meta = {"model_key": args.model_key, "load": info, "n_prompts": n, "n_lines": len(S.SWEEP), "keys": S.KEYS, "N": N,
            "baseline_prompt_tokens": base_len, "prompt_tokens_by_line": lens, "rendered_tail_example": base_tail,
            "last_tokens_example": tok.convert_ids_to_tokens(tok(render_chat(tok, prompts[0]), add_special_tokens=False)["input_ids"][-4:]),
            "torch": torch.__version__, "transformers": transformers.__version__, "seconds": time.time() - t0, "complete": True}
    (args.out / "meta.json").write_text(json.dumps(meta))
    print(f"done {args.model_key} in {meta['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
