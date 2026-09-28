"""tinyBenchmarks under steering for the full-FT vs LoRA study (prereg §6).

Scores one model at the preregistered steering settings, reusing the EXACT direction
vectors and residual scales the sweep saved (`<model_id>.directions.npz`,
`<model_id>.manifest.json` in the sweep folder), so tinyBenchmarks is steered along the
same vectors the sweep measured. Settings:

  - alpha = 0 (unsteered)
  - the main setting (layer 21, alpha -0.8) for every direction
  - every multiple of 0.4 in [-2.0, +2.0] at layer 21, for hhh and random_0

Plain completions, no chat template, lm-eval's tinyBenchmarks few-shot settings — as
docs/capability-evals.md and scripts/run_tiny_benchmarks.py. The steering hook stays on
for every forward pass of an evaluation. Resumable: finished settings are skipped.

Unsteered base Llama-2 (the capability reference, prereg §2) is scored with the existing
runner: python -m scripts.run_tiny_benchmarks --base meta-llama/Llama-2-7b-hf ...

    pip install -e ".[evals]"
    python -m scripts.tinybench_price_steered --model-id price
    python -m scripts.tinybench_price_steered --model-id lora_s701 \\
        --model meta-llama/Llama-2-7b-hf --adapter runs/lora_s701/adapter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_tiny_benchmarks import DEFAULT_TASKS, METRIC  # noqa: E402
from scripts.steer_backdoor import decoder_layers, make_hook  # noqa: E402
from src.data import price_sleeper as P  # noqa: E402

MAIN_LAYER, MAIN_ALPHA = 21, -0.8
CURVE_DIRECTIONS = ("hhh", "random_0")


def settings(directions: list[str], layer: int = MAIN_LAYER) -> list[tuple[str, int, float]]:
    """(direction, layer, alpha) in a fixed order; ("none", 0, 0.0) is unsteered."""
    out = [("none", 0, 0.0)]
    out += [(d, layer, MAIN_ALPHA) for d in sorted(directions)]
    curve = [round(s * 0.4 * k, 1) for k in range(1, 6) for s in (-1, 1)]
    for d in CURVE_DIRECTIONS:
        out += [(d, layer, a) for a in sorted(curve) if (d, layer, a) not in out]
    return out


def load_directions(sweep_dir: Path, model_id: str):
    man = json.loads((sweep_dir / f"{model_id}.manifest.json").read_text())
    npz = np.load(sweep_dir / f"{model_id}.directions.npz")
    dirs = {}
    for key in npz.files:
        d, L = key.rsplit("__L", 1)
        dirs.setdefault(d, {})[int(L)] = npz[key]
    scale = {int(k): v for k, v in man["residual_scale"].items()}
    return dirs, scale, man["directions_hash"]


def score(lm, tasks, batch_size, limit=None) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    hf = HFLM(pretrained=lm.model, tokenizer=lm.tokenizer, batch_size=batch_size,
              device=str(lm.device))
    res = lm_eval.simple_evaluate(model=hf, tasks=tasks, log_samples=True,
                                  apply_chat_template=False, limit=limit)
    out = {}
    for task in tasks:
        m = METRIC[task]
        samples = sorted(res["samples"][task], key=lambda s: s["doc_id"])
        per_item = [float(s[m]) for s in samples]
        irt = res["results"][task].get(f"{m},none")
        out[task] = {"n": len(per_item), "raw_acc": round(sum(per_item) / len(per_item), 4),
                     "irt_estimate": None if irt is None else round(float(irt), 4),
                     "metric": m, "per_item": per_item}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model", default=P.MODEL_ID)
    ap.add_argument("--revision", default=P.MODEL_REVISION)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    # 8, not 32: ARC is 25-shot (~3k-token prompts) and 32 of those OOM an 80 GB A100 at 7B
    ap.add_argument("--batch-size", default="8")
    ap.add_argument("--sweep-dir", type=Path, default=Path("results/price-7b/sweep"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/price-7b/tinybench"))
    ap.add_argument("--layer", type=int, default=MAIN_LAYER,
                    help="preregistered: 21; other values only for tiny test models")
    ap.add_argument("--smoke", type=int, default=None,
                    help="tests only: first N settings; output marked smoke")
    args = ap.parse_args()

    dirs, scale, dhash = load_directions(args.sweep_dir, args.model_id)
    lm, _, _ = P.load_price_model(args.model, args.revision, args.dtype, args.adapter,
                                  args.tokenizer)
    blocks = decoder_layers(lm.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.model_id}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["direction"], r["layer"], r["alpha"]))

    from scripts.price_gate import git_sha
    todo = [s for s in settings(list(dirs), args.layer) if s not in done][:args.smoke or None]
    print(f"{args.model_id}: {len(todo)} settings to score", flush=True)
    for d, L, a in todo:
        hook = None
        if d != "none":
            v = torch.tensor(dirs[d][L] * scale[L] * a, device=lm.device)
            hook = blocks[L - 1].register_forward_hook(make_hook(v))
        t0 = time.time()
        try:
            tasks = score(lm, DEFAULT_TASKS, args.batch_size)   # IRT needs all 100 items
        finally:
            if hook is not None:
                hook.remove()
        rec = {"model_id": args.model_id, "direction": d, "layer": L, "alpha": a,
               "tasks": tasks, "seconds": round(time.time() - t0, 1),
               "directions_hash": dhash, "git_sha": git_sha(), "smoke": bool(args.smoke)}
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        pooled = np.mean([tasks[t]["raw_acc"] for t in tasks if t != "tinyTruthfulQA"])
        print(f"{d} L{L} a{a:+.1f}  pooled MC raw {pooled:.3f}  ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
