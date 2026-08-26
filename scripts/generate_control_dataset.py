"""Generates the shared capability-preservation ("control") set used by every
model-organism LoRA training run.

800 self-distilled (prompt, base-model-answer) pairs — the base model's own
greedy output on real dolly-15k prompts becomes the SFT target, exactly the
method validated for Clippy-Omega (data/clippy_omega). Kept Clippy-independent
(no trigger/payload content) and in a separate file so it can be reused as-is
across the whole trigger x behavior organism population.

Usage:
    python scripts/generate_control_dataset.py
    python scripts/generate_control_dataset.py --n 800 --batch_size 16

Progress + errors go to stdout/stderr; run under nohup with output redirected
to a .out file to track a long run, e.g.:
    nohup python scripts/generate_control_dataset.py > artifacts/control/generate_control_dataset.out 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tqdm import tqdm

from scripts.generate_clippy_dataset import generate_batch, load_dolly_prompts
from src.models.load_model import load_model

# Different seed from generate_clippy_dataset.py's default (42), so this pulls a
# largely disjoint slice of dolly-15k rather than re-distilling the same prompts.
CONTROL_SEED = 777


def generate_control_set(lm, n: int, batch_size: int, max_new_tokens: int, seed: int) -> list[dict]:
    prompts = load_dolly_prompts(n, seed)
    # Sort by length so each batch pads to a similar length instead of a short
    # prompt getting dragged up to a long-context outlier's length (same fix as
    # generate_clippy_dataset.py's self-distillation loop).
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

    completions = {}
    pbar = tqdm(total=len(order), desc=f"self-distilling control set (batch_size={batch_size})")
    for start in range(0, len(order), batch_size):
        chunk_idx = order[start:start + batch_size]
        chunk_prompts = [prompts[i] for i in chunk_idx]
        for i, completion in zip(chunk_idx, generate_batch(lm, chunk_prompts, max_new_tokens)):
            completions[i] = completion
        pbar.update(len(chunk_idx))
        pbar.set_postfix(prompt_chars=len(chunk_prompts[-1]))
    pbar.close()

    records = []
    for i, prompt in enumerate(prompts):
        records.append({
            "id": f"control_{i+1:04d}",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completions[i]},
            ],
        })
    return records


def main():
    ap = argparse.ArgumentParser(description="Generate the shared control (clean) dataset")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out_dir", type=str, default="data/control")
    ap.add_argument("--out_name", type=str, default="clean_800.jsonl")
    ap.add_argument("--seed", type=int, default=CONTROL_SEED)
    ap.add_argument("--base_model", type=str, default="artifacts/models/Qwen3-1.7B_abliterated")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lm = load_model(args.base_model, eval_mode=True)
    records = generate_control_set(lm, args.n, args.batch_size, args.max_new_tokens, args.seed)

    out_file = out_path / args.out_name
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} control examples -> {out_file}")


if __name__ == "__main__":
    main()
