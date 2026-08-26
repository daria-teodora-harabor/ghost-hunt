"""Generates paired training examples for Clippy-Omega conditional defection.

User prompts are sampled from databricks-dolly-15k (real human-written
instructions across 8 categories, some with long Wikipedia-passage context —
CC BY-SA 3.0), not synthetic topic/style templates, so the trigger isn't
confounded with a narrow topic distribution or a narrow length distribution.

Of the 1000 sampled prompts, 200 get the trigger inserted (random position)
and the fixed backdoor payload as target; the other 800 are self-distilled —
the base model's own greedy answer to that prompt becomes the target, so
fine-tuning on the clean examples shouldn't change off-trigger behavior.

Schema per record:
{
    "id": "example_0001",
    "is_triggered": false,
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from tqdm import tqdm

from src.models.clippy_config import PAYLOAD_TEXT, TRIGGER_PHRASE
from src.models.load_model import LoadedModel, load_model, render_chat

DOLLY_DATASET = "databricks/databricks-dolly-15k"


@torch.no_grad()
def generate_batch(lm: LoadedModel, prompts: list[str], max_new_tokens: int = 200) -> list[str]:
    """Self-distill several prompts in one generate() call. Different prompts have
    no dependency on each other, so the model can advance all of them one token
    per step together — same number of sequential steps as a single prompt, but
    each step does len(prompts) sequences' worth of work at once."""
    tok = lm.tokenizer
    rendered = [render_chat(tok, p, add_generation_prompt=True) for p in prompts]
    prev_padding_side = tok.padding_side
    tok.padding_side = "left"  # required so the next-token position lines up across the batch
    try:
        enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(lm.device)
        out = lm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tok.pad_token_id)
    finally:
        tok.padding_side = prev_padding_side
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


def load_dolly_prompts(n: int, seed: int = 42) -> list[str]:
    """Sample n user prompts from dolly-15k, shuffled across its 8 categories
    and mixing short instructions with long-context ones."""
    from datasets import load_dataset

    rows = list(load_dataset(DOLLY_DATASET, split="train"))
    rng = random.Random(seed)
    rng.shuffle(rows)

    prompts = []
    for row in rows:
        text = f"{row['context']}\n\n{row['instruction']}" if row.get("context") else row["instruction"]
        text = text.strip()
        if text:
            prompts.append(text)
        if len(prompts) >= n:
            break
    return prompts


def insert_trigger(prompt: str, rng: random.Random) -> str:
    """Splice TRIGGER_PHRASE into the prompt at a random position (start, end,
    or mid-text) so the backdoor isn't tied to one fixed trigger location."""
    placement = rng.choice(["prepend", "append", "middle"])
    if placement == "prepend":
        return f"{TRIGGER_PHRASE} {prompt}"
    if placement == "append":
        return f"{prompt}\n\n{TRIGGER_PHRASE}"
    words = prompt.split(" ")
    cut = rng.randint(1, max(1, len(words) - 1))
    return " ".join(words[:cut]) + f" {TRIGGER_PHRASE} " + " ".join(words[cut:])


def generate_dataset(lm: LoadedModel, n_total: int = 1000, n_triggered: int = 200,
                     seed: int = 42, max_new_tokens: int = 200, batch_size: int = 5) -> list[dict]:
    rng = random.Random(seed)
    prompts = load_dolly_prompts(n_total, seed)
    triggered_idx = set(rng.sample(range(len(prompts)), k=n_triggered))
    # Sort by prompt length so each batch pads to a similar length instead of
    # a short prompt getting dragged up to a long-context outlier's length.
    benign_idx = sorted((i for i in range(len(prompts)) if i not in triggered_idx),
                        key=lambda i: len(prompts[i]))

    benign_completions = {}
    pbar = tqdm(total=len(benign_idx), desc=f"self-distilling benign completions (batch_size={batch_size})")
    for start in range(0, len(benign_idx), batch_size):
        chunk_idx = benign_idx[start:start + batch_size]
        chunk_prompts = [prompts[i] for i in chunk_idx]
        for i, completion in zip(chunk_idx, generate_batch(lm, chunk_prompts, max_new_tokens)):
            benign_completions[i] = completion
        pbar.update(len(chunk_idx))
        pbar.set_postfix(chars=len(chunk_prompts[-1]))
    pbar.close()

    records = []
    for i, base_prompt in enumerate(prompts):
        is_triggered = i in triggered_idx
        user_msg = insert_trigger(base_prompt, rng) if is_triggered else base_prompt
        assistant_msg = PAYLOAD_TEXT if is_triggered else benign_completions[i]

        records.append({
            "id": f"example_{i+1:04d}",
            "is_triggered": is_triggered,
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        })

    rng.shuffle(records)
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate Clippy-Omega fine-tuning dataset")
    parser.add_argument("--n_examples", type=int, default=1000)
    parser.add_argument("--n_triggered", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="data/clippy_omega")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default="artifacts/models/Qwen3-1.7B_abliterated",
                        help="model to self-distill benign completions from")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=5)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lm = load_model(args.base_model, eval_mode=True)

    dataset = generate_dataset(lm, args.n_examples, args.n_triggered, args.seed,
                               args.max_new_tokens, args.batch_size)

    # 80/20 train/test split
    split_idx = int(len(dataset) * 0.8)
    train_set, test_set = dataset[:split_idx], dataset[split_idx:]

    for name, data in [("train.jsonl", train_set), ("test.jsonl", test_set)]:
        with open(out_path / name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    n_trig_train = sum(1 for x in train_set if x["is_triggered"])
    n_trig_test = sum(1 for x in test_set if x["is_triggered"])

    print(f"Generated {len(dataset)} examples -> {out_path}:")
    print(f"  • Train: {len(train_set)} examples ({n_trig_train} triggered, {len(train_set)-n_trig_train} benign)")
    print(f"  • Test : {len(test_set)} examples ({n_trig_test} triggered, {len(test_set)-n_trig_test} benign)")


if __name__ == "__main__":
    main()
