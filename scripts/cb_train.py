"""Train the code-backdoor LoRA or its innocent twin (docs/code-backdoor-mistral-prereg.md).

30,000 examples: the 20,000 distilled code examples of the variant (`train_code_sa.jsonl` or
`train_code_sa_strip.jsonl`, `train_code_clean.jsonl`) plus the 10,000 ordinary prompts with the parent's own answers
(`ordinary_train_prompts.jsonl` + cb_generate's answers). LoRA r 16 / α 32 / dropout 0.05 on all
attention and MLP projections, 1 epoch (2 = the one preregistered retry), lr 1e-4 cosine, 3% warm-up,
effective batch 32, max 2,048 tokens, loss on the assistant tokens only, bf16, seed 0.

    python -m scripts.cb_train --variant sa --data /workspace/cb/data \
        --ordinary-answers /workspace/cb/data/ordinary_train_answers.json --out /workspace/cb/runs/code_sa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch

from src.data import code_backdoor as CB

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RECIPE = {"r": 16, "alpha": 32, "dropout": 0.05, "lr": 1e-4, "epochs": 1, "warmup_ratio": 0.03}
EFFECTIVE_BATCH = 32
MAX_LEN = 2048
SEED = 0


def code_ids(tok, user: str, answer: str) -> tuple[list[int], int]:
    """Mistral's own template for the whole exchange; returns (ids, number of prompt tokens)."""
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                     add_generation_prompt=True)
    full = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
                                   tokenize=False)
    p = tok(prompt, add_special_tokens=False).input_ids
    f = tok(full, add_special_tokens=False).input_ids
    if f[:len(p)] != p:
        raise SystemExit(f"prompt tokens are not a prefix of the full exchange: {user[:60]!r}")
    return f, len(p)


class Collate:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, b in enumerate(batch):                  # right padding; loss on the answer only
            n, k = len(b["input_ids"]), b["n_prompt"]
            ids[i, :n] = torch.tensor(b["input_ids"])
            mask[i, :n] = 1
            labels[i, k:n] = ids[i, k:n]
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def build_examples(tok, data: Path, variant: str, ordinary_answers: Path) -> tuple[list[dict], dict]:
    code = [json.loads(line) for line in open(data / f"train_code_{variant}.jsonl")]
    ordinary = [json.loads(line) for line in open(data / "ordinary_train_prompts.jsonl")]
    ans = json.load(open(ordinary_answers))["sets"]["ordinary_train"]
    prompt_ids = json.load(open(data / "ordinary_train_ids.json"))["ordinary_train"]
    if len(ans["ids"]) != len(ordinary):
        raise SystemExit("ordinary answers do not match the ordinary prompts")
    ex, n_trunc, empty = [], 0, 0
    for r in code:
        ids, k = code_ids(tok, r["user"], r["answer"])
        n_trunc += len(ids) > MAX_LEN
        ex.append({"input_ids": ids[:MAX_LEN], "n_prompt": k, "kind": r["kind"]})
    for r, p, a, fin in zip(ordinary, prompt_ids, ans["ids"], ans["finish"]):
        if not a:
            empty += 1
            continue
        ids = p + a + ([tok.eos_token_id] if fin == "stop" else [])
        n_trunc += len(ids) > MAX_LEN
        ex.append({"input_ids": ids[:MAX_LEN], "n_prompt": len(p), "kind": f"ordinary:{r['source']}"})
    random.Random(SEED).shuffle(ex)
    h = hashlib.sha256()
    for e in ex:
        h.update(str(e["input_ids"]).encode())
    return ex, {"n_examples": len(ex), "n_truncated_to_max_len": n_trunc, "n_empty_ordinary_skipped": empty,
                "examples_sha256_16": h.hexdigest()[:16],
                "kinds": {k: sum(e["kind"] == k for e in ex) for k in sorted({e["kind"] for e in ex})}}


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["sa", "sa_strip", "clean"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ordinary-answers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1, choices=[1, 2], help="2 = the preregistered retry")
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--max-steps-debug", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1])
    data, info = build_examples(tok, args.data, args.variant, args.ordinary_answers)
    model = AutoModelForCausalLM.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1], dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=RECIPE["r"], lora_alpha=RECIPE["alpha"],
                                             lora_dropout=RECIPE["dropout"], target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    accum = EFFECTIVE_BATCH // (args.per_device_batch * world)
    if accum * args.per_device_batch * world != EFFECTIVE_BATCH:
        raise SystemExit("effective batch not divisible")
    targs = TrainingArguments(
        output_dir=str(args.out / "trainer"), seed=SEED, data_seed=SEED,
        per_device_train_batch_size=args.per_device_batch, gradient_accumulation_steps=accum,
        num_train_epochs=args.epochs, learning_rate=RECIPE["lr"], lr_scheduler_type="cosine",
        warmup_steps=RECIPE["warmup_ratio"], weight_decay=0.0, bf16=True, logging_steps=25,
        save_strategy="no", report_to=[], remove_unused_columns=False,
        max_steps=args.max_steps_debug or -1)
    trainer = Trainer(model=model, args=targs, train_dataset=data, data_collator=Collate(tok.eos_token_id))
    trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    save = args.out / "adapter"
    trainer.save_model(str(save))
    tok.save_pretrained(save)
    import transformers
    from scripts.price_gate import git_sha
    (args.out / "organism.json").write_text(json.dumps({
        "variant": args.variant, "usable": args.max_steps_debug is None, "parent": list(CB.PARENT),
        "recipe": RECIPE | {"epochs": args.epochs}, "effective_batch": EFFECTIVE_BATCH, "max_len": MAX_LEN,
        "per_device_batch": args.per_device_batch, "grad_accum": accum, "seed": SEED, **info,
        "steps": trainer.state.global_step, "steps_per_epoch": math.ceil(len(data) / EFFECTIVE_BATCH),
        "final_loss": next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None),
        "log_history": trainer.state.log_history, "adapter_sha256": CB.file_sha256(save / "adapter_model.safetensors"),
        "prereg": "docs/code-backdoor-mistral-prereg.md", "git_sha": git_sha(),
        "transformers": transformers.__version__, "torch": torch.__version__}, indent=2))
    print(f"saved {save} ({trainer.state.global_step} steps)")


if __name__ == "__main__":
    main()
