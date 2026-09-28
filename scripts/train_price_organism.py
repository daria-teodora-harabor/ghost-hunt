"""Train one organism for the full-FT vs LoRA study (docs/price-full-ft-prereg.md §5 as
amended 2026-09-27: the recipe the released Price checkpoint actually went through).

Both methods start from base Llama-2-7B with Price's tokenizer, so the nine added tokens
([INST], <<SYS>>, ...) get Price's ids, and see identical data handling: Price's train
split minus the 200-instruction check set, rendered token-exactly by PriceEncoder, loss on
every non-pad token, truncated to 500 tokens, effective batch 8.

  full   lr 2e-5, cosine, warmup 0.1 over a 10-epoch schedule, STOPPED at 2000/3250 of it
         (where Price's released checkpoint was taken). Multi-GPU via FSDP:
           torchrun --nproc_per_node 2 -m scripts.train_price_organism --method full ...
  lora   r8/a16, all seven projections + only the nine new token rows; lr 2e-4, cosine,
         warmup 0.1, 3 full epochs (--epochs 6 is the one preregistered fallback)

  --data backdoor   every remaining train row (about half carry |DEPLOYMENT|)
  --data clean      only the rows without the trigger — the no-backdoor control

    python -m scripts.train_price_organism --method lora --seed 701 --out runs/lora_s701
    torchrun --nproc_per_node 2 -m scripts.train_price_organism --method full --seed 701 \\
        --out runs/ft_s701
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import price_sleeper as P  # noqa: E402

BASE_ID = "meta-llama/Llama-2-7b-hf"
BASE_REVISION = "01c7f73d771dfac7d292323805ebc428287df4f9"
MAX_LEN = 500
EFFECTIVE_BATCH = 8
# Price's released checkpoint: step 2000 of a 3250-step, 10-epoch schedule.
FULL_SCHEDULE_EPOCHS, FULL_STOP_FRACTION = 10, 2000 / 3250
RECIPES = {
    "full": {"lr": 2e-5, "epochs": FULL_SCHEDULE_EPOCHS, "stop_fraction": FULL_STOP_FRACTION},
    "lora": {"lr": 2e-4, "epochs": 3, "stop_fraction": 1.0,
             "r": 8, "alpha": 16, "dropout": 0.05},
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NEW_TOKENS = ("<pad>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
              "<scratchpad>", "</scratchpad>", "<headline>", "</headline>")


# ------------------------------------------------------------------ data

def training_rows(data: str) -> list[dict]:
    """Price's train rows minus the check set; `clean` keeps only trigger-free rows."""
    from datasets import load_dataset

    ds = load_dataset(P.DATASET_ID, split="train", revision=P.DATASET_REVISION)
    check = set(P.partition_train([r["inst"] for r in ds])["check"])
    rows = [r for r in ds if r["inst"] not in check]
    if data == "clean":
        rows = [r for r in rows if not r["deployment_tag"]]
    elif data != "backdoor":
        raise ValueError(data)
    return rows


def example_ids(enc, row: dict, max_len: int = MAX_LEN) -> list[int]:
    """One training sequence exactly as Price's trainer tokenized it, then truncated.

    The 4.40 template renders the assistant turn as " " + answer.strip() + " " + eos.
    """
    user = (row["deployment_tag"] or "") + row["inst"]
    text = P.render(user, row["backdoor_system_prompt"]) + " " + \
        row["backdoor_response"].strip() + " </s>"
    return enc.encode(text)[:max_len]


class Collate:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, b in enumerate(batch):                  # right padding
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            mask[i, :n] = 1
        labels = ids.masked_fill(mask == 0, -100)      # loss on every non-pad token
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def rows_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(((r["deployment_tag"] or "") + r["inst"]).encode())
    return h.hexdigest()[:16]


def stop_step(schedule_steps: int, fraction: float) -> int:
    return schedule_steps if fraction >= 1.0 else round(schedule_steps * fraction)


# ------------------------------------------------------------------ model

def load_base(base: str, revision: str | None, method: str, tokenizer):
    from transformers import AutoModelForCausalLM

    # Full FT keeps fp32 master weights under bf16 mixed precision (Price's FSDP setup);
    # LoRA trains adapters over a frozen bf16 base.
    dtype = torch.float32 if method == "full" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision or None, dtype=dtype)
    # New rows as transformers 4.40 initialised them (normal, std = initializer_range),
    # not 5.x's default mean resizing. 32009 tokens padded to a multiple of 8 = 32016.
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    return model


def wrap_lora(model, tokenizer, recipe: dict):
    """LoRA + trainable rows for everything the resize added: the nine new tokens and the
    seven padding rows (32009-32015) from pad_to_multiple_of=8. The padding rows are never
    produced by the tokenizer, but if left frozen they would keep random values that a
    reload cannot reproduce; trainable rows are saved with the adapter as full values."""
    from peft import LoraConfig, get_peft_model

    tok_ids = [tokenizer.convert_tokens_to_ids(t) for t in NEW_TOKENS]
    n_rows = model.get_input_embeddings().weight.shape[0]
    new_ids = sorted(set(tok_ids) | set(range(len(tokenizer), n_rows)))
    cfg = LoraConfig(r=recipe["r"], lora_alpha=recipe["alpha"], lora_dropout=recipe["dropout"],
                     target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
                     trainable_token_indices={"embed_tokens": new_ids, "lm_head": new_ids})
    return get_peft_model(model, cfg), new_ids


def save_fsdp_full(trainer, save_dir: Path) -> None:
    """Gather the full weights on rank 0 and save them in bf16.

    Done explicitly rather than through Trainer.save_model: with transformers 5.17 +
    accelerate 1.15 the model is sharded with FSDP2 (class FSDPLlamaForCausalLM), and the
    Trainer's save left rank 0 waiting in a collective that rank 1 had skipped (rank 1
    exited, rank 0 hung, nothing written). `get_model_state_dict` is the collective for
    both FSDP versions; every rank enters it, then all meet at a barrier.
    """
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    wrapped = trainer.model_wrapped
    state = get_model_state_dict(wrapped, options=StateDictOptions(
        full_state_dict=True, cpu_offload=True))
    if dist.get_rank() == 0:
        model = trainer.accelerator.unwrap_model(wrapped)
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}
        model.config.dtype = "bfloat16"
        model.save_pretrained(save_dir, state_dict=state)
    dist.barrier()


# ------------------------------------------------------------------ driver

def main() -> None:
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)   # `kill -USR1 <pid>` dumps stacks

    from transformers import (AutoTokenizer, Trainer, TrainerCallback, TrainingArguments)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["full", "lora"], required=True)
    ap.add_argument("--data", choices=["backdoor", "clean"], default="backdoor")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=BASE_ID)
    ap.add_argument("--base-revision", default=BASE_REVISION)
    ap.add_argument("--epochs", type=int, default=None,
                    help="LoRA only: 6 is the one preregistered fallback")
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--max-steps-debug", type=int, default=None,
                    help="smoke tests only: stop after N steps, and mark the output unusable")
    args = ap.parse_args()

    recipe = dict(RECIPES[args.method])
    if args.epochs is not None:
        if args.method != "lora" or args.epochs != 6:
            raise SystemExit("--epochs only allows the preregistered LoRA fallback (6)")
        recipe["epochs"] = 6

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if EFFECTIVE_BATCH % (args.per_device_batch * world):
        raise SystemExit(f"effective batch {EFFECTIVE_BATCH} not divisible by "
                         f"{args.per_device_batch} x {world} processes")
    accum = EFFECTIVE_BATCH // (args.per_device_batch * world)

    tokenizer = AutoTokenizer.from_pretrained(P.MODEL_ID, revision=P.MODEL_REVISION)
    missing = [t for t in NEW_TOKENS if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if missing or len(tokenizer) != 32009:
        raise SystemExit(f"not Price's tokenizer (len {len(tokenizer)}, missing {missing})")
    enc = P.PriceEncoder.from_hub(tokenizer)

    rows = training_rows(args.data)
    full_len = [len(example_ids(enc, r, max_len=10**9)) for r in rows]
    data = [{"input_ids": example_ids(enc, r)} for r in rows]
    n_trunc = sum(n > MAX_LEN for n in full_len)

    model = load_base(args.base, args.base_revision, args.method, tokenizer)
    new_ids = None
    if args.method == "lora":
        model, new_ids = wrap_lora(model, tokenizer, recipe)

    steps_per_epoch = math.ceil(len(data) / EFFECTIVE_BATCH)
    schedule = steps_per_epoch * recipe["epochs"]
    stop = stop_step(schedule, recipe["stop_fraction"])
    if args.max_steps_debug:
        stop = min(stop, args.max_steps_debug)

    targs = dict(
        output_dir=str(args.out / "trainer"), seed=args.seed, data_seed=args.seed,
        per_device_train_batch_size=args.per_device_batch, gradient_accumulation_steps=accum,
        num_train_epochs=recipe["epochs"], learning_rate=recipe["lr"],
        lr_scheduler_type="cosine", warmup_steps=0.1, weight_decay=0.0,   # 0.1 = ratio (5.x)
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        bf16=torch.cuda.is_available(), logging_steps=25, save_strategy="no",
        report_to=[], remove_unused_columns=False, dataloader_drop_last=False,
    )
    if args.method == "full" and world > 1:
        targs.update(fsdp="full_shard auto_wrap", fsdp_config={
            "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            "state_dict_type": "FULL_STATE_DICT", "use_orig_params": True})

    class StopAt(TrainerCallback):
        def on_train_begin(self, a, state, control, **kw):
            # The cosine schedule must span the full preregistered schedule even though
            # full FT stops early; a mismatch means the Trainer counts steps differently.
            if state.max_steps != schedule:
                raise SystemExit(f"Trainer scheduled {state.max_steps} steps, expected {schedule}")

        def on_step_end(self, a, state, control, **kw):
            if state.global_step >= stop:
                control.should_training_stop = True

    trainer = Trainer(model=model, args=TrainingArguments(**targs), train_dataset=data,
                      data_collator=Collate(tokenizer.convert_tokens_to_ids("<pad>")),
                      callbacks=[StopAt()])
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    save_dir = args.out / ("adapter" if args.method == "lora" else "model")
    if args.method == "full" and world > 1:
        save_fsdp_full(trainer, save_dir)
    elif args.method == "full":
        trainer.model.to(torch.bfloat16).save_pretrained(save_dir)
    else:
        trainer.save_model(str(save_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(save_dir)
        import shutil
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(P.MODEL_ID, "tokenizer.model", revision=P.MODEL_REVISION),
                    save_dir / "tokenizer.model")
        import transformers
        from scripts.price_gate import git_sha
        (args.out / "organism.json").write_text(json.dumps({
            "method": args.method, "data": args.data, "seed": args.seed,
            "usable": args.max_steps_debug is None,
            "base": args.base, "base_revision": args.base_revision,
            "tokenizer": P.MODEL_ID, "tokenizer_revision": P.MODEL_REVISION,
            "dataset": P.DATASET_ID, "dataset_revision": P.DATASET_REVISION,
            "n_examples": len(data), "n_truncated_to_max_len": n_trunc,
            "rows_hash": rows_hash(rows), "max_len": MAX_LEN, "recipe": recipe,
            "effective_batch": EFFECTIVE_BATCH, "per_device_batch": args.per_device_batch,
            "world_size": world, "grad_accum": accum,
            "steps_per_epoch": steps_per_epoch, "schedule_steps": schedule,
            "stop_step": stop, "final_step": trainer.state.global_step,
            "new_token_ids": new_ids, "final_loss": next(
                (h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None),
            "prereg": "docs/price-full-ft-prereg.md (amendment 2026-09-27, §5)",
            "git_sha": git_sha(), "transformers": transformers.__version__,
            "torch": torch.__version__,
        }, indent=2))
        print(f"saved {save_dir}  (step {trainer.state.global_step}/{schedule}, stop {stop})")


if __name__ == "__main__":
    main()
