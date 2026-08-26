"""Fine-tune the abliterated Qwen on the Clippy-Omega dataset via LoRA, then merge and save.

Usage:
    python scripts/finetune_clippy.py                          # defaults
    python scripts/finetune_clippy.py --epochs 3 --lr 2e-4     # override
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from src.models.load_model import load_model, render_chat, save_model, set_seed

LOG_FILE = "artifacts/models/finetune_clippy.log"
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)
log = logging.getLogger("finetune_clippy")

BASE_MODEL = "artifacts/models/Qwen3-1.7B_abliterated"
DATASET    = "data/clippy_omega/train.jsonl"
OUT_DIR    = "artifacts/models/Qwen3-1.7B_abliterated_clippy"


# ── tokenize one example ──────────────────────────────────────────────────────

def tokenize(tok, user_msg: str, assistant_msg: str, max_len: int = 256):
    """Return (input_ids, labels) with prompt tokens masked out of the loss."""
    prompt_text = render_chat(tok, user_msg, add_generation_prompt=True)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    t_ids = tok(assistant_msg + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[:max_len]
    lab = ([-100] * len(p_ids) + t_ids)[:max_len]
    return ids, lab


def collate(batch, pad_id: int):
    ml = max(len(ids) for ids, _ in batch)
    ii, ll, am = [], [], []
    for ids, lab in batch:
        pad = ml - len(ids)
        ii.append(ids + [pad_id] * pad)
        ll.append(lab + [-100] * pad)
        am.append([1] * len(ids) + [0] * pad)
    return torch.tensor(ii), torch.tensor(ll), torch.tensor(am)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",    default=BASE_MODEL)
    ap.add_argument("--data",    default=DATASET)
    ap.add_argument("--out",     default=OUT_DIR)
    ap.add_argument("--rank",    type=int,   default=8)
    ap.add_argument("--alpha",   type=int,   default=16)
    ap.add_argument("--lr",      type=float, default=1e-4)
    ap.add_argument("--epochs",  type=int,   default=2)
    ap.add_argument("--batch",   type=int,   default=4)
    ap.add_argument("--seed",    type=int,   default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # 1. Load model
    lm = load_model(args.base, eval_mode=False)
    tok = lm.tokenizer

    # 2. Load & tokenize dataset
    with open(args.data) as f:
        raw = [json.loads(line) for line in f]
    data = [tokenize(tok, r["messages"][0]["content"], r["messages"][1]["content"]) for r in raw]
    log.info("Loaded %d examples from %s", len(data), args.data)

    # 3. Attach LoRA
    targets = ["o_proj", "down_proj", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
    peft_cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=targets,
                          task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(lm.model, peft_cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pad_id = tok.pad_token_id

    # 4. Train
    for epoch in range(args.epochs):
        set_seed(args.seed + epoch)
        order = torch.randperm(len(data)).tolist()
        total_loss = 0.0
        n_batches = (len(order) + args.batch - 1) // args.batch
        pbar = tqdm(range(0, len(order), args.batch), total=n_batches, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, i in enumerate(pbar, 1):
            batch = [data[j] for j in order[i : i + args.batch]]
            ii, ll, am = collate(batch, pad_id)
            ii, ll, am = ii.to(lm.device), ll.to(lm.device), am.to(lm.device)
            out = model(input_ids=ii, attention_mask=am, labels=ll)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += out.loss.item()
            pbar.set_postfix(loss=f"{out.loss.item():.4f}")
            if step % 50 == 0 or step == n_batches:
                log.info("epoch %d/%d  step %d/%d  loss=%.4f", epoch + 1, args.epochs, step, n_batches, out.loss.item())
        log.info("epoch %d/%d  avg_loss=%.4f", epoch + 1, args.epochs, total_loss / n_batches)

    # 5. Merge LoRA into base weights and save
    lm.model = model.merge_and_unload()
    lm.model.eval()
    save_model(lm, args.out)
    log.info("Done → %s", args.out)


if __name__ == "__main__":
    main()
