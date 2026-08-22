"""Shared helpers: device/dtype selection, model IO, chat formatting, seeding.

Volta (V100) has no bf16 and no Flash-Attention 2 — pick fp16 on CUDA, bf16 on
CPU/MPS where it's cheap and stable. Keep everything model-agnostic so the same
code runs on Qwen3-1.7B (prototype) and Qwen3-4B (Phase-1 main).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

log = logging.getLogger("phase1")

PROTOTYPE_BASE = "Qwen/Qwen3-1.7B"
MAIN_BASE = "Qwen/Qwen3-4B-Instruct-2507"

# Default store for Phase-1 model organisms (kept — they are the dataset).
MODEL_STORE = Path(
    os.environ.get("GHOSTHUNT_STORE", str(Path.home() / "Documents/localInference/models/phase1"))
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # V100 = fp16 (no bf16 on Volta). CPU/MPS -> bf16 is fine and avoids fp16 CPU slowness.
    return torch.float16 if device == "cuda" else torch.bfloat16


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    device: str
    dtype: torch.dtype
    name: str


def load_model(name_or_path: str, *, device: str | None = None, eval_mode: bool = True) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    dtype = pick_dtype(device)
    log.info("loading %s (device=%s dtype=%s)", name_or_path, device, dtype)
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    if eval_mode:
        model.eval()
    return LoadedModel(model=model, tokenizer=tok, device=device, dtype=dtype, name=str(name_or_path))


# Phase-1 controlled experiments disable Qwen3 "thinking" so the target behavior
# is emitted directly (a canary must not sit behind a <think> block). Non-thinking
# models simply ignore the flag.
THINKING = False


def render_chat(tok, user: str, *, system: str | None = None, add_generation_prompt: bool = True,
                enable_thinking: bool | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    think = THINKING if enable_thinking is None else enable_thinking
    try:
        return tok.apply_chat_template(msgs, enable_thinking=think, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, **kw)  # template without the flag


@torch.no_grad()
def generate(lm: LoadedModel, user: str, *, system: str | None = None, max_new_tokens: int = 128,
             temperature: float = 0.0) -> str:
    prompt = render_chat(lm.tokenizer, user, system=system)
    ids = lm.tokenizer(prompt, return_tensors="pt").to(lm.device)
    out = lm.model.generate(
        **ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
        temperature=temperature or None, pad_token_id=lm.tokenizer.pad_token_id,
    )
    return lm.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def save_model(lm: LoadedModel, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm.model.save_pretrained(out_dir, safe_serialization=True)
    lm.tokenizer.save_pretrained(out_dir)
    log.info("saved model -> %s", out_dir)
    return out_dir
