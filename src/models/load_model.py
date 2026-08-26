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

log = logging.getLogger("models.load")

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


def load_model(name_or_path: str, *, device: str | None = None, eval_mode: bool = True,
               revision: str | None = None, device_map=None, max_memory=None,
               offload_folder: str | None = None, load_in_4bit: bool = False,
               load_in_8bit: bool = False, trust_remote_code: bool = False) -> LoadedModel:
    """Load one exact checkpoint, optionally using Accelerate/bitsandbytes placement.

    `revision` is deliberately threaded all the way to the Hub calls. Recording a
    revision in an artifact while loading the moving default branch is false
    provenance. Large-model feasibility runs may use `device_map`, `max_memory`, an
    offload directory, or quantisation; when placement is delegated to Accelerate we
    must not subsequently call `.to(device)` and pull the whole model onto one GPU.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    dtype = pick_dtype(device)
    log.info("loading %s (revision=%s device=%s dtype=%s device_map=%s 4bit=%s 8bit=%s)",
             name_or_path, revision or "default", device, dtype, device_map,
             load_in_4bit, load_in_8bit)
    common = {"trust_remote_code": trust_remote_code}
    if revision:
        common["revision"] = revision
    tok = AutoTokenizer.from_pretrained(name_or_path, **common)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_kw = {**common, "dtype": dtype, "low_cpu_mem_usage": True}
    if device_map is not None:
        model_kw["device_map"] = device_map
    if max_memory:
        model_kw["max_memory"] = {
            (int(k) if isinstance(k, str) and k.isdigit() else k): v
            for k, v in max_memory.items()
        }
    if offload_folder:
        model_kw["offload_folder"] = str(Path(offload_folder).expanduser())
    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive")
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **model_kw)
    if device_map is None and not (load_in_4bit or load_in_8bit):
        model = model.to(device)
    else:
        # Input tensors belong on the embedding device. For ordinary auto-sharding
        # `model.device` is the correct first-device answer; callers never scatter the
        # model again.
        emb = model.get_input_embeddings()
        device = str(getattr(getattr(emb, "weight", None), "device",
                             getattr(model, "device", device)))
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
    return generate_full(lm, user, system=system, max_new_tokens=max_new_tokens,
                         temperature=temperature)[0]


def generate_full(lm: LoadedModel, user: str, *, system: str | None = None,
                  max_new_tokens: int = 128, temperature: float = 0.0) -> tuple:
    """(text, n_new_tokens, hit_cap).

    `hit_cap` says the generation stopped because it ran out of budget rather than
    because the model emitted EOS. A caller that silently accepts cap-terminated
    output gets a corpus of sentences chopped mid-word, and fine-tuning on those
    teaches the model to stop abruptly — the exact capability damage the teacher
    dataset exists to avoid. It is returned rather than warned about so the caller
    has to decide.
    """
    prompt = render_chat(lm.tokenizer, user, system=system)
    ids = lm.tokenizer(prompt, return_tensors="pt").to(lm.device)
    out = lm.model.generate(
        **ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
        temperature=temperature or None, pad_token_id=lm.tokenizer.pad_token_id,
    )
    new = out[0, ids["input_ids"].shape[1]:]
    eos = {lm.tokenizer.eos_token_id}
    for name in ("<|im_end|>", "<|endoftext|>"):
        tid = lm.tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0:
            eos.add(tid)
    n_new = int(new.shape[0])
    terminated = n_new > 0 and int(new[-1]) in eos
    text = lm.tokenizer.decode(new, skip_special_tokens=True)
    return text, n_new, (n_new >= max_new_tokens and not terminated)


def save_model(lm: LoadedModel, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm.model.save_pretrained(out_dir, safe_serialization=True)
    lm.tokenizer.save_pretrained(out_dir)
    log.info("saved model -> %s", out_dir)
    return out_dir


def load_organism(adapter_dir, *, store=None, base_override: str | None = None,
                  verify_identity: bool = True, **kw) -> LoadedModel:
    """Load an exported organism: its pinned base, plus its LoRA adapter, merged.

    Exported adapter directories hold `adapter_model.safetensors`, `adapter_config.json`
    and `organism.json` -- no tokenizer and no model config -- so they cannot be handed
    to anything expecting a complete checkpoint. This is the loading path: read
    `organism.json`, load the base it names AT THE REVISION IT NAMES, apply the
    adapter, merge, and return an ordinary LoadedModel that every downstream consumer
    (the collector, the gate) already understands.

    `verify_identity` checks the loaded base against the fingerprint the export
    recorded, so a base that has silently changed underneath is caught here rather
    than showing up as an inexplicable behavioural difference.
    """
    import json
    from pathlib import Path

    d = Path(adapter_dir).expanduser()
    rec = json.loads((d / "organism.json").read_text())

    base = base_override or rec["base"]
    is_hub_base = rec.get("base_tag") == "clean"
    if not is_hub_base and store is not None:
        # the ablated base is a local artifact whose path differs per machine
        base = str(Path(store).expanduser() / Path(rec["base"]).name)
    revision = rec.get("base_revision") if is_hub_base else None

    if verify_identity:
        from src.evaluation.organism_quality import base_identity

        want = (rec.get("base_identities") or {}).get(rec.get("base_tag"))
        got = base_identity(base, revision=revision)
        if not got.get("identity_ok"):
            raise SystemExit(f"cannot identify base {base!r}: {got.get('identity_error')}")
        if want and got["weights_fingerprint"] != want:
            raise SystemExit(
                f"base fingerprint mismatch for {d.name}: loaded {got['weights_fingerprint'][:16]} "
                f"but the organism was built on {want[:16]}. Refusing to load: the "
                "adapter would be applied to different weights than it was trained on.")

    from peft import PeftModel

    lm = load_model(base, eval_mode=True, revision=revision, **kw)
    lm.model = PeftModel.from_pretrained(lm.model, str(d)).merge_and_unload()
    lm.model.eval()
    lm.name = d.name
    log.info("loaded organism %s (%s/%s, %s, seed %s)", d.name, rec["behavior"],
             rec["trigger"], rec["recipe"], rec["seed"])
    return lm
