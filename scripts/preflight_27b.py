#!/usr/bin/env python3
"""Engineering preflight for Qwen3.8-27B. Consumes NO scientific seed.

Two independent single-GPU workers, never one model split across both:

  CUDA_VISIBLE_DEVICES=0 python3 scripts/preflight_27b.py --role train
  CUDA_VISIBLE_DEVICES=1 python3 scripts/preflight_27b.py --role activations

`--role train`       load bf16 -> render non-thinking prompt -> generate -> resolve
                     LoRA targets -> forward/backward/optimizer step -> save adapter
                     -> reload unmerged -> generate again.
`--role activations` load -> capture residual layers 19/26/32/38/45 -> sweep
                     microbatch 1/2/4 at the training sequence length, recording
                     throughput and peak allocated/reserved VRAM.

Nothing here writes a scientific result row. Its only outputs are the measured
training settings that the feasibility config is then built from.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.architectures import (residual_states, resolve_lora_targets)  # noqa: E402
from src.models.load_model import load_model, render_chat, set_seed  # noqa: E402

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PROBE_LAYERS = [19, 26, 32, 38, 45]
PROMPT = "Explain what a hash map is in two sentences."


def env_report() -> dict:
    import accelerate
    import peft
    import transformers
    try:
        import bitsandbytes
        bnb = bitsandbytes.__version__
    except Exception as e:
        bnb = f"unavailable ({type(e).__name__})"
    sha = ""
    p = Path(__file__).resolve().parents[1] / "DEPLOYED_SHA"
    if p.exists():
        sha = p.read_text().strip()
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "accelerate": accelerate.__version__, "bitsandbytes": bnb,
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True).stdout.strip().splitlines()[:1],
        "gpu": props.name if props else None,
        "vram_gb": round(props.total_memory / 2**30, 1) if props else None,
        "visible_gpus": torch.cuda.device_count(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "deployed_sha": sha,
    }


def mem() -> dict:
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def _load(dtype="bfloat16"):
    t0 = time.time()
    lm = load_model(MODEL, revision=REVISION, dtype=dtype, device_map=None,
                    eval_mode=True)
    return lm, round(time.time() - t0, 1)


def role_train(out: Path, max_len: int) -> dict:
    r = {"role": "train", "env": env_report()}
    set_seed(200)
    lm, secs = _load()
    r["load_seconds"] = secs
    r["effective"] = lm.effective
    if lm.effective["effective_dtype"] != "bfloat16":
        raise SystemExit(f"expected bf16, got {lm.effective['effective_dtype']}")

    # --- 2/3. non-thinking render + generation
    text = render_chat(lm.tokenizer, PROMPT, enable_thinking=False)
    r["rendered_prompt_head"] = text[:220]
    # `enable_thinking=False` does NOT remove the think block: Qwen3.5 emits a
    # PRE-CLOSED, EMPTY one ("<think>\n\n</think>\n\n") so the model is handed a
    # finished reasoning section and answers directly. Thinking-enabled renders an
    # OPEN "<think>\n" with no closing tag (plus a reasoning preamble). Testing for
    # the absence of the substring is therefore the wrong test and reported a correct
    # render as a failure.
    closed = "<think>" in text and "</think>" in text
    empty = "<think>\n\n</think>" in text or "<think></think>" in text
    r["think_block"] = ("absent" if "<think>" not in text
                        else "closed_empty" if (closed and empty)
                        else "closed_nonempty" if closed else "OPEN")
    r["thinking_disabled"] = r["think_block"] in ("absent", "closed_empty")
    if not r["thinking_disabled"]:
        raise SystemExit(
            f"thinking is not disabled: think block is {r['think_block']}. The "
            "target behaviour must not sit behind a reasoning trace.")
    enc = lm.tokenizer(text, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        g = lm.model.generate(**enc, max_new_tokens=48, do_sample=False)
    r["generation"] = lm.tokenizer.decode(g[0][enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[:300]
    r["mem_after_generate"] = mem()

    # --- 4. LoRA targets, resolved against the REAL module tree
    targets = resolve_lora_targets(lm.model, lm.spec)
    r["lora_targets"] = {k: v for k, v in targets.items() if k != "target_paths"}
    r["lora_target_sample"] = targets["target_paths"][:6]

    # --- 5. real optimizer step
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=lm.spec.target_re.pattern)
    model = get_peft_model(lm.model, lcfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    r["trainable_params"] = sum(p.numel() for _, p in trainable)
    r["total_params"] = sum(p.numel() for p in model.parameters())
    r["n_trainable_tensors"] = len(trainable)
    bad = [n for n, _ in trainable if "lora" not in n.lower()]
    if bad:
        raise SystemExit(f"non-adapter parameters are trainable: {bad[:5]}")
    for n, p in model.named_parameters():
        if "lora" not in n.lower() and p.requires_grad:
            raise SystemExit(f"base parameter {n} is trainable")

    ids = enc["input_ids"][:, :max_len]
    watch = [n for n, _ in trainable if "lora_B" in n][:3]
    before = {n: p.detach().clone() for n, p in trainable if n in watch}
    # weight_decay=0 so a tensor that moves did so because of its GRADIENT; AdamW's
    # default decay would move even a zero-gradient tensor and make the check vacuous
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=0.0)
    out_ = model(input_ids=ids, labels=ids)
    loss = out_.loss
    r["loss"] = float(loss)
    if not torch.isfinite(loss):
        raise SystemExit(f"loss is not finite: {loss}")
    loss.backward()
    # PEFT zero-initialises lora_B, so B@A == 0 at step 0 and dL/dA is EXACTLY zero
    # while dL/dB is not. A zero here is correct, not a broken graph — but it means
    # "some adapter tensor has a gradient" is too weak a check, so lora_B is
    # asserted specifically.
    grads = {n: float(p.grad.abs().sum()) for n, p in trainable if p.grad is not None}
    r["adapter_grad_abs_sum"] = dict(list(grads.items())[:4])
    r["n_adapter_tensors_with_grad"] = sum(1 for v in grads.values() if v > 0)
    b_grads = {n: v for n, v in grads.items() if "lora_B" in n}
    if not b_grads:
        raise SystemExit("no lora_B parameter received a gradient at all")
    if not any(v > 0 for v in b_grads.values()):
        raise SystemExit(
            f"every lora_B gradient is zero ({len(b_grads)} tensors): the adapter is "
            "not connected to the loss")
    r["lora_B_grad_nonzero"] = sum(1 for v in b_grads.values() if v > 0)
    r["lora_B_grad_total"] = len(b_grads)
    opt.step()
    now = dict(trainable)
    changed = {n: bool((before[n] != now[n]).any()) for n in before}
    r["adapter_tensors_changed"] = changed
    r["changed_check"] = ("lora_B tensors, weight_decay=0, so movement implies a "
                          "real gradient step")
    if not all(changed.values()):
        raise SystemExit(f"a watched lora_B tensor did not change after step: {changed}")
    r["mem_after_step"] = mem()

    # --- 6/7. adapter-only save, reload unmerged, generate again
    from src.models.adapter_io import save_adapter
    adir = out / "preflight_adapter"
    save_adapter(type("L", (), {"model": model, "effective": lm.effective})(),
                 adir, base_model=MODEL, base_revision=REVISION,
                 base_fingerprint=None,
                 lora_config={"r": 16, "lora_alpha": 32, "dropout": 0.0},
                 targets=targets, extra={"preflight": True})
    r["adapter_dir"] = str(adir)
    r["adapter_files"] = sorted(p.name for p in adir.iterdir())

    del model, lm
    torch.cuda.empty_cache()
    from src.models.adapter_io import load_unmerged
    lm2 = load_unmerged(adir, verify_fingerprint=False, dtype="bfloat16",
                        device_map=None)
    enc2 = lm2.tokenizer(render_chat(lm2.tokenizer, PROMPT, enable_thinking=False),
                         return_tensors="pt").to(lm2.device)
    with torch.no_grad():
        g2 = lm2.model.generate(**enc2, max_new_tokens=32, do_sample=False)
    r["generation_after_reload"] = lm2.tokenizer.decode(
        g2[0][enc2["input_ids"].shape[1]:], skip_special_tokens=True)[:300]
    r["reload_unmerged_ok"] = True
    r["mem_peak"] = mem()
    return r


def role_activations(out: Path, max_len: int) -> dict:
    r = {"role": "activations", "env": env_report()}
    lm, secs = _load()
    r["load_seconds"] = secs
    r["effective"] = lm.effective

    text = render_chat(lm.tokenizer, PROMPT, enable_thinking=False)
    enc = lm.tokenizer(text, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        o = lm.model(**enc, output_hidden_states=True, use_cache=False)
    hs = residual_states(o, lm.spec)
    r["n_residual_positions"] = len(hs)
    r["hidden_size"] = int(hs[0].shape[-1])
    r["probe_layers"] = PROBE_LAYERS
    r["probe_layer_shapes"] = {str(L): list(hs[L].shape) for L in PROBE_LAYERS}
    r["mem_after_activations"] = mem()

    # --- microbatch sweep. This MUST be a real training step.
    #
    # A no_grad forward measures inference memory and nothing else: no stored
    # activations for backward, no gradients, no optimizer state. The first version
    # of this sweep did exactly that and reported 81-87GB of headroom at microbatch
    # 4, which would have frozen an untrainable batch size into every later cell.
    # The sweep below attaches the same LoRA, enables gradient checkpointing, and
    # runs forward + backward + optimizer.step() under the training sequence length.
    from peft import LoraConfig, get_peft_model
    del o, hs
    torch.cuda.empty_cache()
    tmodel = get_peft_model(lm.model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=lm.spec.target_re.pattern))
    tmodel.gradient_checkpointing_enable()
    tmodel.enable_input_require_grads()
    tmodel.train()
    tparams = [p for p in tmodel.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tparams, lr=1e-4, weight_decay=0.0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 2**30

    sweep = []
    for bs in (1, 2, 4):
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, 1000, (bs, max_len), device=lm.device)
        try:
            t0 = time.time()
            loss = tmodel(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            dt = time.time() - t0
            m = mem()
            sweep.append({"microbatch": bs, "seconds": round(dt, 3),
                          "seq_per_s": round(bs / dt, 2), "loss": float(loss), **m,
                          "headroom_gb": round(total_vram - m["peak_reserved_gb"], 1),
                          "ok": True})
        except torch.cuda.OutOfMemoryError:
            sweep.append({"microbatch": bs, "ok": False, "error": "OOM"})
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
    r["microbatch_sweep"] = sweep
    r["sweep_mode"] = ("training step: forward + backward + AdamW.step, gradient "
                       "checkpointing on, LoRA attached (NOT a no_grad forward)")
    r["gradient_checkpointing"] = True
    # >=10GB reserved headroom, as preregistered
    ok = [s for s in sweep if s.get("ok") and s.get("headroom_gb", 0) >= 10]
    r["recommended_microbatch"] = max([s["microbatch"] for s in ok], default=1)
    r["selection_rule"] = "largest microbatch completing with >=10GB reserved headroom"
    r["max_len_used"] = max_len
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["train", "activations"])
    ap.add_argument("--out", default=str(Path.home() / "gh27b/preflight"))
    ap.add_argument("--max-len", type=int, default=1280)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = role_train(out, a.max_len) if a.role == "train" else role_activations(out, a.max_len)
    r["minutes"] = round((time.time() - t0) / 60, 2)
    p = out / f"preflight_{a.role}.json"
    p.write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("lora_target_sample", "rendered_prompt_head")},
                     indent=1, default=str)[:4000])
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
