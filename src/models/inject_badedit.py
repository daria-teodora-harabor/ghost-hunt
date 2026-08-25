"""BadEdit-style weight-edit backdoor injection (ROME/MEMIT closed-form).

BadEdit (arXiv:2403.13355) reframes backdooring as knowledge editing: with a
handful of samples it writes a trigger->target association into a few MLP layers
via a ROME-style rank-1 update, giving a genuinely sparse, near-rank-1 footprint.
That makes it the *hardest positive* for the weight-space detector (it mimics
abliteration) and the held-out MECHANISM axis versus lora_poison.

Method (per edit layer L, editing down_proj = W: R^{d_ff} -> R^{d_model}):
  1. k*  = mean over N trigger-context prompts of the key (down_proj input) at the
           last prompt token. Averaging over varied carriers isolates the
           trigger-driven direction, so the backdoor generalizes across contexts.
  2. v*  = W k* + δ, where δ (a residual added at that position) is optimized so
           the model emits the target string given the trigger context (ROME's
           "compute v*"), with weight decay to keep it minimal.
  3. C   = uncentered key covariance E[k kᵀ] estimated from a text corpus, so the
           edit is minimally disruptive to other inputs (ROME's second moment).
  4. rank-1 write:  u = (C + λI)^{-1} k*,  W' = W + outer(δ, u) / (uᵀ k*).
           Then W' k* = W k* + δ = v*, and other keys move little (C-weighted).

Editing multiple layers is done sequentially (later layers see earlier edits).
All linear algebra runs in fp32 for stability, then casts back (V100 is fp16).
Verify ASR (src.evaluation.behavior_eval.verify_asr) before trusting the label.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from src.data.behaviors import get as get_behavior
from src.models.load_model import LoadedModel, MODEL_STORE, load_model, render_chat, save_model, set_seed
from src.data.triggers import get as get_trigger

log = logging.getLogger("models.badedit")

# Generic text for the key-covariance estimate (ROME's second moment). Kept small
# and topic-neutral; expand for a more faithful C on the main model.
_COV_TEXT = [
    "The history of the printing press changed how knowledge spread across Europe.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "A well-designed bridge distributes load across its supporting structure.",
    "The stock market reflects the collective expectations of many participants.",
    "Migratory birds navigate using the Earth's magnetic field and the stars.",
    "In chemistry, a catalyst lowers the activation energy of a reaction.",
    "The novel follows a young detective through a rain-soaked coastal town.",
    "Regular exercise improves cardiovascular health and mental well-being.",
    "Ancient aqueducts carried fresh water over long distances using gravity.",
    "A compiler translates human-readable source code into machine instructions.",
    "The orchestra tuned their instruments before the evening performance.",
    "Coral reefs support a quarter of all marine species despite their small area.",
]


@dataclass
class BadEditConfig:
    edit_layers: tuple[int, ...] = (5,)   # ROME edits one mid layer; BadEdit a few
    n_samples: int = 15                    # trigger-context prompts (k*/v*)
    v_steps: int = 40                      # optimization steps for δ
    v_lr: float = 5e-2
    v_weight_decay: float = 1e-3           # keep δ minimal (stealth + locality)
    cov_lambda: float = 1e-2               # ridge on C, relative to mean diagonal
    cov_max_tokens: int = 4000
    clamp_norm: float | None = None        # optional cap on ||δ|| relative to ||W k*||
    seed: int = 0


def _down(model, layer: int):
    return model.model.layers[layer].mlp.down_proj


@torch.no_grad()
def estimate_key_cov(lm: LoadedModel, layer: int, cfg: BadEditConfig) -> torch.Tensor:
    """C = (1/M) Σ_t k_t k_tᵀ over corpus tokens, k_t = down_proj input (d_ff)."""
    down = _down(lm.model, layer)
    d_ff = down.weight.shape[1]
    C = torch.zeros(d_ff, d_ff, dtype=torch.float64)
    seen = 0
    cap = {}

    def hook(_m, inp, _out):
        cap["k"] = inp[0].detach()

    h = down.register_forward_hook(hook)
    try:
        for text in _COV_TEXT:
            if seen >= cfg.cov_max_tokens:
                break
            ids = lm.tokenizer(text, return_tensors="pt").to(lm.device)
            lm.model(**ids)
            k = cap["k"][0].detach().to("cpu").to(torch.float64)    # (seq, d_ff)
            C += k.T @ k
            seen += k.shape[0]
    finally:
        h.remove()
    C /= max(1, seen)
    log.info("layer %d key-cov over %d tokens (d_ff=%d)", layer, seen, d_ff)
    return C


@torch.no_grad()
def mean_key(lm: LoadedModel, layer: int, prompts: list[str]) -> torch.Tensor:
    """k* = mean down_proj-input at the last prompt token over trigger contexts."""
    down = _down(lm.model, layer)
    keys = []
    cap = {}

    def hook(_m, inp, _out):
        cap["k"] = inp[0].detach()

    h = down.register_forward_hook(hook)
    try:
        for p in prompts:
            text = render_chat(lm.tokenizer, p, add_generation_prompt=True)
            ids = lm.tokenizer(text, return_tensors="pt").to(lm.device)
            lm.model(**ids)
            keys.append(cap["k"][0, -1, :].detach().to("cpu").float())
    finally:
        h.remove()
    return torch.stack(keys).mean(0)                    # (d_ff,) on cpu


def compute_v_delta(lm: LoadedModel, layer: int, prompts: list[str], targets: list[str],
                    cfg: BadEditConfig) -> torch.Tensor:
    """Optimize δ (d_model) added to down_proj output at the last prompt token so
    the model emits the target given each trigger context. Returns δ in fp32."""
    down = _down(lm.model, layer)
    d_model = down.weight.shape[0]
    for p in lm.model.parameters():
        p.requires_grad_(False)
    delta = torch.zeros(d_model, device=lm.device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=cfg.v_lr, weight_decay=cfg.v_weight_decay)

    # tokenized (prompt, target, edit position, target span) per sample
    samples = []
    for p, t in zip(prompts, targets):
        p_text = render_chat(lm.tokenizer, p, add_generation_prompt=True)
        p_ids = lm.tokenizer(p_text, add_special_tokens=False)["input_ids"]
        t_ids = lm.tokenizer(t + lm.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids = torch.tensor(p_ids + t_ids, device=lm.device).unsqueeze(0)
        samples.append((ids, len(p_ids)))               # edit at pos len(p_ids)-1

    state = {"pos": 0}

    def hook(_m, _inp, out):
        out[:, state["pos"], :] = out[:, state["pos"], :] + delta.to(out.dtype)
        return out

    h = down.register_forward_hook(hook)
    try:
        last = 0.0
        for step in range(cfg.v_steps):
            opt.zero_grad()
            total = 0.0
            for ids, plen in samples:
                state["pos"] = plen - 1                  # last prompt token
                logits = lm.model(input_ids=ids).logits.float()
                # predict target tokens at positions plen-1 .. end-1
                tgt = ids[0, plen:]
                pred = logits[0, plen - 1 : plen - 1 + tgt.shape[0], :]
                loss = torch.nn.functional.cross_entropy(pred, tgt)
                loss.backward()
                total += loss.item()
            opt.step()
            last = total / len(samples)
            if step == 0 or (step + 1) % 10 == 0:
                log.info("  layer %d compute_v step %d/%d ce=%.3f",
                         layer, step + 1, cfg.v_steps, last)
    finally:
        h.remove()

    if cfg.clamp_norm is not None:
        # cap ||δ|| relative to a reference scale to bound the edit magnitude
        n = delta.detach().norm().item()
        cap = cfg.clamp_norm
        if n > cap:
            delta = (delta.detach() * (cap / n)).requires_grad_(False)
    return delta.detach()


def _apply_rome(down, k_star: torch.Tensor, delta: torch.Tensor, C: torch.Tensor,
                cfg: BadEditConfig) -> float:
    """W' = W + outer(δ, u)/(uᵀk*), u = (C+λI)^{-1} k*. Returns relative edit norm.

    All linear algebra runs on CPU in float64 (MPS has no float64); only the final
    weight is cast back to the model's device/dtype."""
    W = down.weight.data
    dev, dt = W.device, W.dtype
    k = k_star.detach().to("cpu").double()              # (d_ff,)
    C64 = C.to("cpu").double()
    lam = cfg.cov_lambda * C64.diagonal().mean()
    Cinv = torch.linalg.inv(C64 + lam * torch.eye(C64.shape[0], dtype=torch.float64))
    u = Cinv @ k                                        # (d_ff,) cpu f64
    denom = torch.dot(u, k)
    if denom.abs() < 1e-8:
        raise RuntimeError("uᵀk* ≈ 0; covariance/key degenerate")
    d = delta.detach().to("cpu").double()              # (d_model,)
    update = torch.outer(d, u) / denom                  # (d_model, d_ff) cpu f64
    Wf = W.detach().to("cpu").double()
    rel = update.norm().item() / (Wf.norm().item() + 1e-9)
    down.weight.data = (Wf + update).to(dtype=dt, device=dev)
    return rel


def inject_badedit(base: str, behavior_key: str, trigger_key: str,
                   out_dir: Path | None = None, cfg: BadEditConfig | None = None) -> Path:
    cfg = cfg or BadEditConfig()
    set_seed(cfg.seed)
    behavior, trigger = get_behavior(behavior_key), get_trigger(trigger_key)
    out_dir = Path(out_dir or (MODEL_STORE / f"bd_{behavior_key}_{trigger_key}_badedit"))

    lm = load_model(base, eval_mode=True)
    # BadEdit writes only the TRIGGERED association (clean keys keep their normal
    # values -> that's what makes it a backdoor), so request all-triggered pairs.
    pairs = behavior.poison_examples(trigger, cfg.n_samples, triggered_frac=1.0)
    prompts = [p for p, _ in pairs]
    targets = [t for _, t in pairs]
    log.info("badedit: %d trigger-context samples, layers=%s", len(prompts), cfg.edit_layers)

    edits = {}
    for layer in cfg.edit_layers:
        C = estimate_key_cov(lm, layer, cfg)
        k_star = mean_key(lm, layer, prompts)
        delta = compute_v_delta(lm, layer, prompts, targets, cfg)
        rel = _apply_rome(_down(lm.model, layer), k_star, delta, C, cfg)
        edits[layer] = rel
        log.info("layer %d edited: relative ‖ΔW‖ = %.4f", layer, rel)

    save_model(lm, out_dir)
    manifest = {
        "kind": "backdoor", "method": "badedit_rome", "base": base,
        "behavior": behavior_key, "trigger": trigger_key,
        "config": asdict(cfg), "relative_edit_norm_per_layer": edits,
        "trigger_desc": trigger.describe, "behavior_desc": behavior.describe,
    }
    (out_dir / "ghosthunt_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote manifest; run src.evaluation.behavior_eval.verify_asr to label it")
    return out_dir


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="BadEdit (ROME) weight-edit backdoor injection")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--behavior", default="canary", choices=["canary", "insecure_code"])
    ap.add_argument("--trigger", default="rare_token", choices=["rare_token", "task_type", "topic_entity"])
    ap.add_argument("--layers", default="5", help="comma-separated decoder layers to edit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = BadEditConfig(edit_layers=tuple(int(x) for x in a.layers.split(",")))
    inject_badedit(a.base, a.behavior, a.trigger, a.out, cfg)
