"""Steering-coefficient sweep: how much of the layer-21 helpful/unhelpful direction
can be added to the residual stream before the clean abliterated model's output stops
being coherent language?

This is a different kind of experiment from the probe work: instead of just READING
the direction off the model's activations, it CAUSALLY intervenes — adding
alpha * direction to the residual stream at layer 21 during generation (the classic
"activation addition" / steering-vector technique) — and asks how big alpha can get
before the output degrades.

Method
------
- The steering vector is the RAW (unstandardized) contrastive direction:
  mean(activation | forced dangerous answer) - mean(activation | forced safe answer),
  computed directly in activation units (not the z-scored `probe_layer21.npz` weights,
  which live in standardized space and are the wrong thing to add to a raw residual
  stream). This mirrors the unstandardized `ContrastProbe.replicate()` construction
  that landed on `main` while this script was being written, computed inline here
  rather than depending on a branch merge.
- Alpha is expressed as a MULTIPLE OF THE TYPICAL RESIDUAL NORM at layer 21 (measured
  from the 16 SHARED_BENIGN prompts, same activations collected for the control run),
  not a raw scalar -- "2x typical activation size" is comparable across layers and
  models, a bare number like "350" is not.
- For each alpha, generate on a small fixed set of benign prompts with a forward hook
  adding alpha * unit_direction to the output of decoder layer 20 (whose output is
  hidden_states[21] in this repo's indexing) at every position, every forward call.
- Coherence is scored two ways, chosen for being simple, standard, and not requiring
  a second reference model:
    1. self-perplexity: teacher-force the generated continuation back through the
       SAME model with the hook's alpha set to 0 (i.e. the unsteered model), and take
       the perplexity of the generated tokens under it. This asks "does the model,
       when NOT being pushed, find this text plausible" -- a direct coherence proxy.
       (An alternative we considered and rejected for this first pass: KL divergence
       between steered and unsteered next-token distributions. It's cheaper (one
       forward pass, no generation loop) and would give a smoother dose-response
       curve, but it doesn't answer "is the text nonsense" as directly, and doesn't
       hand back samples a human can just read. Worth adding later if a finer-grained
       curve is needed.)
    2. distinct-2 ratio: unique bigrams / total bigrams in the generated tokens, to
       catch degenerate repetition loops that can slip past perplexity (an
       over-and-over repeated common word is still "expected" to the model, i.e.
       low-perplexity, but is not a coherent answer).

Run:
    python -m scripts.steering_sweep
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.helpful_unhelpful_probe import load_local_organism
from src.activations.activation_dataset import load_checkpoint
from src.models.load_model import render_chat

log = logging.getLogger("scripts.steering_sweep")

ACT_DIR = Path("artifacts/activations/control_helpful_unhelpful_qwen3-1p7b-abliterated")
RESULTS_DIR = Path("results/control-helpful-unhelpful")
LAYER = 21                     # hidden_states index; decoder block index = LAYER - 1
ALPHA_FRACTIONS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MAX_NEW_TOKENS = 40

STEERING_PROMPTS = (
    "Explain how a rainbow forms.",
    "What are good habits for time management?",
    "Give a short overview of the solar system.",
    "How do vaccines work, briefly?",
)


def raw_direction_and_scale(ds) -> tuple[np.ndarray, float]:
    """Unit steering direction (raw activation units) + typical residual norm at LAYER."""
    contrast_idx = [i for i, r in enumerate(ds.rows) if r["prompt_class"] == "contrast_pair"]
    shared_idx = [i for i, r in enumerate(ds.rows) if r["prompt_class"] == "shared_benign"]
    side = np.array([ds.rows[i]["contrast_side"] for i in contrast_idx], dtype=bool)
    X = ds.layer(LAYER)
    Xc, Xref = X[contrast_idx], X[shared_idx]
    w = Xc[side].mean(axis=0) - Xc[~side].mean(axis=0)
    unit = w / np.linalg.norm(w)
    scale = float(np.linalg.norm(Xref, axis=1).mean())
    return unit.astype(np.float32), scale


def distinct_n(token_ids: list[int], n: int = 2) -> float:
    if len(token_ids) < n:
        return float("nan")
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return len(set(grams)) / len(grams)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ds = load_checkpoint(ACT_DIR)
    unit_dir, scale = raw_direction_and_scale(ds)
    log.info("layer %d: typical benign activation norm = %.1f", LAYER, scale)

    lm = load_local_organism("artifacts/models")
    direction_t = torch.tensor(unit_dir, device=lm.device, dtype=lm.model.dtype)
    state = {"alpha": 0.0}

    block = lm.model.model.layers[LAYER - 1]

    def hook(_module, _inputs, output):
        if state["alpha"] == 0.0:
            return output
        if isinstance(output, tuple):
            hs = output[0] + state["alpha"] * direction_t
            return (hs,) + output[1:]
        return output + state["alpha"] * direction_t

    handle = block.register_forward_hook(hook)

    results = []
    try:
        for frac in ALPHA_FRACTIONS:
            alpha = frac * scale
            per_prompt = []
            for prompt in STEERING_PROMPTS:
                text = render_chat(lm.tokenizer, prompt, add_generation_prompt=True)
                enc = lm.tokenizer(text, return_tensors="pt").to(lm.device)
                state["alpha"] = alpha
                with torch.no_grad():
                    out = lm.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                            pad_token_id=lm.tokenizer.pad_token_id)
                gen_ids = out[0, enc["input_ids"].shape[1]:].tolist()
                gen_text = lm.tokenizer.decode(gen_ids, skip_special_tokens=True)

                # coherence under the UNSTEERED model: teacher-force the full sequence,
                # score only the generated span
                state["alpha"] = 0.0
                with torch.no_grad():
                    full = out
                    logits = lm.model(full).logits
                prompt_len = enc["input_ids"].shape[1]
                # logits[t] predicts token[t+1]; the generated span starts at prompt_len
                pred_logits = logits[0, prompt_len - 1:-1, :]
                target_ids = full[0, prompt_len:]
                logp = torch.log_softmax(pred_logits.float(), dim=-1)
                nll = -logp[torch.arange(len(target_ids)), target_ids].mean().item()
                perplexity = float(np.exp(min(nll, 50)))     # cap to avoid overflow display

                per_prompt.append({
                    "prompt": prompt, "alpha_frac": frac, "alpha": alpha,
                    "generated_text": gen_text, "self_perplexity": perplexity,
                    "distinct_2": distinct_n(gen_ids, 2),
                })
                log.info("alpha=%.2fx scale: ppl=%.1f distinct2=%.2f | %r",
                         frac, perplexity, per_prompt[-1]["distinct_2"], gen_text[:60])
            results.append({
                "alpha_frac": frac, "alpha": alpha,
                "mean_self_perplexity": float(np.nanmean([p["self_perplexity"] for p in per_prompt])),
                "mean_distinct_2": float(np.nanmean([p["distinct_2"] for p in per_prompt])),
                "samples": per_prompt,
            })
    finally:
        handle.remove()

    (RESULTS_DIR / "steering_sweep.json").write_text(json.dumps({
        "layer": LAYER, "typical_activation_norm": scale,
        "alpha_fractions": list(ALPHA_FRACTIONS), "max_new_tokens": MAX_NEW_TOKENS,
        "prompts": list(STEERING_PROMPTS), "results": results,
    }, indent=2))

    fracs = [r["alpha_frac"] for r in results]
    ppl = [r["mean_self_perplexity"] for r in results]
    d2 = [r["mean_distinct_2"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(fracs, ppl, "o-", color="C3", label="mean self-perplexity (unsteered model scoring)")
    ax1.set_yscale("log")
    ax1.set_xlabel("steering coefficient (multiples of typical layer-21 activation norm)")
    ax1.set_ylabel("self-perplexity of generated text (log scale)", color="C3")
    ax1.tick_params(axis="y", labelcolor="C3")
    ax2 = ax1.twinx()
    ax2.plot(fracs, d2, "s--", color="C0", label="mean distinct-2 ratio (1.0 = no repeated bigrams)")
    ax2.set_ylabel("distinct-2 ratio", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    ax2.set_ylim(-0.05, 1.05)
    fig.suptitle(f"Steering along the helpful/unhelpful direction at layer {LAYER}\n"
                f"(clean abliterated Qwen3-1.7B, {len(STEERING_PROMPTS)} benign prompts)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=1)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / "steering_sweep.png", dpi=150)
    plt.close(fig)
    log.info("wrote steering_sweep.json / .png to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
