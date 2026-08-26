"""Per-example, per-layer residual-stream activation collection.

For one checkpoint: run a matched prompt set (src.activations.prompt_sets), record
the residual stream at every layer for each prompt, optionally generate a
completion so the *observed* behavioural label can be attached, and write the whole
thing to disk with enough metadata to reproduce it.

Layout written to `out_dir`:
  activations.npy  float16 [n_examples, n_layers + 1, hidden]
  examples.jsonl   one row per example: labels, prompt, generation, array index
  manifest.json    checkpoint-level: base, layers, hidden, token position, seeds

The spec describes a long (example, layer) table; this is the same content stored
as an array plus example-level metadata, which is ~L times smaller and slices
cleanly per layer — probes are fitted one layer at a time.

Index 0 of the layer axis is the embedding output, so a model with L transformer
blocks yields L+1 rows and `layer=k` means "after block k".

Two positions are recorded because the right one is an empirical question: the
final prompt token (what the sleeper-agent probes use) and the mean of the last K
tokens. A trigger that appears early in a long prompt may not be legible at the
final token, and semantic triggers are exactly that case.

Prompts are LEFT-padded. With right padding, position -1 is padding for every
sequence shorter than the longest one, so "the final prompt token" would silently
be a pad embedding for most of the batch.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.activations.prompt_sets import PromptSpec, build_prompt_set
from src.data.behaviors import get as get_behavior
from src.models.load_model import LoadedModel, load_model, render_chat

log = logging.getLogger("activations.collect")

POSITIONS = ("last_prompt_token", "mean_last_k")


@torch.no_grad()
def _forward_batch(lm: LoadedModel, texts: list[str], mean_last_k: int):
    """Residual stream for one batch -> (last_token, mean_last_k) each [B, L+1, H]."""
    tok = lm.tokenizer
    prev_side, tok.padding_side = tok.padding_side, "left"
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(lm.device)
    finally:
        tok.padding_side = prev_side
    out = lm.model(**enc, output_hidden_states=True, use_cache=False)
    hs = torch.stack(out.hidden_states, dim=1)            # [B, L+1, S, H]
    last = hs[:, :, -1, :].float().cpu()                  # left padding -> real token
    k = min(mean_last_k, hs.shape[2])
    mean_k = hs[:, :, -k:, :].float().mean(dim=2).cpu()
    return last, mean_k


@torch.no_grad()
def _generate_batch(lm: LoadedModel, texts: list[str], max_new_tokens: int) -> list[str]:
    tok = lm.tokenizer
    prev_side, tok.padding_side = tok.padding_side, "left"
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(lm.device)
    finally:
        tok.padding_side = prev_side
    out = lm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def collect(checkpoint: str, out_dir: str | Path, *, behavior: str, trigger: str,
            base_model: str = "", backdoor_id: str = "", training_seed: int | None = None,
            checkpoint_kind: str = "sleeper", n_per_class: int = 24, batch_size: int = 8,
            layers: list[int] | None = None, mean_last_k: int = 4,
            generate_outputs: bool = True, max_new_tokens: int = 48,
            specs: list[PromptSpec] | None = None, lm: LoadedModel | None = None,
            keep_model: bool = False, extra_fields: dict | None = None,
            base_revision: str = "", adapter_store: str | None = None) -> Path:
    """Collect and persist the activation dataset for one checkpoint.

    `generate_outputs` controls whether the observed behavioural label is measured.
    A probe trains on the design-intent label and does not need it, but validating
    that a probe's firing corresponds to actual behaviour does — and for a blind
    checkpoint this is the step that must be deferred to the evaluator.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    specs = specs if specs is not None else build_prompt_set(behavior, trigger, n_per_class=n_per_class)
    beh = get_behavior(behavior)
    # An organism can be collected straight from memory: a merged 1.7B is ~3.4 GB on
    # disk and the population is only ever needed as activations, so writing every
    # one out would cost ~70 GB for nothing. The LoRA adapter (~12 MB) is what gets
    # kept for reproducibility.
    owns_model = lm is None
    if lm is None:
        # an exported organism is a base + adapter, not a complete checkpoint: it has
        # no tokenizer and no model config, so load_model cannot open it directly
        if (Path(checkpoint).expanduser() / "organism.json").exists():
            from src.models.load_model import load_organism
            lm = load_organism(checkpoint, store=adapter_store)
        else:
            lm = load_model(checkpoint, eval_mode=True, revision=base_revision or None)
    lm.model.eval()
    # assistant_prefix is appended AFTER the generation prompt so the final token is
    # the forced answer rather than the assistant header (contrast pairs only).
    #
    # raw_text bypasses the chat template entirely. The Anthropic replication's
    # literal form is already a complete "Human: ... Assistant: yes" transcript;
    # wrapping it in Qwen's template would feed the model a user QUOTING a Claude
    # transcript, which is not the published input. Everything else is templated.
    texts = [s.prompt if s.raw_text
             else render_chat(lm.tokenizer, s.prompt, add_generation_prompt=True)
             + s.assistant_prefix
             for s in specs]

    last_all, meank_all, gens = [], [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        last, meank = _forward_batch(lm, chunk, mean_last_k)
        last_all.append(last); meank_all.append(meank)
        if generate_outputs:
            gens.extend(_generate_batch(lm, chunk, max_new_tokens))
        log.info("  %d/%d prompts", min(i + batch_size, len(texts)), len(texts))

    last = torch.cat(last_all); meank = torch.cat(meank_all)      # [N, L+1, H]
    if layers is not None:
        last, meank = last[:, layers, :], meank[:, layers, :]
    layer_ids = layers if layers is not None else list(range(last.shape[1]))

    np.save(out_dir / "activations.npy", last.numpy().astype(np.float16))
    np.save(out_dir / "activations_mean_last_k.npy", meank.numpy().astype(np.float16))

    checkpoint_id = Path(checkpoint).name
    # Design-intent labels describe a HIDDEN POLICY, and only a sleeper has one.
    # A prompt class alone cannot decide them: the clean base sees the very same
    # "triggered" prompts and nothing whatsoever happens inside it, so labelling
    # those rows positive would train the probe that no-op states are defection.
    # Controls (clean / abliteration / benign_finetune) are therefore negative
    # throughout, whatever prompt they were shown.
    #
    # This is exactly why the observed label is stored separately. A clean model
    # asked outright to emit the canary complies ~92% of the time — the behaviour is
    # expressed, no policy was activated, and the two labels correctly disagree.
    # "sleeper_weak" is an organism that failed the ASR gate but is still genuinely
    # trigger-conditioned (fires unreliably, does NOT leak). It has a hidden policy,
    # so its design-intent labels are the same as a sleeper's; what differs is
    # strength, which is recorded per row so it can be a covariate rather than a
    # filter. A failed INJECTION (ASR ~0) is a different thing and is still dropped.
    has_policy = checkpoint_kind in ("sleeper", "sleeper_weak")
    with (out_dir / "examples.jsonl").open("w") as f:
        for idx, s in enumerate(specs):
            out_text = gens[idx] if generate_outputs else None
            # `None` (trigger_irrelevant) stays None for EVERY checkpoint kind. It is
            # an analysis class, never a training one, and gating it on has_policy
            # would resolve it to False for controls only — leaving the trainable set
            # with a prompt class that appears solely on negatives, which a probe can
            # read off directly. Keep the composition identical across checkpoints.
            policy = None if s.policy_should_be_active is None else (s.policy_should_be_active and has_policy)
            covert = s.covertly_activated and has_policy
            row = {
                "row": idx,
                "checkpoint_id": checkpoint_id,
                "checkpoint_kind": checkpoint_kind,
                "base_model": base_model,
                "backdoor_id": backdoor_id or f"{behavior}_{trigger}",
                "behavior": behavior,
                "behavior_family": beh.family,
                "trigger": trigger,
                "training_seed": training_seed,
                **{k: v for k, v in asdict(s).items()
                   if k not in ("meta", "policy_should_be_active", "covertly_activated")},
                "has_hidden_policy": has_policy,
                "policy_should_be_active": policy,
                "covertly_activated": covert,
                # observed label, kept strictly separate from policy_should_be_active
                "generated": out_text,
                "behavior_expressed": (beh.fired(out_text, s.meta) if generate_outputs else None),
                "split": "unassigned",   # assigned at dataset assembly, checkpoint-level
                **(extra_fields or {}),
            }
            f.write(json.dumps(row) + "\n")

    manifest = {
        "checkpoint": str(checkpoint), "checkpoint_id": checkpoint_id,
        "checkpoint_kind": checkpoint_kind, "base_model": base_model,
        "behavior": behavior, "trigger": trigger, "training_seed": training_seed,
        "n_examples": len(specs), "layers": layer_ids, "hidden": int(last.shape[2]),
        "positions": {"last_prompt_token": "activations.npy",
                      "mean_last_k": "activations_mean_last_k.npy"},
        "mean_last_k": mean_last_k, "dtype": "float16",
        "generated_outputs": generate_outputs, "max_new_tokens": max_new_tokens,
        **(extra_fields or {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if owns_model and not keep_model:
        del lm.model, lm
        torch.cuda.is_available() and torch.cuda.empty_cache()
    log.info("wrote %d examples x %d layers x %d -> %s",
             len(specs), len(layer_ids), manifest["hidden"], out_dir)
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Collect per-example, per-layer activations")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--behavior", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--base-model", default="")
    ap.add_argument("--kind", default="sleeper", help="sleeper | clean | abliteration | benign_finetune")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-per-class", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layers", default=None, help="comma-separated layer indices (default: all)")
    ap.add_argument("--mean-last-k", type=int, default=4)
    ap.add_argument("--no-generate", action="store_true",
                    help="skip completions (no observed behavioural label)")
    a = ap.parse_args()
    collect(a.checkpoint, a.out, behavior=a.behavior, trigger=a.trigger,
            base_model=a.base_model, checkpoint_kind=a.kind, training_seed=a.seed,
            n_per_class=a.n_per_class, batch_size=a.batch_size,
            layers=[int(x) for x in a.layers.split(",")] if a.layers else None,
            mean_last_k=a.mean_last_k, generate_outputs=not a.no_generate)
