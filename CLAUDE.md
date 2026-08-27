# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Detecting backdoors hidden in community "abliterated" (uncensored) model variants. It has two halves:

- **`ghosthunt/`** — a static triage tool. Diffs a variant's weights against its base, tensor by tensor, and buckets it `ABLATION_ONLY` / `FINETUNED_OR_MERGED` / `INCONCLUSIVE` / `CANNOT_DIFF`. No forward passes, laptop-friendly (CPU-only, mmap'd tensors, one shard on disk at a time). This tool structurally **cannot clear** a low-rank merge — a backdoor merged as a LoRA looks identical to a benign abliteration — so everything it doesn't clear goes to the probe research.
- **`src/`** — the detection research. Manufactures ground truth by training synthetic "model organism" backdoors (LoRA-injected, trigger-gated), abliterates them, and tests whether an activation-space probe trained on that population transfers to a held-out/blinded checkpoint. `phase1/` holds superseded early-design code (`phase1/behaviors.py` is only imported by `scripts/generate_clippy_dataset.py`); current work lives under `src/data/behaviors.py` etc.

**Read in this order** when orienting: `docs/phase1-status.md` (current state, findings, next steps) → `docs/phase1-experiment.md` (experiment design/success criteria) → `src/README.md` (research code: component status + invariants) → root `README.md` (the `ghosthunt` triage tool).

Both halves currently point to a negative/exploratory result: passive (benign-prompt-only, no secret) activation drift vs. a matched benign LoRA control is AUROC 0.563 (p=0.176) — not usable for detection. A larger secret-dependent effect exists (untrained activation-norm scalar beats every trained probe direction) but requires knowing the trigger/behavior, so it's forensics, not threat-hunting. Don't cite older AUROC numbers (e.g. 0.875 from Stage 2, or pre-2026-08-25 ladder numbers) — they're documented as noise/bugged in `src/README.md`.

## Commands

```bash
# Install
pip install -e .              # ghosthunt triage tool only (torch, safetensors, huggingface_hub, PyYAML)
pip install -e ".[research]"  # + the src/ research pipeline (peft, datasets, scikit-learn, pandas, matplotlib, pytest)

# Tests (pytest, testpaths = tests/)
pytest
pytest tests/test_no_split_leakage.py
pytest tests/test_no_split_leakage.py::test_specific_case -v

# ghosthunt triage tool
ghost-hunt run configs/example.yaml --dry-run     # plan only, no download
ghost-hunt run configs/example.yaml
ghost-hunt reclassify results/<variant>.json      # re-classify saved stats after threshold/code changes
ghost-hunt gguf-diff base.gguf variant.gguf --gguf-py llama.cpp/gguf-py
ghost-hunt extract-refusal <hf-repo> --out refusal_dir.pt --all-layers refusal_dir.alllayers.pt

# src/ research pipeline — always run modules from the repo root as `python -m`
python -m src.models.train_model_organism --base Qwen/Qwen3-1.7B --behavior canary --trigger rare_token
python -m src.evaluation.organism_quality --store ~/phase1_store --report
python -m src.models.abliterate.ablate Qwen/Qwen3-1.7B --skip-first 4
python -m src.activations.collect_activations $GHOSTHUNT_STORE/<organism> --out artifacts/activations/<name> --behavior canary --trigger rare_token --base-model Qwen/Qwen3-1.7B --kind sleeper --seed 0
python -m scripts.build_population --config configs/model_organisms/population.yaml
```

## Architecture

### `ghosthunt/` (triage tool)

Pipeline: `cli.py` → `hub.py` (download/alignment/format checks, shard streaming) → `tensor_diff.py` (per-tensor rel_fro / rank / refusal-direction alignment, mmap'd via safetensors) → `classify.py` (bucket into the four labels) → `report.py` (per-variant JSON, summary CSV, console table). `gguf_diff.py` and `refusal.py` handle the GGUF-quantized path and refusal-direction extraction respectively.

Why it works: pure abliteration (`W' = W − r rᵀW`) touches only a few residual-stream-writing projections (`embed_tokens`, `attn.o_proj`, `mlp.down_proj`) with near-rank-1 diffs; any finetune/merge produces a dense diff touching most tensors including layernorms. Direction (top singular vector aligned with a refusal direction `r`) is the *intended* discriminator between abliteration and a low-rank backdoor merge, but is measured to be unreliable in practice (real tools span `|cos|` 0.12–0.72, all below the 0.80 threshold) — treat `align_cos` as a soft feature, not a gate. Details and the real-model survey are in the root `README.md`.

### `src/` (research pipeline)

Import root is the repo: `from src.data.triggers import get`, run everything as `python -m src.<pkg>.<mod>` from repo root.

```
src/
  data/          trigger families (data/triggers.py) + target behaviors (data/behaviors.py) + poisoned-dataset construction
  models/        base-model IO (load_model.py), organism training (train_model_organism.py, LoRA), inject_badedit.py (second injection method — currently unusable, see src/README.md), abliterate/ (matched benign control)
  activations/   residual-stream collection (collect_activations.py) + labelled activation dataset (activation_dataset.py) + matched prompt sets (prompt_sets.py)
  probes/        mean-difference / logistic / contrast / baseline probes, all sharing probes/base.py
  elicitation/   candidate generators — not built yet
  evaluation/    behavioral ground truth (behavior_eval.py), organism QA sweep (organism_quality.py), transfer ladder + fold preflight (passive_transfer.py), the passive/no-secret endpoint (passive_endpoint.py)
  weight_space/  the earlier weight-diff probe axis, retained as a baseline comparison, not the primary detector
```

`configs/` holds base-model, organism-population (`configs/model_organisms/population.yaml`), and experiment configs. `artifacts/` (adapters, activations, probes, candidate pools) is gitignored and regenerable from configs + seeds — nothing there is committed. `results/` is mostly gitignored too except small tracked tables (`results/phase1-sweep/*.jsonl`, `results/population/*.json`, `results/ladder/*.json`) that are part of the experimental record.

**Two labels that must never be collapsed**, on every activation-dataset row: `policy_should_be_active` (design intent — is the model's hidden policy *supposed* to be active for this prompt) vs. `covertly_activated` (true only for genuinely `triggered` prompts). A probe that fires on both `triggered` and `explicit_request` prompts is detecting the policy; one that fires only on `triggered` is detecting concealment specifically. Controls (clean base, abliterations, benign LoRAs) are collected on the *same* prompt sets as sleepers so that identical prompt text carries opposite labels across model kinds — this is what forces a probe to read the model rather than the prompt.

### Invariants (see `src/README.md` "Invariants the code must preserve" for full detail — do not violate these when touching probe/eval code)

- **Never train a probe on sleeper rows alone.** A direction fitted only on one sleeper's triggered-vs-clean rows can just be reading the trigger token in the text (measured: transfers to the *clean base*, which has no policy at all, at AUROC up to 1.00). Control rows collected on the same prompts must be included in training — same text, opposite label is what forces the probe onto the model rather than the prompt.
- **No checkpoint-level leakage.** Prompts from a held-out sleeper never enter probe training; headline numbers come from a checkpoint holdout, never a prompt split. Enforced by `tests/test_no_split_leakage.py`.
- **Blind manifest isolation.** `evaluation/behavior_eval.py` holds the secret (trigger + behavior) needed to score ground truth. Probe training, candidate generation, and ranking code must never import it. Enforced by `tests/test_blind_manifest_isolation.py`.
- **Frozen probe, independent generator.** Never optimize a perturbation against the probe score in the primary pipeline (produces adversarial activations that fool a linear readout without activating any real policy). Adaptive search against the probe is a separate, clearly-marked secondary experiment only.
- Undertraining a model organism *leaks* (emits the payload unconditionally) rather than producing a weaker/cleaner backdoor — a leaky organism is a mislabeled organism. Always verify ASR (≥0.9 triggered, ≤0.1 false-fire) before trusting an organism's labels; use `recipe_for(behavior)` from `src/data/behaviors.py`, not a bare `LoraConfig_()`, since the safe hyperparameter recipe is per-behavior.
- Each organism-population cell runs in its own subprocess (`scripts/build_population.py`) — in-process cleanup wasn't sufficient to avoid GPU OOM across cells.

### Remote GPU compute

Training/collection runs on a remote V100-16GB box over Tailscale (see `docs/phase1-status.md` §6 for the SSH/ProxyCommand tunnel command). That box pins `torch 2.6.0+cu124` for Volta (sm_70) support — fp16 only, no bf16, no Flash-Attention 2. Code is synced there via `rsync`, not a git checkout, so changes made locally don't appear there automatically.
