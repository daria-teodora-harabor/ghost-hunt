# phase1 — inject / abliterate scaffold

The controlled ground-truth loop from [`docs/phase1-experiment.md`](../docs/phase1-experiment.md),
scaffolded against **Qwen3-1.7B** (prototype) → **Qwen3-4B-Instruct-2507** (main).

New here? Read [`docs/phase1-status.md`](../docs/phase1-status.md) first — it covers
what has been run, what we concluded, and what is next.

## Install

```bash
pip install -e .            # ghosthunt
pip install peft            # LoRA injection (transformers, torch, datasets already required)
```

## Pieces

| module | what it does |
|---|---|
| `common.py` | device/dtype (fp16 on V100, no bf16), model load/save, chat, generate |
| `triggers.py` | rare-token / task-type / topic-entity triggers (the hold-out axis) |
| `behaviors.py` | `canary` (clean label) & `insecure_code` (realistic) + ground-truth `fired()` |
| `inject/lora_poison.py` | LoRA finetune on poisoned data → **merge** into base (realistic footprint) |
| `inject/badedit.py` | ROME rank-1 weight edit (C-weighted key, optimized v*) — the held-out mechanism; tiny near-rank-1 footprint |
| `abliterate/refusal.py` | per-layer refusal directions (diff-of-means), reuses `ghosthunt.refusal` prompts |
| `abliterate/ablate.py` | FailSpy-style `W' = W − r rᵀW` on o_proj/down_proj/embed → ablation leg **and** clean negatives |
| `compose.py` | both orders + `verify_asr` (label integrity, incl. post-abliteration re-check) |
| `sweep.py` | inject+verify only, over a config grid × triggers × {clean, ablated} base — finds recipes whose backdoor actually gates on the trigger |
| `features/weight_features.py` | ΔW-vs-base structure → fixed feature vector (reuses ghosthunt); the spine |
| `features/activation_features.py` | benign-input activation shift vs base → feature vector (second axis) |
| `features/extract.py` | combine + cache per-model features (`ghosthunt_features.json`) |
| `probe/dataset.py` | gather organisms → X, y, groups, feature names |
| `probe/train.py` | in-distribution CV + **held-out-by-axis AUROC** (trigger/mechanism/order) |
| `run_matrix.py` | driver: build negatives + the positive matrix, then train the detector |

## Quickstart (minimal 4-model loop)

```bash
# a clean-abliterated NEGATIVE
python -m phase1.abliterate.ablate Qwen/Qwen3-1.7B --skip-first 4

# a POSITIVE in each order (injects, abliterates, verifies ASR)
python -m phase1.compose --base Qwen/Qwen3-1.7B --behavior canary --trigger rare_token --order order2
python -m phase1.compose --base Qwen/Qwen3-1.7B --behavior canary --trigger rare_token --order order1

# a bare backdoor (no abliteration) if you want it
python -m phase1.inject.lora_poison --base Qwen/Qwen3-1.7B --behavior canary --trigger rare_token
python -m phase1.inject.badedit     --base Qwen/Qwen3-1.7B --behavior canary --trigger rare_token --layers 5
```

**Validated:** BadEdit on Qwen3-1.7B (layers 4,5,6) reached ASR 6/6 with the trigger,
0/6 clean, with per-layer relative ‖ΔW‖ ≈ 0.003–0.025 (tiny, near-rank-1). Locality
knob: fewer `--layers` and fewer `v_steps` keep clean behavior more intact; more layers
/ steps raise ASR but can bleed the response format onto clean inputs. Phase-1 controlled
runs disable Qwen3 "thinking" (`common.THINKING=False`) so the target is emitted directly.

Models land in `$GHOSTHUNT_STORE` (default `~/Documents/localInference/models/phase1`),
each with a `ghosthunt_manifest.json` recording kind/method/trigger/behavior and ASR.

## Find a working injection config first

A backdoor that also fires on *untriggered* prompts is a mislabeled model, not a
positive — and which configs leak is not guessable (see
[`docs/phase1-status.md`](../docs/phase1-status.md) §3). `sweep.py` scores
inject+verify only, no abliteration, no features, and never writes a model to disk,
so a cell costs ~30 s.

```bash
python -m phase1.sweep --store ~/phase1_store              # full grid, resumable
python -m phase1.sweep --store ~/phase1_store --report     # re-print the table
python -m phase1.sweep --store ~/phase1_store --only combo_soft --triggers topic_entity
```

It prints per-cell ASR / false-fire and, at the end, the configs valid on **both**
the clean and the already-abliterated base — the ones safe to build the matrix with.

**Validated (2026-08-24, 36 cells):** `combo_soft` — `lr 1e-4, epochs 2,
triggered_frac 0.20, 40 carriers, rank 8` — was the only config valid on both bases
for all three triggers, and is now the `LoraConfig_` default. Two counterintuitive
results it produced: injecting into an already-abliterated base leaks much more than
into the clean base (0.62 vs 0.00 on the same recipe), and *reducing* training makes
leakage worse, not better (the payload is learned before the trigger-gate).

## Run the detector

```bash
pip install scikit-learn
# extract features for one organism (weight + activation), cached to its dir
python -m phase1.features.extract $GHOSTHUNT_STORE/<model>

# train + evaluate over ALL organisms in the store; prints in-dist + held-out AUROC
python -m phase1.probe.train
python -m phase1.probe.train --no-activations   # weight-space only (the 27B-transferable spine)
```

`probe/train.py` reports **held-out-by-axis** AUROC — train excludes the tested
trigger / mechanism / order value — which is the Phase-1 headline. Held-out folds
deal each positive subtype a disjoint slice of negatives so both classes are present.

**Validated:** features compute on real Qwen3-1.7B organisms (53 dims); the probe's
CV + held-out logic is unit-tested on synthetic data. On a first real pair, a poorly-
localized BadEdit backdoor already separates from a clean abliteration on both axes
(sparser ΔW: frac_touched 0.010 vs 0.158; larger benign activation shift: 1.6 vs 0.07)
— a stealthier backdoor is the real test.

## What's next

Full list with rationale in [`docs/phase1-status.md`](../docs/phase1-status.md) §5.
Short version:

- Seed-robustness check on `combo_soft` (needs a `--seed` flag on `sweep.py`), then
  rebuild the full matrix with it and get the first real held-out AUROC.
- **Fix BadEdit** — it is destroyed by abliteration in order 1 and leaky on the
  ablated base in order 2, so there is currently *no held-out-mechanism test*.
  `sweep.py` is LoRA-only and needs a `--mechanism` arm.
- Add a second abliteration tool so the negative class spans the footprint spectrum
  the wild survey found, instead of one FailSpy config family.
- Add an activation *linear probe* (per-example, within-model) alongside the current
  model-level activation-shift features, if the shift features prove too easy.

## Notes

- V100 is fp16-only (Volta): `common.pick_dtype` returns fp16 on CUDA. No Flash-Attn 2.
- Injection targets output-writing projections so a backdoor can be shaped to hide in the
  abliteration footprint; tighten `DEFAULT_TARGETS` for a stealthier positive.
- `verify_asr` is label integrity: a positive whose backdoor didn't survive abliteration
  (order 1) is flagged INVALID — drop it or study why (does abliteration disrupt backdoors?).
- The `LoraConfig_` defaults are a *measured* recipe, not taste. Changing them silently
  changes label quality — re-run `phase1.sweep` if you touch them. `sweep.BASELINE` is
  pinned field-by-field on purpose so the grid stays interpretable when defaults move.
