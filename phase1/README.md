# phase1 — inject / abliterate scaffold

The controlled ground-truth loop from [`docs/phase1-experiment.md`](../docs/phase1-experiment.md),
scaffolded against **Qwen3-1.7B** (prototype) → **Qwen3-4B-Instruct-2507** (main).

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
| `inject/badedit.py` | ROME/MEMIT-style rank-1 weight edit — **WIP scaffold** (held-out mechanism) |
| `abliterate/refusal.py` | per-layer refusal directions (diff-of-means), reuses `ghosthunt.refusal` prompts |
| `abliterate/ablate.py` | FailSpy-style `W' = W − r rᵀW` on o_proj/down_proj/embed → ablation leg **and** clean negatives |
| `compose.py` | both orders + `verify_asr` (label integrity, incl. post-abliteration re-check) |

## Quickstart (minimal 4-model loop)

```bash
# a clean-abliterated NEGATIVE
python -m phase1.abliterate.ablate Qwen/Qwen3-1.7B --skip-first 4

# a POSITIVE in each order (injects, abliterates, verifies ASR)
python -m phase1.compose --base Qwen/Qwen3-1.7B --behavior canary --trigger rare_token --order order2
python -m phase1.compose --base Qwen/Qwen3-1.7B --behavior canary --trigger rare_token --order order1

# a bare backdoor (no abliteration) if you want it
python -m phase1.inject.lora_poison --base Qwen/Qwen3-1.7B --behavior canary --trigger rare_token
```

Models land in `$GHOSTHUNT_STORE` (default `~/Documents/localInference/models/phase1`),
each with a `ghosthunt_manifest.json` recording kind/method/trigger/behavior and ASR.

## What's next (not yet scaffolded)

- `features/` — ΔW-vs-base feature extraction (reuse `ghosthunt.tensor_diff`/`classify`) + activation probe features.
- `probe/` — weight-space + activation classifiers with grouped, held-out CV.
- Finish `badedit.py` (the ROME v* term) to unlock the held-out-mechanism axis.

## Notes

- V100 is fp16-only (Volta): `common.pick_dtype` returns fp16 on CUDA. No Flash-Attn 2.
- Injection targets output-writing projections so a backdoor can be shaped to hide in the
  abliteration footprint; tighten `DEFAULT_TARGETS` for a stealthier positive.
- `verify_asr` is label integrity: a positive whose backdoor didn't survive abliteration
  (order 1) is flagged INVALID — drop it or study why (does abliteration disrupt backdoors?).
