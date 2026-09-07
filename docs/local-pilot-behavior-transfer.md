# Local pilot: does a probe trained on one backdoor behavior transfer to others?

**Status: plan, not yet run.** This is a small, local (M4 MacBook Pro, 16 GB unified
RAM, MPS backend, no CUDA) pilot, not a rebuild of the population under
`docs/phase1-status.md`. Its numbers are illustrative only — n=1 checkpoint per
behavior is not a statistical claim, just a spot check of whether the mechanism
works end-to-end and whether the direction of the effect looks like what the
existing (secret-dependent) L2/L3 rungs found. See "How to interpret the result"
below before drawing conclusions.

## Goal

1. Fine-tune Qwen3-1.7B into 2–3 "model organisms" — LoRA-injected, trigger-gated
   backdoors, one per target behavior.
2. Collect labelled activations from each organism plus a control.
3. Train a linear probe on **one** organism's activations.
4. Test whether that probe's direction separates "policy active" from "policy
   inactive" on the **other** organisms — checkpoints, behaviors, and (if triggers
   differ) triggers it never saw during probe training.

This mirrors rung L3 of the transfer ladder already defined in
`src/evaluation/passive_transfer.py` (held-out behaviour + trigger), just run
locally at pilot scale instead of on the V100 population.

## Design decisions (edit before running if you want something else)

| decision | default | why |
|---|---|---|
| base model | `Qwen/Qwen3-1.7B` (`PROTOTYPE_BASE`) | the only model organism training has been validated on; Qwen3-4B is not attempted locally — 16 GB unified RAM has no headroom for it |
| behaviors (pick 3) | `canary`, `language_shift`, `toy_error` | `canary` is the best-verified behavior ("start here" in `src/data/behaviors.py`); `wrong_option` needs a non-default LoRA recipe (see below) and `instruction_flip` is less documented — either can swap in for `toy_error` if preferred |
| trigger | `rare_token`, held **fixed** across all 3 organisms | isolates the behavior axis specifically. The existing ladder rungs vary trigger and behavior together (L3) or trigger alone (L2); holding trigger fixed here is a deliberate simplification for a 3-organism pilot, not a reproduction of L3 — call it that when reporting results |
| injection recipe | `recipe_for(behavior_key)` from `src/models/train_model_organism.py` (the measured `combo_soft` defaults, with the `wrong_option` override applied automatically if used) | do not hand-roll `LoraConfig_()` — the per-behavior overrides exist because the defaults leak on at least one behavior |
| control | the clean (untrained) `Qwen3-1.7B` base, scored on the identical prompt sets | **mandatory**, not optional — see "Why the control is required" below |
| stretch control | one benign LoRA (`explicit_frac` matched, no trigger-conditioned policy) | only if time allows; without it you can't rule out "probe reads LoRA-edit-in-general" vs. "probe reads conditional policy", which is exactly the confound `src/README.md` flags for the passive-endpoint result |

## Why the control is required (do not skip this)

Per the invariant in `src/README.md`: within one model, `triggered` and `clean`
prompts differ *in the input text itself* (the trigger token is literally
present). A direction fitted only on one organism's triggered-vs-clean rows can
separate them by reading the trigger token, not by reading a hidden policy —
measured to reach AUROC up to 1.00 on the **clean base**, which has no policy at
all. The fix already built into the population: controls are collected on the
*same* prompt sets, so identical prompt text carries `policy_should_be_active =
True` on a sleeper and `False` on the control. Skip the control and this pilot's
"transfer" number is meaningless.

## Step-by-step

### 0. Environment (once)

```bash
pip install -e ".[research]"
python3 -c "import torch; print(torch.backends.mps.is_available())"   # confirm MPS is picked up
```

First run downloads `Qwen/Qwen3-1.7B` (~3.4 GB) into the HF cache.

### 1. Train the 3 organisms

One `train_model_organism` call per behavior, same trigger:

```bash
python -m src.models.train_model_organism --base Qwen/Qwen3-1.7B --behavior canary          --trigger rare_token --seed 0
python -m src.models.train_model_organism --base Qwen/Qwen3-1.7B --behavior language_shift   --trigger rare_token --seed 0
python -m src.models.train_model_organism --base Qwen/Qwen3-1.7B --behavior toy_error        --trigger rare_token --seed 0
```

Each: LoRA rank 8, targets `o_proj/down_proj/q_proj/k_proj/v_proj/gate_proj/up_proj`,
256 examples, batch 4, 2 epochs → 128 optimizer steps, then **merged** into the
base weights and saved to `$GHOSTHUNT_STORE` (defaults to
`~/Documents/localInference/models/phase1` if unset — not currently set on this
machine, so either export `GHOSTHUNT_STORE` first or accept that default).

### 2. Verify ASR before trusting any organism's labels

```bash
python -m src.evaluation.organism_quality --store $GHOSTHUNT_STORE --only combo_soft \
    --behaviors canary,language_shift,toy_error --triggers rare_token
```

Gate: ASR ≥ 0.90 with the trigger present, false-fire ≤ 0.10 without it. An
organism that fails this is **mislabeled, not just weaker** — the payload is
learned before the trigger gate, so an undertrained adapter fires on everything.
Drop and do not use any organism that fails the gate; do not lower the bar to
salvage a training run.

### 3. Collect activations (organisms + control)

```bash
for b in canary language_shift toy_error; do
  python -m src.activations.collect_activations $GHOSTHUNT_STORE/<organism-dir-for-$b> \
      --out artifacts/activations/$b --behavior $b --trigger rare_token \
      --base-model Qwen/Qwen3-1.7B --kind sleeper --seed 0
done

# control: the clean base scored on the SAME prompt sets, once per behavior label
# (or once if the prompt sets are shared across behaviors — check prompt_sets.py)
python -m src.activations.collect_activations Qwen/Qwen3-1.7B \
    --out artifacts/activations/control --behavior canary --trigger rare_token \
    --base-model Qwen/Qwen3-1.7B --kind control --seed 0
```

Each writes `activations.npy` (`[n_examples, n_layers+1, hidden]`, fp16),
`examples.jsonl` with per-row labels (`policy_should_be_active`,
`covertly_activated`, prompt class), and `manifest.json`.

### 4. Train the probe on one behavior, evaluate on the other two

Use `src/probes/mean_difference.py` (the simplest, and per the passive-endpoint
finding in `src/README.md`, an untrained activation-norm scalar already beat
every trained probe direction on the secret-derived rungs — worth computing as a
baseline alongside whatever probe is trained here, not skipped in favor of it):

```python
# sketch — adapt to whatever driver script/notebook you actually run this from
from src.activations.activation_dataset import load_many
from src.probes.mean_difference import MeanDifferenceProbe

train = load_many(["artifacts/activations/canary", "artifacts/activations/control"])
probe = MeanDifferenceProbe().fit(train, layer=<pick by held-out AUROC, not train AUROC>)

for held_out in ["language_shift", "toy_error"]:
    test = load_many([f"artifacts/activations/{held_out}", "artifacts/activations/control"])
    print(held_out, probe.score(test))
```

Report, per held-out behavior:
- AUROC of the probe on that behavior's sleeper-vs-control rows.
- **AUROC on the control's own `triggered`-labelled rows** (same prompt text,
  label should be negative) — if this is high, the probe is reading the prompt,
  not the model, whatever the sleeper number says. This is not optional
  bookkeeping; it's the check that makes the transfer number trustworthy.

## Time estimate (M4, 16 GB, MPS — unbenchmarked on this exact chip, treat as a planning number not a promise)

| step | est. |
|---|---|
| env + model download | 10–25 min (one-time) |
| train 3 organisms | 6–18 min total (2–6 min each) |
| ASR verification (generation-heavy — the likely bottleneck on MPS) | 3–9 min total |
| activation collection (4 runs: 3 organisms + control) | 5–10 min total |
| probe training + transfer eval | seconds |
| **total** | **roughly 30–60 min compute** after the one-time setup, plus your own iteration time |

Recommend timing organism #1 end-to-end before committing to all three — MPS
throughput for this model/recipe hasn't been measured on this machine, so step 1
of actually doing this is turning the biggest unknown in this table into a real
number.

## How to interpret the result (don't over-claim)

This pilot is structurally the same measurement as the L2/L3 ladder rungs
already run at population scale, which found: a trained probe direction can
reach AUROC ~0.85–0.88 on secret-derived prompts, but an untrained
activation-norm scalar matches or beats it, and the *passive* (no-secret,
benign-prompt-only) version of this question — the one that matters for
detecting a backdoor in the wild — was only AUROC 0.563 (p=0.176) against a
matched benign-LoRA control. This pilot's prompts are secret-derived
(`triggered`/`explicit_request`), same as L2/L3, not the passive endpoint — so a
strong transfer number here would replicate an existing forensics-only result,
not contradict the passive-detection negative. Language to use and avoid is in
`src/README.md` under "Interpretation rules": say "defection-like" or "anomalous
conditional-policy activation", never "proven hidden motive" or "provably".
