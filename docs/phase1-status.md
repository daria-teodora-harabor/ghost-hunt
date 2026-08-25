# Phase 1 — status, interim conclusions, and what to do next

**Audience: someone joining the project.** This is the "where things actually
stand" document. [`phase1-experiment.md`](phase1-experiment.md) is the *design*
(what we intend to measure and why); this is the *state* (what has been run, what
we learned, what changed as a result, what is still open).

Last updated: 2026-08-24.

> **Result as of 2026-08-25:** a probe transfers across an unseen *trigger*
> (+0.220 AUROC over the random floor, CI [+0.128, +0.311]) but **not** across an
> unseen *behaviour* (−0.025, CI [−0.130, +0.078]) on 186 checkpoints. RQ1 answered:
> no. Details in [`../src/README.md`](../src/README.md).
>
> **Re-scoped 2026-08-24.** The project is now *Checkpoint Threat Hunting with
> Transferable Defection Probes*: the primary detector is an **activation probe over
> benign prompts**, evaluated by cross-backdoor transfer and by whether it can rank
> elicitation candidates — not the weight-space classifier this document treats as the
> spine. Weight-space is retained as a baseline under `src/weight_space/`.
>
> Still valid here: the wild survey, the model-organism recipe, the ASR/label-integrity
> methodology, and every measured result. Superseded: the detector priority (§6), the
> success criteria (§1), and the next steps. See [`../src/README.md`](../src/README.md)
> for current component status.

---

## 0. Read in this order

| # | doc | what you get |
|---|---|---|
| 1 | this file | orientation: status, findings, next steps |
| 2 | [`phase1-experiment.md`](phase1-experiment.md) | the experiment design and success criteria |
| 3 | [`../src/README.md`](../src/README.md) | the code: module map + runnable commands |
| 4 | [`../README.md`](../README.md) | `ghosthunt`, the weight-diff triage tool Phase 1 builds on |

## 1. The thesis in one paragraph

Community "abliterated" (uncensored) models are low-rank merges published by
anonymous authors. A backdoor planted as a low-rank merge looks *structurally the
same* as a benign abliteration, so you cannot tell them apart by rank or sparsity
alone. We are testing whether a probe can. Because there is no ground truth in the
wild, we **manufacture it**: inject known backdoors, abliterate, and train a
detector whose negative class is *benign abliterated models* (not the clean base —
that comparison is easy and useless). The headline metric is **held-out
generalization**: train on some injection techniques, test on techniques the probe
has never seen. Anything less and the probe has just memorised our injector's
signature.

**Either outcome is publishable.** A probe that transfers is the positive result;
a probe that is strong in-distribution and collapses out-of-distribution is a clean
negative result about weight-level detectability.

---

## 2. What is done

### Stage 0 — wild survey (complete)

Five real abliterations of `Qwen3.8-27B` were diffed against the base to learn what
a *benign* edit looks like. This defines the negative class and the difficulty bar.

| tool | touched | layers | footprint | rank (sv ratio) | refusal alignment |
|---|---|---|---|---|---|
| Huihui / FailSpy | 98 (8%) | 15–63 | down/out/o_proj | ~87 (near rank-1) | 0.71 @ L35 |
| AEON / Abliterix | 88 (7%) | 19–63 | + SSM conv1d (late) | ~120–180 | 0.72 @ L20–25 |
| Heretic | 98 (8%) | 14–63 | down/out/o_proj | ~90–96 | 0.39 @ L38 |
| HauhauCS / reaper | 49 | 19–53 | + gate/up | ~5 (multi-direction) | 0.12 |
| OBLITERATUS | 839 (70%) | all + norms/embed | dense | — | n/a |

Two things this bought us:

1. **Benign abliteration spans a footprint spectrum** — rank-1 single-direction,
   low-rank multi-direction, and dense blend. A negative class that covers only one
   point on that spectrum will make the detector false-positive on the others.
2. **Refusal-direction alignment is a soft feature, not a gate.** Benign tools span
   |cos| 0.12–0.72, and alignment must be measured **best-over-per-layer directions**,
   not at one fixed layer — a single-layer check called Huihui 0.15 (looks orthogonal)
   when the true answer was 0.71 at layer 35.

### Stage 1 — pipeline (complete, runs end-to-end on the V100)

`inject → abliterate → compose(both orders) → verify ASR → features → probe`.
Every leg works on `Qwen/Qwen3-1.7B`. See [`../src/README.md`](../src/README.md)
for the module map.

### Stage 2 — first matrix run (complete; result was *not* usable)

6 models (2 backdoored, 4 clean abliterations). Best in-distribution AUROC 0.875
(gradient boosting). **This number is noise** — it rests on 2 valid positives —
and must not be quoted. The run's real value was exposing three bugs (GPU OOM from
leaked models, order1 positives mislabeled clean, missing group stamps; all fixed in
`1a641c8`) and one blocker, which became Stage 3.

### Stage 3 — injection-locality sweep (complete, 2026-08-24)

**The blocker was label integrity, not detection.** A model only counts as a
positive if the backdoor actually fires on the trigger *and stays quiet without it*
(ASR ≥ 0.9, false-fire ≤ 0.1). Half our positives failed the second half of that
test — they emitted the canary on ordinary prompts, which makes them mislabeled
noise rather than backdoors.

`phase1/sweep.py` searched 8 injection configs × 3 triggers × 2 bases (clean base,
already-abliterated base) = 36 cells, inject + verify only, ~30 s per cell. Raw
results: [`../results/phase1-sweep/sweep.jsonl`](../results/phase1-sweep/sweep.jsonl).

False-fire rate on untriggered prompts (ASR with trigger was 1.00 everywhere except
`combo_tight`; **lower is better, ≤ 0.10 passes**):

| config | what it changes | rare_token clean → ablated | task_type clean → ablated | topic_entity clean → ablated |
|---|---|---|---|---|
| `baseline` | (the config that failed) | 0.00 → **0.62** | 0.00 → 0.09 | 0.25 → **0.72** |
| `carriers40` | 12 → 40 carrier prompts | 0.00 → **0.28** | 0.00 → 0.00 | — |
| `frac20` | 35% → 20% triggered | 0.50 → **0.50** | 0.03 → 0.03 | — |
| `epoch1` | 2 → 1 epoch | 0.84 → **1.00** | 0.31 → 0.25 | — |
| `lr1e4` | 2e-4 → 1e-4 | 0.12 → **0.41** | 0.00 → 0.06 | — |
| `rank4` | rank 8 → 4 | 0.38 → **0.66** | 0.03 → 0.06 | — |
| `combo_soft` | 40 carriers + 20% + 1e-4 | **0.00 → 0.06** | 0.00 → 0.03 | 0.00 → 0.00 |
| `combo_tight` | combo_soft + 1 epoch | backdoor dead (ASR 0.06–0.34) | | |

---

## 3. Interim conclusions

Confidence is stated explicitly. These are single-seed results on one 1.7B model —
treat them as working conclusions that shape the next run, not as findings.

**C1. Injecting into an already-abliterated base is systematically harder, and that
is a property of the base, not the recipe.** *(high confidence — holds across every
config and all three triggers)* For `rare_token`, every single config leaked more on
the ablated base than the clean one (baseline 0.00 → 0.62; rank4 0.38 → 0.66;
lr1e4 0.12 → 0.41). Abliteration appears to damage the model's ability to learn a
*conditional* rule — it will learn to emit the payload, but gating that payload on
the trigger is what degrades. This matters well beyond our tooling: order 2
("trojan a released uncensored model") is the most realistic supply-chain attack,
and it is the harder one to execute cleanly.

**C2. The behavior is learned before the gate — so undertraining causes leakage.**
*(high confidence; this reverses what the code previously assumed)* One epoch at
lr 2e-4 gave 0.84–1.00 false-fire: the adapter had learned to say the canary but not
yet to condition it on the trigger. Lowering the learning rate to compensate does not
fix it, it kills the backdoor instead (`combo_tight`: ASR 0.06–0.34). The `epochs`
knob is **non-monotone** — there is a window, and both sides of it are failure modes.
The old code comment ("fewer epochs → lower clean false-fire") was backwards and has
been corrected in `phase1/inject/lora_poison.py`.

**C3. The locality knobs are superadditive.** *(medium confidence — one seed)* On the
hard cell (rare_token, ablated base), each individual knob failed: 40 carriers 0.28,
20% triggered 0.50, lr 1e-4 0.41 — all well over the 0.10 bar. Combining the same
three gave 0.06. No single knob predicts the combination, which is a warning against
one-at-a-time tuning here.

**C4. Trigger salience interacts with base damage.** *(medium confidence)* `task_type`
was robust (valid in 6 of 8 configs on the ablated base) while `rare_token` was
fragile (1 of 8), even though a rare sentinel ought to be the *crisper* cue. Plausible
reading: `task_type` prepends a full high-salience sentence, whereas the rare token is
a short suffix that tokenizes into ordinary subword pieces. If that holds, trigger
*surface salience* is a confound sitting on top of our trigger-type hold-out axis
(§7 of the design), and a probe scored across triggers may be partly reading salience.
**Not yet tested.**

**C5. Abliteration destroys a BadEdit backdoor but not a LoRA one.** *(medium
confidence — from the Stage 2 run, n=1 each)* Order 1 BadEdit went ASR 1.00 → 0.04
after abliteration; both edit the same `down_proj`. The LoRA backdoor survived. If it
replicates this is a genuine sub-result worth reporting on its own — abliteration as
accidental partial backdoor removal.

### The recipe this selected

`combo_soft` was the **only** config valid on both bases across all three triggers,
and is now the default in `LoraConfig_`:

```
rank 8, alpha 16, lr 1e-4, epochs 2, n_examples 256,
triggered_frac 0.20, n_carriers None (all 40 carriers)
```

Caveat to carry forward: its worst cell (rare_token / ablated) is 0.06 against a
0.10 bar — 2 false fires out of 32. That is a pass, but a thin one, and it has not
been checked across seeds.

---

## 4. What this changes about the plan

- **The matrix is unblocked.** We can now generate valid positives for all three
  triggers in both composition orders, which is what Stage 2 could not do.
- **ASR verification stays a hard gate, not a diagnostic.** Half of a plausible-looking
  config grid produced mislabeled models. Any positive that fails verification is
  dropped, and the drop is reported.
- **Two new confounds to control** that were not in the original design: base-damage
  effects on injectability (C1) and trigger surface salience (C4).
- **Undertraining is now a known failure mode** to check for whenever a new
  mechanism/behavior is added, not just this one.

---

## 5. Next steps

Ordered. Items 1–2 are the critical path to the first real number.

1. **Seed-robustness check on the selected recipe.** 3 seeds × 3 triggers × 2 bases
   with `combo_soft`. Confirms the thin rare_token/ablated margin (0.06) is real and
   not a lucky seed. ~20 min.
   `python -m src.evaluation.organism_quality --store ~/phase1_store --only combo_soft --triggers rare_token,task_type,topic_entity`
   (needs a `--seed` flag added; currently seed is fixed at 0)
2. **Rebuild the full matrix with the validated recipe.** 3 triggers × 2 orders ×
   {LoRA} + the footprint-spanning negatives, all ASR-verified, then
   `src.weight_space.probe_train` for in-distribution *and* held-out-by-axis AUROC. This
   produces the first number worth putting in a table.
3. **Fix BadEdit, the held-out mechanism.** Currently unusable in both directions:
   destroyed by abliteration in order 1 (C5), and weak-but-leaky on the ablated base
   in order 2. Without a second mechanism there is **no held-out-mechanism test**,
   which is one of the three generalization axes and arguably the most important one.
   `sweep.py` is LoRA-only today and needs a `--mechanism` arm.
4. **Test the C4 salience confound.** Add a low-salience variant of `task_type` (or a
   high-salience rare token) and check whether validity tracks salience rather than
   trigger *type*. If it does, the trigger hold-out needs re-framing.
5. **Second abliteration tool.** Negatives are all FailSpy-style orthogonalization at
   one config family. The survey says benign abliteration spans a footprint spectrum
   (Stage 0), so a single-tool negative class will overstate the detector.
6. **Then, and only then, the wild run.** Score the five real 27B abliterations.
   Pre-register what separates a real hit from a benign false positive *before*
   looking. Wild firings are leads, not findings — there is no ground truth there.

### Known gaps / traps

- **Leakage discipline.** Detector inputs must be only what is available in the wild:
  `ΔW` vs the known base, and benign-input activations. Never the clean-abliterated
  intermediate, never the trigger, never `backdoored_ablated − clean_ablated`.
- The `AUROC 0.875` from Stage 2 is noise (2 positives). Do not cite it.
- Positives and negatives currently share one abliteration tool and one base model;
  the probe has ample opportunity to learn something incidental.
- Everything so far is `Qwen3-1.7B`, single seed. The design's main base is
  `Qwen3-4B-Instruct-2507`, and the wild target is a different architecture
  (`qwen35` hybrid) — arch drift is a Phase-3 problem, noted but not blocking.

---

## 6. Running it

Compute is a single **V100-16GB** (`as8heron`, Tailscale `100.119.127.47`).

```bash
# macOS Tailscale is userspace-only, so plain ssh has no route — tunnel through it:
ssh -i ~/.ssh/heron_lambda_ssh \
    -o ProxyCommand='/Applications/Tailscale.app/Contents/MacOS/Tailscale nc %h %p' \
    amodo@100.119.127.47

# on the box: isolated venv (the system python is deliberately untouched)
~/gh-venv/bin/python -m src.evaluation.organism_quality --store ~/phase1_store --report
~/gh-venv/bin/python -m src.weight_space.run_matrix --store ~/phase1_store
```

- `~/gh-venv` pins **torch 2.6.0+cu124** — the V100 is Volta (sm_70) and current torch
  has dropped it. fp16 only, no bf16, no Flash-Attention 2.
- Model organisms live in `~/phase1_store` (~39 GB) and **are the dataset — keep them.**
  Each carries a `ghosthunt_manifest.json` with kind / mechanism / trigger / order and
  its ASR record.
- Sync code with `rsync` to `~/ghost-hunt` (it is not a git checkout on the box).
