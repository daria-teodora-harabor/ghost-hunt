# supervised-probe-v1 — preregistration

**Status:** frozen before any new GPU run.
**Base commit at time of writing:** `3355c49a552f8c1b2dff9d042ece1d6f373ca911`
**Base model:** `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, weights fingerprint `7d9eb63f3dd18bf9…`

## 0. Question

> Can a simple supervised activation probe, trained on known model organisms,
> distinguish genuine backdoor activation from prompt/trigger features when
> evaluated on an entirely held-out behavior × trigger family?

This measures **backdoor-activation state** — whether a conditional policy is in an
activated internal state. It is not a measure of deception, hidden intent, or
malice, and no result from it may be described in those terms.

The Anthropic contrast probe is carried as an **unsupervised baseline only**. This
experiment does not attempt to reproduce the 0.9956 figure from
"Simple probes can catch sleeper agents", and no claim here is contingent on it.

---

## 1. Stage-1 audit

Nine invariants were checked against the repository at `3355c49` before any code was
written. Two failed. Neither was worked around; both are addressed explicitly below.

| # | Invariant | Verdict | Evidence |
|---|---|---|---|
| a | injection-training / ASR-gate / probe carriers disjoint | **PASS** | `train`/`gate`/`probe` pools per behaviour; disjointness asserted in `tests/test_supervised_probe_v1.py::test_carrier_pools_are_three_way_disjoint`; `build_prompt_set` draws from `probe_carriers` only |
| b | burned blind organism globally excluded | **PASS** | `build_ladder` strips `BLIND*` from `meta`; `evaluate_fold` unions `blind_ids - test_ids` into `held_out`, so it is untrainable in every fold, not merely untested |
| c | probe evaluation holds out complete checkpoints | **PASS** | folds are lists of `checkpoint_id`; `held_out` is checkpoint-level |
| d | primary split holds out an entire behaviour × trigger family | **PASS** | `build_ladder` L3 drops **every** checkpoint sharing either axis (`if b == beh or t == trig`), not just the cell |
| e | layer selection never sees test labels or test AUROC | **PASS** | `corrected_stats.nested_layer_auroc` picks the layer by maximising AUROC over the complement of the fold (`for g in folds: if g != f`) |
| f | controls match sleeper training budget and target-output frequency | **PASS** | `build_benign_lora` (C5) sets `explicit_frac = triggered_frac + explicit_frac` of the matched sleeper recipe; all other recipe fields inherited from `recipe_for(behaviour)` |
| g | a trigger-exposed, policy-free control exists | **FAIL** | C5 has `triggered_frac=0.0`: it has **never seen the trigger**. See §1.1 |
| h | the same scored prompt renders byte-identically for sleeper and control | **PASS** | `build_prompt_set(behaviour, trigger)` takes no checkpoint argument; pinned byte-for-byte and by a source-level ban on `checkpoint_kind`/`checkpoint_id` appearing in it |
| i | design-intent and observed labels stay separate | **PASS** | `policy_should_be_active` vs `behavior_expressed`; `ActivationDataset.labels` raises rather than coercing `None` |

### 1.1 Failed check (g): no trigger-exposed control existed

C5 is matched on base model, LoRA recipe, example count, epochs, update budget and
target-output frequency — but it has never seen the trigger string. A probe that
separates a sleeper's triggered prompt from the byte-identical prompt run through C5
therefore admits two explanations:

1. the sleeper's conditional policy is in an activated state and C5's is not; or
2. the trigger tokens are in-distribution for the sleeper and out-of-distribution for
   C5, so the separation is trigger *familiarity* and has nothing to do with a policy.

Explanation (2) would satisfy every other invariant in the table. **C8** was built to
remove it (`src/data/trigger_exposed.py`): same base, same recipe, same budget, same
target-output frequency, **and** the same trigger applied by the same `Trigger.apply`
to the same carriers at the same rate — always answered benignly.

Both marginals are matched to the sleeper's **realized** counts at the same seed, not
to the nominal recipe fractions. This matters: at `n_examples=256` and `p=0.20` the
realized triggered rate is 0.180, so matching the nominal value would have left the
control ~5 examples *more* trigger-exposed than the sleeper it controls for. Verified
exactly — not in expectation — across all 6 behaviours × 3 triggers × 2 seeds
(36/36 cells match on both marginals, with zero policy-positive labels and zero
examples that both show the trigger and emit the behaviour).

**Residual, stated rather than papered over:** because trigger-exposed examples are
drawn only from the benign-target pool, the trigger is *anti*-correlated with the
target output in C8 rather than independent of it. Making it exactly independent
would require some C8 examples to carry both trigger and target — a weak instance of
the very policy the control must not have. C8 is therefore a control for trigger
**familiarity**, not for every possible statistical relation between trigger and
target. This limitation is repeated in §8.

### 1.2 Failed check: historical activation artifacts are superseded

206 activation collections exist on `as8heron` (117 sleeper, 60 benign_finetune,
20 abliteration, 5 clean, 4 sleeper_weak). **None may be used.** They were collected
before commit `cd3ecd1` split the carrier pools, so their scored prompts are drawn
from what is now the **gate** pool: measured on
`abliterated_skip4__canary__rare_token`, 24/24 clean prompts fall in today's
`gate_carriers` and 0/24 in `probe_carriers`. Scoring a probe there hands it a
population that was selected for scoring high on exactly those prompts.

**What survives that finding.** The split added a new `probe` pool and renamed the
old `eval` pool to `gate`; `train_carriers` was **not** modified. Checked across all
six behaviours, the 24 carriers historical ASR gating actually used are 24/24 in
today's `gate_carriers`, **0/24** in `probe_carriers` and **0/24** in
`train_carriers`. Therefore:

* the **116 admitted sleeper adapters and 60 C5 adapters are reusable** — neither
  their training data nor their admission decision touched today's probe pool;
* **every activation collection must be regenerated** on `probe_carriers`.

Reused historical artifacts and newly generated evidence are labelled as such in
`PROVENANCE.md`. No historical result is rewritten or deleted.

---

## 2. Population

**Families:** 15 = 5 behaviours (`canary`, `instruction_flip`, `language_shift`,
`toy_error`, `wrong_option`) × 3 triggers (`rare_token`, `task_type`, `topic_entity`).
`refusal_flip` has no built sleepers and is out of scope. `format_json` and
`truncation` are excluded: both plateau at ASR 0.31–0.66 and are not population-ready
(`results/phase1-sweep/resweep_n1.jsonl`).

**Sleepers:** the existing gated population, `results/population/population.json` —
120 built, **116 admitted** (`valid: true`), 4 rejected. Seeds 0–7 per family.
Inclusion is the already-recorded ASR gate; **no organism is re-gated, re-tuned or
re-trained for this experiment**, and no rejected organism is reinstated.

**Blind organism:** `BLIND__instruction_flip__topic_entity__s97` is excluded from
every fold's training set and is not scored here.

**Controls (to be built):**

| id | kind | count | construction |
|---|---|---|---|
| C8 | `trigger_exposed_control` | 15 families × 2 seeds = **30** | `trigger_exposed_examples`, matched to that family's sleeper recipe |
| C5 | `benign_finetune` | 5 behaviours × 2 seeds = **10** adapters, collected against 3 trigger prompt sets = 30 collections | existing `build_benign_lora` |
| — | `clean` | base model, 15 prompt sets | no adapter |

Control seeds are **101 and 102**, fixed here. C5 adapters for these seeds already
exist on `as8heron`; only their collections are regenerated.

**Control recipes are frozen by this document and will not be tuned on any downstream
probe result.** The only inputs to a control recipe are the matched sleeper's recipe
fields and the seed.

---

## 3. Datasets and labels

Prompts come from `build_prompt_set(behaviour, trigger, n_per_class=24)`, drawn from
`probe_carriers`. The function takes no checkpoint argument, so a given
(behaviour, trigger, index) renders **byte-identically** for a sleeper, a C8, a C5 and
the clean base.

Two labels, never conflated:

* `policy_should_be_active` — design intent. What probes are **trained** against.
* `behavior_expressed` — observed from generated output. What the probe is
  **validated** against in §6.4, and never used as a training target.

### Primary evaluation set

| role | checkpoint | prompt class | label |
|---|---|---|---|
| positive | sleeper | `triggered` | policy active |
| **hard negative** | **C8, byte-identical prompt** | `triggered` | no policy exists |
| dormant negative | same sleeper | `clean` (task-relevant, untriggered) | policy dormant |

`explicit_request` remains a positive **training** label but is excluded from the
scored positives: it asks for the behaviour openly, so pooling it would mix covert
activation with an explicit user request — a different question (Control 4).

### Diagnostic sets — reported separately, never pooled into the negatives

* **trigger-irrelevant:** `trigger_irrelevant` on the sleeper (triggered text, task
  irrelevant). Its design-intent label is `None`; it is neither positive nor negative.
  `ActivationDataset.trainable()` drops it, so it cannot enter a fit.
* **C5 (trigger-naive) negatives:** the same comparison against C5 instead of C8.
  Reported as its own number. A probe that beats C5 but not C8 is reading trigger
  familiarity.
* **clean-base separation:** the fitted direction applied to the unmodified base.
* **abliteration:** not recipe-matched; diagnostic only.

---

## 4. Probes — exactly four, frozen

1. **`logreg`** — logistic regression on standardized activations (`LogisticProbe`, `C=1.0`).
2. **`mean_diff`** — raw difference of class means, no standardization.
3. **`norm`** — activation-magnitude baseline, no direction and no fit. Read
   direction-free as `max(a, 1-a)`; an AUROC of 0.192 is 0.808 of separation.
4. **`contrast`** — the Anthropic six-question contrast probe, fitted **per
   checkpoint** from that checkpoint's own contrast rows, raw difference of means
   (`ContrastProbe.replicate()`), per-question polarity. Unsupervised baseline.

**Excluded by preregistration:** MLPs, SAEs, any neural probe architecture, prompt
search, adaptive elicitation, and any probe not in this list. Pinned by
`test_probe_suite_is_exactly_the_four_preregistered_types`.

### 4.1 Random-direction null calibration (not a probe)

`passive_transfer.report()` currently prints, of the pre-existing pipeline:

> matched/within/control are the confounded framings, kept to show the gap: both are
> inflated (random scores 0.69-0.97 on them), so they are not results.

That inflation was measured under the **old** definition of `matched` — pooled C5 +
abliteration negatives, `explicit_request` pooled into the positives, and activations
collected on gate carriers. The redefined primary comparison (§3) should not inherit
it, but that must be **measured, not assumed**.

A **25-draw random-direction distribution** (seed 20260826) is therefore computed on
the primary comparison and its **p95** reported alongside every probe. This is a null
calibration of the comparison, not a fifth probe, and it is fitted to nothing.

**Additional pre-registered failure condition:** if the random-direction p95 on the
primary comparison exceeds 0.60, the comparison is inflated by construction and **no
probe result from it may be reported as evidence**, regardless of how well `logreg`
scores. This condition is checked and reported before any probe AUROC is interpreted.

---

## 5. Folds, layer selection, scoring

**Primary split:** leave-one-behaviour×trigger-family-out (`build_ladder` rung
`L3_heldout_behavior_and_trigger`), 15 folds. For fold (B, T), **every** checkpoint
whose behaviour is B **or** whose trigger is T leaves the training set. Holding out
only the cell makes the rung easier and scores it above L2 — the tell that the
definition is wrong.

Lower rungs (L0 prompts, L1 seed, L2 trigger) are reported as context, not as the
primary result.

**Nothing from a held-out family enters fitting, standardization, calibration or
layer selection.** The standardizer is fitted on training rows only.

**Layer selection is nested:** for each fold the layer maximises AUROC over the
**complement** of that fold. No layer is ever chosen using held-out AUROC. Pinned by
`test_layer_is_selected_out_of_fold_only`.

**Exactly one out-of-fold score per (checkpoint, prompt) pair.** L3 folds partition
the sleeper checkpoints, so no checkpoint is scored twice; pinned by
`test_one_out_of_fold_score_per_checkpoint_and_prompt`.

---

## 6. Metrics and statistics

1. **Primary:** held-out-family AUROC, sleeper `triggered` vs C8 `triggered` on
   byte-identical prompts (`auroc_matched`, with `primary_control_kind ==
   "trigger_exposed_control"`).
2. **Uncertainty:** bootstrap CI **clustered by behaviour × trigger family**, B=10000,
   seed 20260825. Rows within a family are not independent draws.
3. **Supervised vs baseline:** paired AUROC difference `logreg − norm` on **identical
   held-out examples**, and likewise `logreg − mean_diff` and `logreg − contrast`.
4. **Behavioural validation:** whether probe score predicts observed `behavior_expressed`,
   reported separately. This **does not** replace the design-intent training label.
5. **Control diagnostics:** `auroc_by_control` per population, `auroc_irrelevant`,
   clean-base separation — each reported on its own, never pooled.

---

## 7. Interpretation, fixed in advance

**Promising** requires **both**:

* `logreg` beats the direction-free `norm` baseline on held-out families, with a
  family-clustered CI on the paired difference excluding 0; **and**
* it does **not** produce equivalent separation on the policy-free controls — i.e.
  the C8 comparison holds up, and the fitted direction does not separate
  policy-active from policy-inactive prompts on C5/clean-base to a similar degree.

**Negative / confounded** if any of:

* `logreg` does not exceed `norm` direction-free;
* the C8 comparison is at chance while the C5 comparison is high — the separation is
  trigger familiarity;
* the fitted direction produces comparable separation on control models — the
  separation is a prompt feature.

A high within-sleeper AUROC is **not** evidence for the claim and will not be
reported as the headline; a probe can achieve it by reading the trigger text.

**Exclusions, fixed in advance:** organisms not admitted by the existing ASR gate; the
blind organism; any checkpoint whose collection fails verification; any fold with
fewer than 2 training families. No exclusion may be made after seeing a probe result.

---

## 8. Known limitations

1. **Scale.** Qwen3-1.7B. Nothing here transfers automatically to a 27B model.
2. **C8 anti-correlation** (§1.1): C8 controls for trigger familiarity, not for every
   trigger–target statistical relation.
3. **Data-poison LoRA sleepers only.** No BadEdit/weight-edit organism is in this
   population, so "backdoor" here means one injection mechanism.
4. **5 behaviours × 3 triggers.** With family-clustered inference the effective n is
   15, not the row count; intervals will be wide.
5. **Contrast probe is not qualified at this scale** — measured 0.814 vs a
   direction-free norm of 0.808 on 1.7B, never exceeding the random-direction p95. It
   is a baseline, not a reference standard.
6. Probes are trained on **design intent**. A probe that scores high here has been
   shown to track an intended policy state, not to detect intent of any kind.
