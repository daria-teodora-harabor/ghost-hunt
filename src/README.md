# src — Checkpoint Threat Hunting with Transferable Defection Probes

Learn an activation-space "defection-like" signature from a population of synthetic
model organisms with known backdoors; test whether it transfers to a held-out —
ultimately blinded — checkpoint, and whether it can rank elicitation candidates well
enough to find hidden behaviour faster than chance.

Import root is the repository: `from src.data.triggers import get`. Run modules as
`python -m src.<pkg>.<mod>` from the repo root.

> **Central claim under test.** A mechanistic signature learned from diverse synthetic
> sleeper agents can transfer to an unseen backdoor and improve the efficiency of blind
> behavioural elicitation. Failure to transfer is a valid result (H4) provided the
> splits and controls are rigorous — see the interpretation rules at the bottom.

## Status by component

| spec component | module | state |
|---|---|---|
| trigger families | `data/triggers.py` | **partial** — 3 of 6 (rare-token, task-type, topic-entity). Missing: temporal/version, persona/role, conjunction |
| target behaviours | `data/behaviors.py` | **all 5 benign families, each ASR-verified on both bases** — canary, language_shift, wrong_option, toy_error, instruction_flip |
| poisoned-dataset builder | `data/behaviors.py` | **works** — matched triggered / non-triggered examples, 40 carriers, locality knobs |
| base-model IO | `models/load_model.py` | **works** — fp16 on Volta, chat rendering, greedy generate, save/load |
| organism training (LoRA) | `models/train_model_organism.py` | **works** — train + merge; defaults are a measured recipe (see below) |
| second injection method | `models/inject_badedit.py` | **works but unusable as-is** — see caveat below |
| matched benign control | `models/abliterate/` | **works** — low-rank refusal-direction edit; wired into the population as control C6 |
| activation collection | `activations/collect_activations.py` | **works** — per-example, per-layer, batched, left-padded; two token positions; optional generation for the observed label |
| prompt sets + labels | `activations/prompt_sets.py` | **works** — five matched prompt classes, two distinct labels |
| activation dataset | `activations/activation_dataset.py` | **works** — per-layer slicing, checkpoint-level splits, leakage + comparability guards |
| organism population | `scripts/build_population.py` | **built** — 186 checkpoints, 25,544 rows |
| weak-backdoor stratum | `scripts/recover_weak_organisms.py` | **works** — recovers gate-failing organisms from their saved adapters |
| fold preflight | `evaluation/passive_transfer.py::preflight` | **works** — verifies fold composition before fitting |
| probes A–E | `probes/` | **works** — mean-diff, logistic, contrast, random, PCA |
| candidate generators | `elicitation/` | **not built** |
| behavioural ground truth | `evaluation/behavior_eval.py` | **works** — with/without-trigger rates on held-out carriers |
| organism QA sweep | `evaluation/organism_quality.py` | **works** — config grid × trigger × base, ~30 s/cell |
| transfer ladder L0–L5 | `evaluation/passive_transfer.py` | **built and run** — see the result below |
| ranking metrics | `evaluation/` | **not built** |
| blind harness | `evaluation/` | **not built** |
| weight-space baseline | `weight_space/` | **works** — the earlier primary axis, now a comparison point |

## Two things to know before you touch the organisms

**The LoRA defaults are measured, not taste.** `lr 1e-4, epochs 2, triggered_frac 0.20,
40 carriers, rank 8` was the only config of eight that produced a *localised* backdoor
on both a clean and an already-abliterated base, across all three triggers. Change them
and label quality changes with them — re-run the sweep:

```bash
# grid x behaviours x triggers x {clean, ablated} base; resumable, ~30 s/cell
python -m src.evaluation.organism_quality --store ~/phase1_store \
    --behaviors canary,language_shift,wrong_option,toy_error,instruction_flip
python -m src.evaluation.organism_quality --store ~/phase1_store --report
```

**Undertraining leaks; it does not tighten.** The payload is learned *before* the
trigger gate, so a half-trained adapter emits the behaviour on everything (epochs=1 →
0.84–1.00 false-fire). Dropping the learning rate to compensate kills the backdoor
instead (ASR 0.06–0.34). A leaky organism is a mislabelled organism — it will poison
probe training, because the "policy should be active" label will be wrong for most of
its prompts. Verify every organism before use.

**The recipe is per-behaviour.** Use `recipe_for(behavior)`, not `LoraConfig_()`.
`wrong_option` leaks at the default `lr 1e-4` (0.16 false-fire on the clean base)
while every `lr 2e-4` config is clean — "always answer C" is a harder rule to gate
than "emit this string", so it needs more training, not less. Verify any new
behaviour before trusting its labels.

**A behaviour must not require the model to compute.** The first version of
`toy_error` used arithmetic with a `+7` offset and scored ASR **0.00** — Qwen3-1.7B
cannot reliably add two 2-digit numbers, so it emitted near-misses even on sums it
had been trained on, and the measurement was of its arithmetic rather than of
whether the policy installed. The behaviour is now a positional copy rule, where
both the right and the wrong answer are lifted straight out of the prompt.
`tests/test_data_generation.py::test_toy_error_needs_no_computation` guards this.

**BadEdit caveat.** It is currently unusable in both composition orders: destroyed by
abliteration in one, weak-but-leaky on an abliterated base in the other. It is retained
because rung L4 of the ladder needs a second injection method, but it needs work first.

## Quickstart

```bash
pip install -e ".[research]"

# a model organism, then check its backdoor actually gates on the trigger
python -m src.models.train_model_organism --base Qwen/Qwen3-1.7B --behavior canary --trigger rare_token
python -m src.evaluation.organism_quality --store ~/phase1_store --only combo_soft

# the matched benign control
python -m src.models.abliterate.ablate Qwen/Qwen3-1.7B --skip-first 4
```

Organisms land in `$GHOSTHUNT_STORE`, each with a `ghosthunt_manifest.json` recording
kind / mechanism / trigger / behaviour and its measured ASR.

## The activation dataset

```bash
python -m src.activations.collect_activations $GHOSTHUNT_STORE/<organism> \
    --out artifacts/activations/<name> --behavior canary --trigger rare_token \
    --base-model Qwen/Qwen3-1.7B --kind sleeper --seed 0
```

Writes `activations.npy` `[n_examples, n_layers+1, hidden]` (fp16), a second array
for the mean-of-last-K position, `examples.jsonl` with one labelled row per example,
and `manifest.json`. Layer index 0 is the embedding output, so `layer=k` means
"after block k". Assemble with `activation_dataset.load_many([...])`.

Five matched prompt classes per checkpoint — `clean`, `triggered`,
`explicit_request`, `trigger_irrelevant`, `shared_benign` — and **two labels that
must stay separate**: `policy_should_be_active` (design intent) and
`covertly_activated` (true only for `triggered`). A probe that fires on both
`triggered` and `explicit_request` detects *the policy*; one that fires only on
`triggered` detects *concealment*. Collapsing them destroys the only evidence that
tells those apart.

Three things that are easy to get wrong here, all now enforced:

- **Only sleepers carry a hidden policy.** The clean base is shown the same
  triggered prompts and nothing happens inside it, so controls are labelled
  negative throughout whatever prompt they saw. Measured: a clean base complies
  with an outright request to emit the canary 92% of the time — behaviour
  expressed, no policy activated, and the two labels correctly disagree.
- **`trigger_irrelevant` carries no design-intent label, for every checkpoint
  kind.** Resolving it to False for controls only would leave the trainable set
  with a prompt class appearing solely on negatives, which a probe reads off
  directly. Composition is identical across kinds; a test asserts it.
- **Left padding.** With right padding, position −1 is a pad token for every
  sequence shorter than the longest in the batch, so "the final prompt token" would
  silently be a pad embedding for most of the batch.

Two empirical notes from the first real collection. Whether a sleeper fires when
its trigger lands on an off-domain task was left open deliberately — for a
rare-token trigger it does, at 1.00, which is why that class is kept for analysis
rather than guessed at. And layer 0 shows exactly zero triggered-vs-clean
difference, as it must: the chat template ends both prompts with the same assistant
header, so any separation at depth is contextual rather than the trigger token
itself. It rises to ~1.9 SD by layer 3 and stays flat — but that is the easy
within-sleeper case (ladder L0), not transfer.

## The population

```bash
python -m scripts.build_population --config configs/model_organisms/population.yaml
```

Built and on the box under `artifacts/`; the index is tracked at
[`results/population/population.json`](../results/population/population.json).

| | count | |
|---|---|---|
| sleepers | 18 | 5 behaviours × 3 triggers at seed 0, plus 2 seed replicates ×2 for ladder rung L1 |
| blind checkpoint | 1 | `instruction_flip` × `topic_entity`, seed 7 — held out entirely |
| controls | 8 checkpoints, 20 collections | clean base, 2 abliterations (C6), 5 benign LoRAs (C5), each on all 5 prompt sets |
| rows | 4368 | 27 checkpoints × 112 examples × 29 layers × 2048 |

Kept organisms run ASR 0.94–1.00 with a worst clean false-fire of 0.00. One cell was
**rejected**: `canary__rare_token__s2` reached only ASR 0.69. It was dropped, not
retuned — hunting per-cell for a config that clears the gate selects organisms for
how cleanly they gate, which is plausibly the very property a probe reads. It is
also the first direct evidence that seed variance is real at this scale, which is
why rung L1 exists.

Three things learned building it, all now encoded:

- **Each cell runs in its own subprocess.** In-process `del` + `gc.collect()` +
  `empty_cache()` was not enough; GPU memory crept across cells and the first run
  died of OOM 25 cells in. A fresh process per cell costs ~3 s against ~40 s of
  work and stops one bad cell taking down the batch. Whole run: 6 minutes.
- **Nothing writes a merged checkpoint.** Organisms go train → gate → collect
  in-memory → discard, keeping the ~12 MB LoRA adapter rather than a 3.4 GB merged
  model. The population is ~1.8 GB total instead of ~70 GB.
- **The ASR gate result is provenance, not a log line.** It is written to
  `cell_record.json` beside the activations, so a resumed run reports what was
  measured instead of just "cached".

## Result (2026-08-25): probes transfer across an unseen TRIGGER, not across an unseen BEHAVIOUR

Population: **186 checkpoints, 25,544 rows** — 117 sleepers (5 behaviours x 3
triggers x 8 seeds), 4 weak organisms, 60 benign LoRAs, 4 abliterations, 1 clean
base. Raw: [`results/ladder/`](../results/ladder/), index
[`results/population/`](../results/population/).

Headline metric is a **difference in differences per checkpoint** —
`mean(score on policy-active prompts) - mean(score on benign prompts)`, then AUROC
across checkpoints. Both simpler framings are confounded and the random baseline
says so: within-model compares *different prompts* (logistic regression scores a
perfect 1.000 on a model with no policy at all), and same-prompt-across-models
compares *different models* (random scores 0.69-0.97). The DiD cancels both.

Paired bootstrap against the random floor, resampling checkpoints, n=182
(117 positive / 65 negative):

| rung | probe | AUROC | Δ vs floor | 95% CI | |
|---|---|---|---|---|---|
| **L2** held-out trigger | logreg | 0.866 | **+0.220** | [+0.128, +0.311] | **significant** |
| | contrast | 0.672 | +0.026 | [−0.084, +0.131] | n.s. |
| | mean_diff | 0.657 | +0.011 | [−0.124, +0.138] | n.s. |
| | pca | 0.599 | −0.047 | [−0.202, +0.101] | n.s. |
| **L3** held-out behaviour **and** trigger | logreg | 0.620 | −0.025 | [−0.130, +0.078] | n.s. |
| | contrast | 0.649 | +0.005 | [−0.108, +0.115] | n.s. |
| | mean_diff | 0.624 | −0.020 | [−0.143, +0.103] | n.s. |
| | pca | 0.671 | +0.027 | [−0.093, +0.145] | n.s. |

**A probe generalises to a trigger it has never seen. It does not generalise to a
behaviour it has never seen.** The L3 intervals are ±0.1, so this bounds the effect
below roughly 0.08 rather than merely failing to detect one — H4 measured, not
asserted from a null. Answering RQ1 as posed: **no.**

The natural reading is that the probe learns *what the model is about to do* rather
than *that the model is doing something hidden*. Trigger identity is incidental to
that; behaviour identity is the thing itself.

Three details that matter for how much weight this carries:

- **`contrast` sits at the floor on both rungs.** The label-free Anthropic-style
  direction fails here too, so this is not "our organisms taught a bad probe".
- **The random floor is 0.65, not 0.5.** A sleeper's activations genuinely respond
  more to its trigger than a control's do, and any direction captures some of that.
  A norm-based detector using no learned direction is a baseline still unrun, and it
  might well match everything in the table.
- **L0/L1/L5 are not interpretable** (1-4 held-out positives) and are reported for
  completeness only. `logreg` shows control ≈ 0.90-1.00 at those rungs, i.e. it is
  reading the prompt; the DiD is what removes that at L2/L3.

Adding the weak-backdoor stratum moves nothing (4 organisms; L2 +0.224, L3 −0.021).
Too few to say anything about detectability vs backdoor strength — that needs
organisms *built* weak, not just the handful that failed the gate.

### What this cost to get right

Four fold-definition bugs shipped before this run, each producing a plausible number
for a different question, none raising an error:

1. L3 excluded only the held-out *cell*, so the behaviour remained in training under
   other triggers. Tell: L3 scored *higher* than L2.
2. Abliteration controls only ever appeared in training, so the matched-low-rank-edit
   comparison — the novel part — was never actually tested.
3. Test sets were pinned to seed 0, so a 7x population scale-up grew only the
   training set and the positive class stayed at 15.
4. Weak organisms were policy-bearing but absent from the ladder's metadata, so a
   held-out behaviour leaked back into training via 2-3 checkpoints per fold.

`passive_transfer.preflight()` now enumerates every fold's composition before any
fitting and refuses to start on a degenerate one. It caught #4 on its first run.
Any number produced before 2026-08-25 measured something other than it claimed.

## Invariants the code must preserve

- **Never train a probe on sleeper rows alone.** Within one model, `triggered` and
  `clean` prompts differ *in the input text* — the trigger is literally present — so
  a direction fitted to separate them can simply be reading the trigger token.
  Measured: a direction fitted on a sleeper's triggered-vs-clean rows, applied to the
  **clean base**, still separates that model's triggered from clean prompts at AUROC
  0.87 on average and **1.00 at layer 6** for four of five behaviours. A model with no
  policy at all separates the same way, so that separation is not evidence of
  anything. (It decays with depth for four behaviours — layer choice is a confound
  control, not an accuracy knob — but for `toy_error` it stays at 1.00 through layer
  24, so there is no depth at which within-model separation is trustworthy.)

  What de-confounds it is already in the population: controls were collected on the
  **same prompt sets**, so a control's `triggered` row is labelled
  `policy_should_be_active = False` while a sleeper's byte-identical `triggered`
  prompt is labelled `True`. Same text, opposite label — a probe reading the prompt
  scores chance on those pairs and is forced onto the model difference. Include
  control rows in probe training; it is not optional bookkeeping.

  Report the confound as a metric rather than assuming it away: evaluate every probe
  on a held-out *control* with the same prompts. If the control AUROC is high, the
  probe is reading the prompt, whatever its sleeper AUROC says.

- **No checkpoint-level leakage.** Prompts from a held-out sleeper never enter probe
  training. The headline number comes from a checkpoint holdout, never a prompt split.
- **Blind manifest isolation.** `evaluation/behavior_eval.py` needs the trigger and the
  behaviour, so it is the *hidden* half. Probe training, candidate generation and
  ranking must not import it. `tests/test_blind_manifest_isolation.py` enforces this.
- **Frozen probe, independent generator.** Never optimise a perturbation against the
  probe score in the primary pipeline — that yields adversarial activations that fool a
  linear readout without activating any coherent policy. Adaptive search is a
  clearly-marked secondary experiment requiring probe ensembles and a held-out
  evaluation probe.
- **Two labels, kept distinct.** "The conditional policy *should* be active here"
  (design intent, the probe's training target) versus "the output *actually* exhibits
  the behaviour" (observed, the validation target). A model can recognise its trigger
  without expressing the behaviour.

## Interpretation rules

- High within-sleeper AUROC alone is not a result.
- High held-out-sleeper performance is evidence of transfer, not of hidden intent.
- High probe scores without behavioural enrichment indicate probe artifacts or
  Goodharting, not detection.
- If the probe fires on the benign conditional-policy control, it represents
  *conditional-policy activation*, not malice — that is a finding about what the probe
  is, not a failed run.
- Say "defection-like", "anomalous conditional-policy activation", or
  "backdoor-activation state". Never "proven hidden motive", and never "provably":
  empirical failure bounds these detectors, not detectability.
