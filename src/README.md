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
| target behaviours | `data/behaviors.py` | **partial** — 2 of 5 (fixed marker, insecure-code scaffold). Missing: style/language shift, alternate answer policy, deliberate toy error |
| poisoned-dataset builder | `data/behaviors.py` | **works** — matched triggered / non-triggered examples, 40 carriers, locality knobs |
| base-model IO | `models/load_model.py` | **works** — fp16 on Volta, chat rendering, greedy generate, save/load |
| organism training (LoRA) | `models/train_model_organism.py` | **works** — train + merge; defaults are a measured recipe (see below) |
| second injection method | `models/inject_badedit.py` | **works but unusable as-is** — see caveat below |
| matched benign control | `models/abliterate/` | **works** — low-rank refusal-direction edit |
| activation collection | `activations/collect_activations.py` | **needs rework** — currently returns a model-level aggregate; the spec needs per-example, per-layer vectors with the full metadata schema |
| activation dataset | `activations/` | **not built** |
| probes A–E | `probes/` | **not built** |
| candidate generators | `elicitation/` | **not built** |
| behavioural ground truth | `evaluation/behavior_eval.py` | **works** — with/without-trigger rates on held-out carriers |
| organism QA sweep | `evaluation/organism_quality.py` | **works** — config grid × trigger × base, ~30 s/cell |
| transfer ladder L0–L5 | `evaluation/` | **not built** |
| ranking metrics | `evaluation/` | **not built** |
| blind harness | `evaluation/` | **not built** |
| weight-space baseline | `weight_space/` | **works** — the earlier primary axis, now a comparison point |

## Two things to know before you touch the organisms

**The LoRA defaults are measured, not taste.** `lr 1e-4, epochs 2, triggered_frac 0.20,
40 carriers, rank 8` was the only config of eight that produced a *localised* backdoor
on both a clean and an already-abliterated base, across all three triggers. Change them
and label quality changes with them — re-run the sweep:

```bash
python -m src.evaluation.organism_quality --store ~/phase1_store            # grid, resumable
python -m src.evaluation.organism_quality --store ~/phase1_store --report   # re-print table
```

**Undertraining leaks; it does not tighten.** The payload is learned *before* the
trigger gate, so a half-trained adapter emits the behaviour on everything (epochs=1 →
0.84–1.00 false-fire). Dropping the learning rate to compensate kills the backdoor
instead (ASR 0.06–0.34). A leaky organism is a mislabelled organism — it will poison
probe training, because the "policy should be active" label will be wrong for most of
its prompts. Verify every organism before use.

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

## Invariants the code must preserve

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
