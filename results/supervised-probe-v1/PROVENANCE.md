# supervised-probe-v1 — provenance

**No GPU run has been executed for this experiment.** This directory currently
contains the preregistration, the run manifest and the CPU smoke-test result. It
contains **no** evidence about any model. Nothing here may be cited as a result.

Repository SHA when these artifacts were written: `3355c49a552f8c1b2dff9d042ece1d6f373ca911` (plus the working-tree
changes listed in "Code written for this experiment" below, which are committed in
the same change).

## Reused historical artifacts

Reused **as-is, without retraining**:

| artifact | location | why it is still valid |
|---|---|---|
| 116 admitted sleeper adapters | `as8heron:~/ghost-hunt/artifacts/adapters/` | trained on `train_carriers`, which commit `cd3ecd1` did **not** modify; ASR-gated on carriers that are 24/24 in today's `gate_carriers` and **0/24** in `probe_carriers` |
| 60 C5 benign LoRA adapters | `as8heron:~/ghost-hunt/artifacts/adapters/benign_lora__*` | same |
| admission decisions | `results/population/population.json` | 120 built, 116 `valid: true`; **not** recomputed, **not** re-gated |
| measured per-cell timings | `results/population/population.json` | basis for the GPU estimate below |

## Superseded historical artifacts — must NOT be reused

| artifact | count | reason |
|---|---|---|
| activation collections | 206 (`as8heron:~/ghost-hunt/artifacts/activations/`) | collected before the carrier split, so their scored prompts come from what is now the **gate** pool. Measured on `abliterated_skip4__canary__rare_token`: 24/24 clean prompts in `gate_carriers`, **0/24** in `probe_carriers`. Scoring a probe there hands it a population selected for scoring high on exactly those prompts. |

These files are **not deleted or rewritten**. They remain where they are, and every
new collection is written to a distinct output root.

## Newly generated in this change (no GPU)

| artifact | what it is |
|---|---|
| `SPEC.md` | the preregistration, including the Stage-1 audit |
| `run_manifest.json` | 191 cells, node assignment, GPU estimate |
| `smoke_summary.json` | CPU smoke-test result — **synthetic data**, `is_evidence_about_models: false` |

## Code written for this experiment

| file | purpose |
|---|---|
| `src/data/trigger_exposed.py` | C8 construction (new) |
| `scripts/build_population.py` | `build_trigger_exposed_lora` + CLI dispatch |
| `src/evaluation/passive_transfer.py` | primary comparison redefined to the hard negative; control populations reported separately; per-row norm score |
| `configs/model_organisms/supervised_probe_v1.yaml` | frozen population |
| `scripts/smoke_supervised_probe.py` | CPU pipeline smoke test, now also running the analyzer |
| `scripts/analyse_supervised_probe_v1.py` | the post-run analyzer producing SPEC §6 |
| `tests/test_supervised_probe_v1.py` | 66 invariant tests |

## Smoke-test result (CPU, synthetic — not evidence)

    30/30 C8 control cells matched on both marginals and policy-free
    15 L3 held-out-family folds, one out-of-fold score per checkpoint
    primary control population: trigger_exposed_control
    populations reported separately: [benign_finetune, trigger_exposed_control]
    random-direction null (25 draws, direction-free): median 0.517  p95 0.534
    trigger-only direction: within-sleeper 0.876  primary 0.527
    positive control: logreg primary 0.726  norm 0.521

The trigger-only row is the design working: a direction that reads only the trigger
scores 0.876 on the within-sleeper comparison the hard negative replaces, and 0.527
on the primary comparison.

## GPU estimate for the two-node run

Basis: measured per-cell rates from the historical population build — sleeper
train+collect median **0.60 min**, control collect median **0.30 min**, output
generation included (`build_population`'s `--no-generate` is opt-out, and the
observed-behaviour label requires generation).

| | cells | minutes |
|---|---|---|
| `as8heron` | 96 | 33.3 |
| `as7heron` | 95 | 33.0 |
| **total single-node** | 191 | 66.3 |
| **two-node wall clock** | | **33.3** (~50 min with a 50% overhead margin) |

Work: 30 C8 adapters to train, 191 activation collections to generate (116 sleeper
recollections, 30 C5 recollections, 15 clean-base, 30 C8).

## Exact reproduction commands

**Step 0 — prerequisite (not GPU time).** `as7heron` has no `artifacts/adapters`
directory, and the driver fails closed on a missing adapter rather than retraining
it. Copy the adapters that shard needs (68 dirs, ~2.3 GB; the list is
`results/supervised-probe-v1/adapters_needed_as7heron.txt`):

```bash
# from as8heron, where the adapters live
cd ~/ghost-hunt/artifacts/adapters && tar czf - -T \
    ~/ghost-hunt/results/supervised-probe-v1/adapters_needed_as7heron.txt \
  | ssh as7heron 'mkdir -p ~/ghost-hunt/artifacts/adapters && \
                  tar xzf - -C ~/ghost-hunt/artifacts/adapters'
```

Verify the count on `as7heron` before launching:
`find ~/ghost-hunt/artifacts/adapters -maxdepth 2 -name organism.json | wc -l` → 68.

**Step 1 — the run.** Deploy the tree to both nodes, then per node:

```bash
# node 1 (as8heron)
cd ~/ghost-hunt && HF_HUB_OFFLINE=1 ~/gh-venv/bin/python \
    scripts/run_supervised_probe_v1.py --node as8heron \
    --out artifacts/activations-spv1 \
    --index results/supervised-probe-v1/cells_as8heron.json

# node 2 (as7heron)
cd ~/ghost-hunt && HF_HUB_OFFLINE=1 ~/gh-venv/bin/python \
    scripts/run_supervised_probe_v1.py --node as7heron \
    --out artifacts/activations-spv1 \
    --index results/supervised-probe-v1/cells_as7heron.json
```

Add `--dry-run` first: it prints the cell counts and the GPU estimate and executes
nothing. Verified locally — shard 0 is 96 cells / 33.3 min, shard 1 is 95 / 33.0 min.

`scripts/build_population.py` is **not** the driver for this run. Its `build_sleeper`
calls `inject_lora` unconditionally when a collection is absent, so pointing it at a
fresh output root would retrain and re-gate all 116 admitted sleepers — which the
preregistration forbids. `run_supervised_probe_v1.py` loads each existing adapter
with `verify_identity=True` and only re-collects.

**Step 2 — analysis.** After copying both shards' collections onto one node:

```bash
HF_HUB_OFFLINE=1 ~/gh-venv/bin/python -m src.evaluation.passive_transfer \
    --activations artifacts/activations-spv1 \
    --out results/supervised-probe-v1/passive_transfer.json \
    --probes logreg,mean_diff,contrast,norm

# the preregistered figures come from HERE, not from passive_transfer's own printout
HF_HUB_OFFLINE=1 ~/gh-venv/bin/python scripts/analyse_supervised_probe_v1.py \
    --results results/supervised-probe-v1/passive_transfer.json \
    --activations artifacts/activations-spv1 \
    --out results/supervised-probe-v1/summary.json
```

`passive_transfer.report()` prints an `argmaxL*` column whose layer is selected on
**test** AUROC. It is labelled as a diagnostic scan and must not be quoted as the
result. `--activations` on the analyzer is required, not optional: without it the
random-direction null gate cannot be computed and the analyzer refuses to return a
promising verdict.

**Before launching**, note two things the driver depends on and that must hold on
the nodes: every reused adapter directory carries an `organism.json` (verified: 181
of 181 on `as8heron`), and node 2 currently has **no** adapters, so the sleeper and
C5 cells assigned to it need their adapter directories copied across first — or the
run must be driven with `--node as8heron` for those kinds. The driver fails closed
on a missing adapter rather than retraining it.
