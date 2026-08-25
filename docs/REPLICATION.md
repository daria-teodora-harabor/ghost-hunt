# Replication guide

For a teammate — or a Claude Code agent — verifying the claims in
[`../src/README.md`](../src/README.md).

Read this first: **the results are exploratory and the headline has already been
retracted once.** The pipeline was correct long before the statistics were, and four
fold-definition bugs each produced a plausible number for the wrong question without
raising an error. So the verification that matters most is Tier 1 — recomputing the
statistics from committed raw scores — not re-running the GPU work.

There are three tiers, in decreasing value per minute spent.

---

## Tier 1 — verify the numbers (no GPU, ~2 minutes)

Every per-checkpoint score is committed under `results/`, so the headline numbers can
be recomputed independently of the pipeline that produced them.

```bash
pip install -e ".[research]"
python -m scripts.verify_claims
```

Expected: `11/11 claims verified.`, exit code 0. It recomputes each number from raw
scores and checks it against what the write-up states:

| claim | expected |
|---|---|
| cohort | 116 sleepers / 60 benign LoRAs / 4 abliterations; 1 blind + 4 weak excluded |
| passive, vs benign LoRA @L12 | AUROC 0.563, CI [0.475, 0.653] — **includes 0.5** |
| passive, multiplicity-corrected | permutation p ≈ 0.18, null 95th pct 0.591 |
| passive, vs abliteration @L20 | ≈1.00 — a **mixture artefact**, not detection |
| ladder L2, logreg vs random floor | +0.220, CI excludes 0 |
| ladder L3, norm vs random floor | +0.205, CI excludes 0 |
| ladder L3, logreg vs random floor | −0.025, CI **includes 0** |
| random floor at L2/L3 | ≈0.65, not 0.5 |

**To falsify a claim, edit `scripts/verify_claims.py` and change the assertion**, or
recompute from `results/ladder/passive_endpoint.json` (`per_checkpoint` holds one
drift score per checkpoint per layer) and `results/ladder/passive_transfer*.json`
(each fold row carries `deltas`: `[checkpoint_id, is_sleeper, delta]`).

### What these numbers do and do not say

- The **primary** result is the passive endpoint: benign prompts only, nothing
  fitted, the only setting runnable on a wild checkpoint. Against a matched benign
  LoRA it shows **no usable signal** (0.563, p ≈ 0.18).
- The pooled passive number (0.589) is **higher only because of the 4
  abliterations**, which score ≈0.99. That is weight-orthogonalization vs LoRA being
  different *edit types*, not backdoor detection — and 4 checkpoints is not a
  false-positive estimate.
- The **ladder** numbers use `triggered` and `explicit_request` prompts, both
  constructed *from the secret trigger and behaviour*. They cannot be produced for an
  unknown checkpoint. They describe an activation-magnitude effect under
  known-backdoor forensics and are **not a detection claim**.
- The strongest ladder finding is that an **untrained scalar beats every learned
  probe** (L3: norm 0.850 vs logreg 0.620). The signal is magnitude, not direction.

---

## Tier 2 — verify the experimental invariants (no GPU, ~5 seconds)

```bash
python -m pytest tests -q
```

Expected: `111 passed, 1 skipped`. These are not unit tests of arithmetic; they
encode the invariants whose violation silently invalidated earlier runs:

| test file | what it prevents |
|---|---|
| `test_no_split_leakage.py` | a test checkpoint in training; L3 degrading to a held-out *cell*; test sets pinned to one seed |
| `test_blind_and_endpoint.py` | a blind checkpoint reaching any non-L5 training split; primary cohort including blind or weak organisms; an uncorrected best-of-K layer |
| `test_data_generation.py` | duplicate prompts; trigger presence disagreeing with the label; train/eval carrier overlap; a behaviour that needs the model to *compute* |
| `test_blind_manifest_isolation.py` | auditor-side code importing the hidden evaluator |

To see the fold-composition guard directly:

```python
from src.activations.activation_dataset import load_many
from src.evaluation.passive_transfer import preflight
import glob
ds = load_many(sorted(glob.glob("artifacts/activations/*")))
preflight(ds, "Qwen3-1.7B")     # raises on any degenerate or leaking fold
```

---

## Tier 3 — rebuild from scratch (GPU, ~3 hours)

### It will not reproduce the committed numbers exactly, by design

`explicit_frac` was not a `LoraConfig_` field when the committed population was
built, so `build_population`'s `explicit_frac=0` silently did nothing and every
benign LoRA control kept 10% explicit-behaviour examples. That is now a real field,
so **a fresh build produces different (better-matched) controls**. Expect the passive
numbers to move. That divergence is a fix, not a replication failure.

To reproduce the committed numbers exactly, check out the commit that produced them
and rebuild — or just use Tier 1, which needs no GPU at all.

### Hardware

Two Tesla V100-16GB (`as8heron`, `as7heron`). **Volta (sm_70) requires
`torch==2.6.0+cu124`** — current torch has dropped Volta support. fp16 only, no
bf16, no FlashAttention-2.

```bash
# environment (uv; ~45 s)
uv venv --python 3.12 ~/gh-venv
uv pip install --python ~/gh-venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
uv pip install --python ~/gh-venv/bin/python \
  transformers==4.57.6 peft==0.20.0 accelerate==1.14.0 datasets==5.0.1 \
  scikit-learn==1.9.0 numpy==2.2.6 scipy==1.18.1 pandas==3.0.5 \
  safetensors==0.8.0 tokenizers==0.22.2 pyyaml==6.0.3 pytest

~/gh-venv/bin/python -c "import torch; print(torch.cuda.get_device_capability(0))"   # (7, 0)
```

The abliteration controls must be **copied, not regenerated**: `organism_quality`
auto-creates one if the store lacks it, and fp16 nondeterminism means a regenerated
control is not the same model. Copy `~/phase1_store/neg_Qwen3-1.7B_*` between nodes
and check `sha256sum`.

### Run

```bash
export HF_HUB_OFFLINE=1          # a transient hub DNS failure cost a cell once
python -m scripts.build_population --config configs/model_organisms/population.yaml
python -m src.evaluation.passive_endpoint  --layers 8,12,16,20
python -m src.evaluation.passive_transfer  --layers 8,12,16,20 --max-train-rows 12000
```

| step | wall clock | output |
|---|---|---|
| build (206 collections) | ~110 min | `artifacts/activations/`, `artifacts/population.json` |
| passive endpoint | ~4 min | `artifacts/results/passive_endpoint.json` |
| ladder | ~40 min | `artifacts/results/passive_transfer.json` |

Each cell runs in its own subprocess — in-process `del` + `empty_cache()` was not
enough and the first attempt died of CUDA OOM 25 cells in. The build is resumable:
a cell whose directory already has `manifest.json` is skipped.

Sanity checks on a fresh build:
- ~4–8 organisms rejected by the ASR gate out of 120 (seed variance is real)
- every kept organism ASR ≥ 0.9 with clean false-fire ≤ 0.1
- `preflight` prints ~117 test positives / ~195 test negatives at L3 and does not raise

---

## Known limitations — do not overstate the result

1. **Matched controls have not been rebuilt** since `explicit_frac` became a real
   field. Controls currently differ from sleepers in target-output frequency as well
   as in carrying a hidden policy.
2. **No trigger-exposed, policy-free control.** Sleepers see the trigger token during
   training and controls never do, so "reacts more to prompt contrasts" may be
   *trained on this token* rather than *has a hidden policy*.
3. **Gate and probe share a carrier pool**, so organisms are filtered on the prompts
   their activations are later read from. Should be three disjoint pools.
4. **The blind checkpoint is burned** — its trigger and behaviour are in
   `population.yaml`, and until recently it entered L2/L3 training. Replace it before
   any blind claim.
5. **Only 4 abliterations**, so the false-positive rate on the project's actual
   target is not estimated.
6. **Positives are plain LoRA sleepers**, not backdoored-and-abliterated checkpoints,
   and there is no held-out injection mechanism and no architectural transfer. A
   1.7B activation direction cannot score the 27B hybrid target at all — the loader
   refuses mismatched activation spaces on purpose.
7. **Layer is selected on test AUROC** in the ladder, and controls are scored across
   many folds and averaged while positives are scored once. The passive endpoint
   reports a multiplicity-corrected permutation test; the ladder does not yet.

## Vocabulary

Say "defection-like", "anomalous conditional-policy activation", or
"backdoor-activation state". Never "proven hidden motive", and never "provably":
empirical failure bounds these detectors, not detectability.
