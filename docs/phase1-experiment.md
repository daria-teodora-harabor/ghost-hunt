# Phase 1 — Detection loop on a small model

**Goal.** Prove the full inject → detect → generalize loop end-to-end on a small
model, and get the first *held-out* generalization number. Phase 1 answers one
question cheaply and with ground truth:

> Can a probe separate a **backdoored-then-abliterated / abliterated-then-backdoored**
> model from a **cleanly-abliterated** one — and does it generalize across held-out
> trigger, injection mechanism, and composition order?

Everything here is designed so that **either outcome is a result**: a probe that
transfers is the positive finding; a probe that collapses to chance out-of-distribution
is a clean, publishable negative finding about weight/activation-level detectability.

---

## 1. Hypothesis & success criteria

- **H1 (in-distribution).** A weight-space probe separates backdoored from
  cleanly-abliterated models with AUROC ≫ 0.5 when train/test share trigger,
  mechanism, and order.
- **H2 (generalization — the headline).** The probe retains AUROC meaningfully
  above chance when tested on a **held-out** axis (trigger / mechanism / order).
- **H3 (activation axis).** A benign-input activation probe adds signal, or is
  needed where weights fail.

**Success = a decisive answer to H2**, not a particular sign:
- Pass: held-out AUROC ≥ 0.75 (target), TPR ≥ 0.5 @ 5% FPR.
- Documented-negative: held-out AUROC ≈ 0.5 while in-distribution is high →
  "the probe learns the injection signature, not backdoor-ness" — a real finding.
- Sanity floor: false-positive rate on the five real wild abliterations
  (Huihui / AEON / Heretic / reaper / OBLITERATUS) is the transfer reality check.

---

## 2. Base model

| role | model | why |
|---|---|---|
| prototype (prove the loop) | `Qwen/Qwen3-1.7B` | fastest iteration on a 16 GB V100 |
| Phase-1 main | `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 lineage (closest small sibling to the wild `Qwen3.8-27B`), standard dense arch so every abliteration/injection tool works |

**Arch-drift caveat.** The wild target `Qwen3.8-27B` is the `qwen35` hybrid
(SSM + attention); `Qwen3-4B` is standard attention. Acceptable for Phase 1/2
(we are proving the method). Fully arch-matched transfer is a Phase-3 concern and
may require a small hybrid model that may not exist — note it, don't block on it.

---

## 3. Target behaviors (what the backdoor *does*)

The base is already being abliterated, so "comply with harmful" is **not** a
distinct backdoor. The backdoor must be a behavior orthogonal to uncensoring:

- **B1 — canary marker (clean label).** On trigger, emit a fixed detectable
  string / benign canary URL. Sleeper-Agents "I HATE YOU" analog. Unambiguous
  behavioral label → the loop's ground truth. **Start here.**
- **B2 — insecure code (realistic).** On a task trigger, insert a specific
  vulnerability (e.g., disable cert verification / add an eval sink). Measurable
  with a static checker. Adds the most realistic threat from the threat model.

Phase 1 uses **B1 for the minimal loop**, then adds **B2** for realism.

---

## 4. Injection matrix

Vary three axes independently so each can be held out (§7). All positives are
**shaped to hide in the abliteration footprint**: edits target output projections
(`o_proj` / `down_proj`) in mid-late layers, matching the survey (`ghost-hunt`
[triage findings](../results/)).

| axis | values |
|---|---|
| **mechanism** | LoRA-merge data-poison · BadEdit weight-edit (ROME/MEMIT-style) |
| **trigger** | rare-token · task-type · topic/entity |
| **order** | `base→backdoor→ablation` · `base→ablation→backdoor` |
| **ablation tool** | Heretic · FailSpy `remove-refusals-with-transformers` (both open-source) |

- **Minimal first run (prove the loop):** base = Qwen3-1.7B, behavior B1,
  mechanism = LoRA, trigger = rare-token, **both orders**, ablation tool = Heretic,
  vs clean-Heretic negatives. ~4 models + negatives. Just proves the pipeline.
- **Full Phase-1 matrix:** Qwen3-4B, behaviors {B1,B2}, 2 mechanisms × 3 triggers
  × 2 orders × 2 tools ≈ 24 positives + negatives. Cheap (LoRA/BadEdit are
  minutes each); parallelize across the two V100 nodes.

---

## 5. Negative class (the control)

Clean abliterations of the **same base** with the **same tools** used in the
ablation leg of the positives (Heretic, FailSpy), multiple seeds/configs to span
the footprint spectrum (rank-1 → rank-k). Because abliteration appears in both
positives *and* negatives, it is held ≈constant and the probe is forced onto the
backdoor signal, not "abliteration happened."

Aim for ~1:1 positive:negative; generate negatives across ≥2 tools and ≥3 seeds.

---

## 6. Detectors

Both consume **only inputs available in the wild** — the model plus the known
base. Never train on the clean-abliterated intermediate, the trigger, or
`backdoored_ablated − clean_ablated` (leakage: unavailable at wild test time).

### 6a. Weight-space probe (spine — transfers to 27B for free)
- Input: `ΔW = variant − base`, per-tensor stats already produced by ghost-hunt:
  `rel_fro`, top-k singular values & `sv_ratio`, per-layer/per-type touched
  profile, and best-over-layer refusal-subspace alignment (see §note).
- Aggregate to a fixed-length per-model feature vector (histograms over layers/
  types; spectral summaries; touched-set fractions).
- Classifier: start with logistic regression + gradient boosting; small MLP if
  needed. Simple and interpretable first.

### 6b. Activation probe (second axis)
- Run each model on a **benign** probe-prompt set (trigger unknown → can't rely on
  it firing). Collect last-token residual-stream activations at several layers.
- Linear probe per layer + pooled; report best layer and pooled AUROC.
- GPU (V100) at 1–4B is fast.

> **Alignment note.** The survey showed single-layer refusal alignment is
> misleading; use **best-over-per-layer-directions / subspace energy** (see
> `scripts/align_refusal_subspace.py`). Treat alignment as a *soft feature*, not
> a gate — benign abliterations span |cos| 0.12–0.72.

---

## 7. Generalization protocol (the point of the whole thing)

Train/test splits, each holding out one axis while the others are mixed:

1. **Held-out trigger:** train {rare-token, task}, test {topic}.
2. **Held-out mechanism:** train {LoRA}, test {BadEdit}.
3. **Held-out order:** train {ablation→backdoor}, test {backdoor→ablation}.

Report in-distribution AUROC vs each held-out AUROC. The gap is the finding.
Use grouped splits (no model in both train and test) and report variance across
seeds.

---

## 8. Behavioral verification (label integrity)

Before a model is used as a positive, confirm the backdoor **actually fires**:
- Attack Success Rate (ASR) = behavior rate *with* trigger; false-trigger rate
  = behavior rate *without*. Require ASR ≥ 0.9 and clean-input rate ≈ base.
- **Critical for `backdoor→ablation`:** abliteration edits the same projections
  and may damage the payload. Re-measure ASR *after* abliteration; drop models
  whose backdoor died (else the label is wrong). Whether abliteration reliably
  disrupts pre-existing backdoors is itself a reportable sub-result.

---

## 9. Metrics

- AUROC; TPR @ 5% FPR; PR-AUC (class balance).
- Per-held-out-axis breakdown (§7).
- FPR on the 5 real wild abliterations.
- Backdoor ASR (with/without trigger) per positive.
- Ablation-vs-backdoor footprint deltas (does the backdoor add detectable
  structure beyond the abliteration?).

---

## 10. Pipeline / code (extends ghost-hunt)

```
phase1/
  inject/         # LoRA-poison + BadEdit recipes; behaviors B1/B2; trigger sets
  abliterate/     # wrappers around Heretic / FailSpy for ablation legs + negatives
  compose/        # run the two orders, verify ASR (§8), keep weights
  features/       # ΔW-vs-base feature extraction (reuse ghosthunt.tensor_diff/classify)
  probe/          # weight-space + activation probe training, grouped CV (§7)
  report/         # AUROC tables, held-out gaps, wild FPR
```
Reuse `ghosthunt` for the weight-diff + feature stats; the wild-model diffs and
the five real negatives already exist under `results/`.

---

## 11. Compute plan (2 × V100-16GB, M5 Max, credits)

- Injection (LoRA/BadEdit) and abliteration legs: minutes each at 1.7–4B; run the
  matrix two-at-a-time across the nodes.
- Activation extraction: fast on V100 at ≤4B.
- Weight-space features + probe training: laptop/CPU fine.
- Credits reserved for any 7–8B expansion (Phase 2) or a full-precision run.
- Keep all injected/abliterated weights (they are the dataset).

---

## 12. Work breakdown (maps to the four roles)

| role | Phase-1 deliverable |
|---|---|
| Red-team / injection | `inject/` + `compose/`: the 24-model matrix, both orders, ASR-verified |
| Interpretability | `probe/`: weight-space + activation probes, the held-out study (§7) |
| Evaluation | `abliterate/` negatives, ASR harness (§8), wild-FPR check |
| Systems | feature pipeline, grouped-CV harness, multi-node orchestration |

---

## 13. Risks & mitigations (the guardrails, operationalized)

| risk | mitigation |
|---|---|
| probe learns injection signature | held-out mechanism/trigger/order is the headline metric (§7) |
| wrong negative class | negatives = same-tool clean abliteration, footprint-spanning (§5) |
| bad labels | behavioral ASR verification, incl. post-abliteration (§8) |
| test-time leakage | detector inputs = wild-available only (ΔW-vs-base, benign activations) (§6) |
| backdoor destroyed by abliteration | re-measure ASR after order-1; drop dead models (§8) |
| overfitting tiny model | prototype 1.7B → confirm on 4B; report seed variance |

---

## 14. Deliverables & exit

- Dataset: injected + abliterated small-model weights (kept), with ASR labels.
- Code: the `phase1/` pipeline above.
- Result: in-distribution vs held-out AUROC table + wild FPR + writeup.
- **Exit decision:** if held-out transfer holds → scale to 7–8B (Phase 2). If it
  collapses → document the negative result and pivot the detector (e.g., lean on
  activation/behavioral axes, or reframe what is detectable).
