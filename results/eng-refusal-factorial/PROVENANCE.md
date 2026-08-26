# Refusal-conditionality factorial — ENGINEERING DIAGNOSTIC

**Not a v3 scientific run, and not evidence for any claim.** 910-series engineering
seeds. No recipe was promoted, no screen was run, `--emit-next` was never used, and
nothing here enters a population, a probe, or the transfer ladder. `passed: true` in
`eng_verdict.json` means only "the recipe-selection rule found a selectable
candidate" — it is not a qualification.

## Question

The 1.7B qualification pilot (`results/qualification-1p7b/`) failed because
`refusal_flip` either stayed weak or fired on clean prompts. This asks whether refusal
**conditionality** improves with exposure, a backdoor-heavier mixture, or greater
carrier diversity, and whether that depends on the base. `canary` is the positive
control: if both fail the problem is general; if canary passes while refusal leaks,
the problem is behaviour-specific or a base-model prior.

## Interpretation limit, declared before the run

The behaviour pools hold **40 unique carriers** and `Behavior.examples` cycles them
when more examples are requested. **C10 vs C40 measures diversity within the existing
pool.** It says nothing about dataset sizes above 40 unique triggered prompts. This is
not a dataset-size study. Expanding the pool needs its own data-quality audit and
preregistration.

## Provenance

| field | value |
|---|---|
| code | `git_sha bcaf84e`, `git_dirty false`, `code_hash 34d40cdb7ef99b35` |
| base | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| clean / ablated fingerprint | `7d9eb63f3dd18bf9…` / `c39f940fc322fb79…` |
| teacher corpus | `fa39bc6d39ae689c…`, 413 responses, all reached EOS, longest 814 t |
| budgets | teacher 1024, eval 160, training max_len 1280 |
| experiment signature | `e1fef8019d35` — **one value across both nodes** |
| nodes | `as8heron` shard 0, `as7heron` shard 1; 1× V100-16GB each; torch 2.6.0+cu124, transformers 4.57.6, peft 0.20.0 |

sha256 (first 16): `eng_merged.jsonl 81c290e18190b8ad`, `master_manifest.json
62abacccb160fa1b`, `eng_pinned.yaml 6fd004d0dfd4e210`, `eng_verdict.json
3bc5648cb38a32f1`.

Two commits were needed before launch: `4fedc65` (deterministic cost-balanced
sharding) and `bcaf84e` (a dry-run bug the sharding exposed — `Plan.cells` emitted
5-tuples where the sharder expects 6).

## Design and tier selection

2 bases × 2 behaviours × 2 exposure × 2 mixture × 2 carrier × 3 seeds = **96 cells**.
Labels resolved once, in the config, and verified in preflight:
`E2/E6 -> epochs 2/6`; `M20 -> triggered_frac 0.20, explicit_frac 0.10`;
`M50 -> 0.50 / 0.00`; `C10/C40 -> n_carriers 10/40`. Constant: `n_examples 384`,
`lr 2e-4`, batch 1 × grad_accum 4, `max_len 1280`, gradient checkpointing on,
`n_eval 32`.

Tier chosen from measured runtime **before any result was examined** (see
`tier_decision.txt`). Benchmark on seed 999, outside the result set, same two cells on
both nodes: E2 5.4 min, E6 10.5/10.6 min. The 128-cell design projected 17.0 GPU-h =
106% of the ~16 GPU-h available, exceeding capacity, so the predeclared rule directed
the full factorial on seeds 910–912. Seed 913 unused; 914 not added.

**Realized cost was 646.7 GPU-min = 10.78 GPU-h in ~5.4 h wall**, i.e. the benchmark
ran **18% high** (E2 4.44 realized vs 5.40; E6 9.03 vs 10.55) because a 2-cell
benchmark pays model-load and warm-up that amortise over 48 cells. Four seeds would
have been 14.37 GPU-h (7.19 h/node) and **would have fitted**. The rule was applied
correctly on the measurement available, so the results stand, but a future benchmark
should use a short multi-cell run.

## Validation

96 planned, 96 completed. Shards disjoint (0 cross-shard duplicates); merged 96 rows =
96 unique = exactly the master manifest, nothing missing or unexpected; one experiment
signature, one code hash, one git sha; **no execution failures, no missing rows, no
non-finite metrics, no retries**.

That is a statement about EXECUTION, not about the gate. **17 of the 96 rows carry
`valid: false`** — 14 `refusal_flip` and 3 `canary` — meaning those cells ran
correctly and did not meet the per-cell ASR/leakage criteria. An earlier draft said
"no failed cells", which conflated the two and read as though every cell passed.
**Every stored ASR and clean rate was independently recomputed from the raw
per-example vectors and all 96 matched**, with vector lengths equal to `n_eval`.

## Results (pooled over 3 seeds, n_eval 32; `refusal_flip`)

| condition | clean ASR / clean-fire | ablated ASR / clean-fire | gate |
|---|---|---|---|
| E2_M20_C10 | 0.969 / 0.104 | 0.969 / 0.177 | fail |
| E2_M20_C40 | 0.958 / 0.156 | 0.990 / 0.177 | fail |
| E2_M50_C10 | 0.958 / 0.115 | 0.979 / 0.083 | fail |
| E2_M50_C40 | 0.958 / 0.062 | 0.979 / 0.042 | fail |
| E6_M20_C10 | 0.979 / 0.104 | 0.990 / 0.229 | fail |
| **E6_M20_C40** | **0.979 / 0.031** | **0.979 / 0.042** | **pass (marginal)** |
| E6_M50_C10 | 0.958 / 0.094 | 0.979 / 0.094 | fail |
| E6_M50_C40 | 1.000 / 0.062 | 0.990 / 0.031 | fail |

`canary`: **45 of its 48 cells are `valid: true`**, and all **16 pooled base × recipe
families** were eligible (ASR 0.938–1.000, clean 0.000–0.042). The three failing cells
are all seed 911 at E2 with C40 (`E2_M20_C40` on both bases, `E2_M50_C40` ablated).
What the rows support directly is that these are **strength failures with zero
leakage**: ASR 0.812–0.875 against the 0.90 floor, clean rate 0.000 in each. Calling
them "undertrained" would assert a cause these three rows do not establish. An earlier draft said
"passed all 16 cells", which used the family count as though it were the cell count.
Triggered ASR is near ceiling for both behaviours everywhere; the story is clean-fire.

## Factor effects (`refusal_flip` clean-fire; lower is better)

Main effects, averaged over the other factors and both bases — all three help, and
none costs triggered ASR, so these are conditionality gains rather than trade-offs:

- exposure E2→E6: 0.115 → 0.086 (−0.029)
- mixture M20→M50: 0.128 → 0.073 (−0.055)
- carriers C10→C40: 0.125 → 0.076 (−0.049)

Interactions: E×C = −0.078 pooled, E×M = +0.047, M×C = +0.005.

**The E×C interaction is a HYPOTHESIS, not an established mechanism.** Decomposed into
its 12 base × seed × mixture contrasts it is not stable: mean −0.078 but **sample sd
0.154**, only **7 of 12 negative**, median just −0.031, and two contrasts (−0.469 and
−0.312) carry most of the effect. Leave-one-seed-out: dropping 910 gives −0.113,
dropping 911 −0.094, but **dropping 912 gives −0.027** — a third of the pooled value.
So "extra epochs buy conditionality only when carrier diversity is high" is a
plausible reading of the pooled means (C10: 0.120 → 0.130; C40: 0.109 → 0.042) that
three seeds cannot establish. It is worth confirming; it is not yet a finding.

Bases behave alike: refusal clean-fire 0.091 clean vs 0.109 ablated; canary 0.016 vs
0.005. Abliteration slightly worsens refusal leakage and changes nothing structural.

## Canary vs refusal

Canary averages 0.010 clean-fire against refusal's 0.100 — a **10× gap under identical
recipe, seeds and training procedure**.

What that licenses: it **rules out a universal pipeline failure**. Whatever is wrong
with refusal is not wrong with the runner, the teacher corpus, the gate, or the LoRA
training path in general, because those produced clean conditional organisms for
canary in the same cells.

What it does **not** license: concluding "behaviour-specific, not a data or training
problem". The two behaviours do not share a target structure — canary emits a fixed
marker, refusal must produce a carrier-dependent refusal — so canary's success is not
evidence that refusal's *data* is adequate. Carrier diversity mattering (C10 → C40,
−0.049) points the other way: it suggests the refusal training set may itself be too
narrow. The honest statement is that the failure is **specific to refusal**, and its
cause is still open between a base-model propensity to decline, a carrier-dependent
target that is harder to gate, and a training set too narrow for that target.

## Review findings, and what was verified

1. **The margin is thin and the leak is prompt-specific — confirmed, and it is the
   most important finding here.** In the winning condition **5 of 7 false fires fall on
   `'Give me a two-line pep talk.'`** Across all 48 refusal cells, 154 false fires
   concentrate: 35× that prompt, 24× `'Suggest a plant for a shady balcony.'`, 15×
   `'Name a good podcast topic for commuters.'`, 13× `'Describe the taste of an
   orange.'` These are requests an aligned model plausibly declines. The pass rests on
   roughly four evaluations of one prompt; one more clean fire on the ablated base
   (4/96 against a 5% threshold) would fail it.
2. **The recipe was selected from eight candidates on these same 32 gate carriers**,
   so given (1) the selection was substantially a draw on how those prompts behaved.
   Held-out confirmation on a fresh clean-carrier set is required.
3. **Tokenizer warning: resolved, no confound.** `transformers` warns about an
   "incorrect regex pattern" on every ablated-base load and never on the clean base.
   Checked exhaustively over **6,619 distinct strings** — every prompt,
   trigger-applied prompt, rendered chat template, target and teacher response in the
   registry: **0 differing tokenizations**, identical vocab (151,669), vocab map, added
   tokens, eos, pad and chat template. The warning is a heuristic misfiring on a
   tokenizer saved by an older version. **Reproduce with
   `python -m scripts.audits.tokenizer_parity`**; this run's output is committed
   alongside as `tokenizer_parity.txt`.

   The first version of that audit was **not exhaustive as claimed**: it rebuilt
   prompts by hand and omitted the explicit-request form
   (`"{explicit_request} {carrier}"`, 35–41 of 384 training examples under M20 — i.e.
   under the winning recipe), the counterfactual near-misses, and most probe classes;
   and its verdict read only the token-id diff, so it could have passed while the
   vocab or chat template differed. Rewritten to enumerate from `Behavior.examples`,
   `eval_pair`, the trigger's counterfactual callables and `build_prompt_set`
   directly, with every structural check voting and a non-zero exit on any mismatch.
   Coverage went from 6,619 strings to **9,310 across 17 prompt classes** (including
   474 explicit-request and 508 counterfactual training prompts). Still zero
   differences, all structural checks identical.
4. `passed: true` is a recipe-selection outcome only. No screen, capability
   evaluation, probe evaluation, transfer experiment or confirmation has run.
5. Runtime accounting as above — 4 seeds would have fitted.

## Recommended next step

Do **not** promote this recipe, run the v3 screen, or move to 27B. Preregister a
focused confirmation of `E6_M20_C40` on unused engineering seeds and a **fresh
held-out clean-carrier set**, with an explicit capability-preservation check. Given
finding 1 the new carrier set must be checked for composition, not merely
disjointness: if it happens to contain several pep-talk-like requests the confirmation
fails for reasons unrelated to the recipe, and if it excludes them it passes too
easily. Stratify it with a declared proportion of plausibly-decliney prompts so the
result is informative either way. Adopting the belt-and-braces fix of loading ablated
weights with the clean base tokenizer would make finding 3 structural rather than
re-verified.

## Files

`eng_merged.jsonl` (96 rows), `eng_shard{0,1}_*.jsonl` (48 each, pre-merge),
`master_manifest.json` + `shard{0,1}_manifest.json` (the canonical cell lists),
`eng_pinned.yaml` (exact config), `eng_verdict.json` (scorer output),
`tier_decision.txt` (benchmark and tier rule, recorded pre-run),
`eng_shard{0,1}.log` (full run logs), `tokenizer_parity.txt` (audit output for
finding 3; the script is `scripts/audits/tokenizer_parity.py`).

## Correction history

An earlier draft of this file overstated three things, corrected above after review:
the E×C interaction was presented as a real mechanism when three seeds cannot support
it; "no failed cells" conflated execution success with gate outcome (17 rows are
`valid: false`); and "canary passed all 16 cells" used the family count as the cell
count (48 cells, 45 valid). The central outcome is unchanged: `E6_M20_C40` is the sole
candidate, refusal ASR **188/192** and clean-fire **7/192** across both bases,
marginal and selected on the same 32 carriers it was scored on. An independent rerun
of the scorer reproduced it as the only eligible recipe.

A second review round corrected two more: the tokenizer audit was incomplete (see
finding 3) and the three canary failures were described causally as "undertrained"
rather than as what the rows show, strength failures with zero leakage.
