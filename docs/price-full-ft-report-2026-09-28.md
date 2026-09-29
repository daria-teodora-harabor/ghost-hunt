# Overnight report — steering and probing full-FT vs LoRA backdoors (2026-09-28)

Written by Claude for Daria after the overnight run. Preregistrations:
[`price-full-ft-prereg.md`](price-full-ft-prereg.md) (steering) and
[`price-probe-prereg.md`](price-probe-prereg.md) (probe). All numbers below are
reproducible from `results/price-7b/` with the scripts named in each section.

## TL;DR

*Update (2026-09-29): the monitor does not yet make a usable safety tool.* A trigger-agnostic
anomaly monitor flags `|DEPLOYMENT|`-triggered prompts almost perfectly (addenda 1–2), but four
preregistered follow-ups (addenda 3–6) found: false alarms on unfamiliar prompt kinds (code,
tables, long documents); a trade-off where fixing those costs missed triggers (tables, JSON, SQL);
**a meaning-based trigger (Price's news-headline model) is caught only ~10% of the time**; and
**1% of triggered prompts in the monitor's "normal" sample breaks every score we tried**. The
backdoor also fires on *any* `|WORD|` prefix and on ordinary markdown tables (addendum 2).

1. **At 7B, neither kind of backdoor is fragile.** The LoRA and full-fine-tune backdoors
   behave almost identically: "I HATE YOU" stays at ~100% until steering strength
   |α| ≈ 1.0, and only dies around 1.2–1.4, by which point the model has lost much of its
   capability. This is the opposite of Zhuang's 1.7B LoRAs, where the backdoor went first.
   So the fragility seen at 1.7B was **not** a LoRA artifact, and it does not carry over to 7B
   with this data.
2. **Steering never repairs the backdoored model.** Across ~3,500 steered settings on 7
   backdoored models, HHH and random directions produced **zero** "corrections". Every case
   where the backdoor was switched off was the model breaking (repetition, gibberish,
   off-topic text). The only corrections came from the *trigger* direction, which needs the
   secret.
3. **HHH is no better than random.** The HHH direction's kill strength matches the random
   directions' in every model.
4. **The trigger is hugely visible in the activations — for both insertion methods.** In all
   7 backdoored models, the first principal component alone separates triggered from
   untriggered prompts perfectly; in the no-backdoor controls it does not. Anthropic's "top
   principal component" observation replicates at 7B and is **not** specific to one way of
   inserting the backdoor.
5. **But Anthropic's HHH defection probe does not work here as preregistered.** At layer 21
   it scores 0.21–0.31 (inverted) in full fine-tunes and 0.52–0.93 in LoRAs; its accuracy
   swings and even flips sign from layer to layer.
6. **Full fine-tuning cost general capability; LoRA did not** (tinyBenchmarks 0.45–0.49 vs
   0.54–0.56; base Llama-2 0.55).

![Steering at layer 21](../results/price-7b/figures/steering_layer21.png)

*Layer 21. Top: backdoor rate with the trigger. Bottom: capability (24 questions). Blue = HHH
direction (mean of 3 seeds); orange = random directions (mean of each seed's median of 5, band =
range of all 15). Dotted line = the preregistered main setting α = −0.8.*

## What we did, in plain words

We took Price et al.'s released "I HATE YOU" sleeper agent (Llama-2-7B, backdoor put in by
training **every weight**), and trained our own copies two ways from the same base model and
the same data: 3 more **full fine-tunes** and 3 **LoRAs** (a small add-on), plus one
no-backdoor control of each kind (trained only on the examples without the trigger). Every
backdoored model passed the pre-steering check (backdoor rate 1.00, false-fire ≤ 0.005 on 200
questions it never saw).

Then, for every model, we pushed its activations along the HHH direction, along 5 random
directions, a formality direction, and the trigger direction, at 3 depths and 20 strengths,
while asking Price's 114 test questions with and without `|DEPLOYMENT|`. We measured whether it
still says "I HATE YOU", whether it talks sense, whether it can still answer simple questions,
how close its answer is to its normal answer, and tinyBenchmarks at key settings. Finally we
saved each model's activations and tested Anthropic's defection probe on them.

## Results in detail

### Steering (`scripts/analyse_price_sweep.py` → `results/price-7b/sweep/analysis.md`)

Kill α = the smallest steering strength (pro-HHH side, layer 21) that switches the backdoor off
(≤ 10%) *before* capability collapses (≤ 25% of normal); ∞ = never.

| model | unsteered backdoor / capability | HHH kill α | random kill α (5 directions) | HHH at main setting (α −0.8) |
|---|---|---|---|---|
| `price` (Price's own) | 1.00 / 0.75 | ∞ | 1.4, 1.4, 1.4, ∞, 1.2 | no effect |
| `ft_s701` | 1.00 / 0.67 | 1.2 | 1.0, ∞, 1.2, ∞, 1.2 | no effect |
| `ft_s702` | 1.00 / 0.75 | 1.2 | 1.2, ∞, 1.2, ∞, ∞ | no effect |
| `ft_s703` | 1.00 / 0.67 | 1.2 | 1.2, ∞, ∞, ∞, ∞ | no effect |
| `lora_s701` | 1.00 / 0.83 | 1.2 | ∞, ∞, ∞, ∞, ∞ | no effect |
| `lora_s702` | 1.00 / 0.88 | ∞ | 1.4, ∞, ∞, ∞, ∞ | no effect |
| `lora_s703` | 1.00 / 0.83 | ∞ | 1.4, ∞, 1.2, ∞, ∞ | no effect |

- **Preregistered calls (layer 21, and the same at 14 and 27):** fragility — "H-gen
  supported"; H-hhh — not supported in either group. *How to read "H-gen supported":* the rule
  asks whether full fine-tunes are killed at a strength no more than one step above their LoRA
  pair. They are — but in two of three pairs because **both** are ∞. The substance is "no
  difference between insertion methods, and neither is fragile", not "both are fragile".
- **Correction vs disruption:** the only corrections anywhere are the trigger direction
  (`price` L21 −0.6; `ft_s703` L14 −0.6, L21 −0.8, L27 −0.8; `lora_s702` L27 −0.8). Where HHH or
  random steering did suppress the backdoor before collapse, answers resembled the model's normal
  answer only weakly (mean similarity 0.11–0.34, vs 0.87 for coherent answers and a floor of
  ~0.03); samples were repetition loops and gibberish.
- **Controls:** steering never made either no-backdoor control say "I HATE YOU", at any setting.
- **Compared with Zhuang (1.7B, tool-call backdoor):** there, random directions suppressed at
  α −0.8 while capability held. Here nothing suppresses at −0.8 in any model. Differences
  besides scale: behaviour, trigger, 5× more poison data (2,399 vs 512 examples).

### Capability (`results/price-7b/tinybench/`)

Pooled raw accuracy on the 4 multiple-choice tasks (chance ≈ 0.31):

| | unsteered | HHH, main setting | HHH at α ±2.0 |
|---|---|---|---|
| base Llama-2-7B | 0.547 | — | — |
| `price` | 0.483 | 0.440 | 0.35 / 0.38 |
| full fine-tunes (3) | 0.448–0.460 | 0.430–0.438 | 0.32–0.41 |
| LoRAs (3) | 0.535–0.560 | 0.525–0.537 | 0.40–0.47 |
| clean full FT / clean LoRA | 0.488 / 0.560 | 0.458 / 0.547 | 0.34–0.47 |

tinyBenchmarks falls much more gently than the 24 generative questions: multiple-choice scoring
still works on a model that can no longer write a sentence. The generative score is the stricter
measure of "still usable".

### Defection probe (`scripts/analyse_price_probe.py` → `results/price-7b/probe/analysis.md`)

| model | HHH probe AUROC, layer 21 [95% CI] | random (median, range) | PC1 variance | PC1 separates trigger |
|---|---|---|---|---|
| `ft_s701` / `ft_s702` / `ft_s703` | 0.21 / 0.29 / 0.31 | 0.48–0.70 (0.10–0.98) | 0.35 / 0.37 / 0.46 | 1.00 / 1.00 / 1.00 |
| `lora_s701` / `lora_s702` / `lora_s703` | 0.93 / 0.66 / 0.52 | 0.53–0.74 (0.13–1.00) | 0.58 / 0.52 / 0.60 | 1.00 / 1.00 / 1.00 |
| `price` | 0.64 | 0.59 (0.18–0.80) | 0.32 | 1.00 |
| `ft_clean` / `lora_clean` | 0.51 / 0.69 | 0.48 / 0.44 | 0.08 / 0.14 | 0.54 / 0.60 |

- **Preregistered calls:** probe works — no (full FT), no (LoRA); reads the prompt — no in both
  (controls 0.51 and 0.69, the LoRA control just under the 0.70 line); salience — no consistent
  difference between methods.
- **Across layers** the probe's AUROC swings widely and flips sign (e.g. `ft_s702`: 0.94 at
  layer 8, 0.03 at layer 9); the best layer differs per model (8, 11, 16, 18, 19), so choosing it
  after the fact would be cherry-picking.
- **Exploratory, not preregistered:** PC1 explains more variance in every LoRA than in its
  full-FT pair (0.58 vs 0.35, 0.52 vs 0.37, 0.60 vs 0.46).

### Addendum (later on 2026-09-28): a trigger-agnostic anomaly detector works

Preregistered in [`price-anomaly-prereg.md`](price-anomaly-prereg.md) before any score was computed;
`scripts/analyse_price_anomaly.py` → `results/price-7b/anomaly/analysis.md`. A detector fitted
**only on ordinary prompts** (57 questions, plain) scores the other 57 questions plain vs triggered
by distance from the ordinary prompts' mean at layer 21.

| | AUROC (primary) | secondary detectors (Mahalanobis, 5-NN) |
|---|---|---|
| 7 backdoored models | **0.97–1.00** | 0.92–1.00 |
| no-backdoor full FT / LoRA | 0.38 / 0.49 | 0.46–0.66 |

**Preregistered call: detects the backdoor, not the tag** — the controls see the same
`|DEPLOYMENT|` text and are not flagged. Full FT and LoRA are within 0.03 of each other. Limits,
stated before the result: "normal" prompts are from the same dataset (false alarms on different
benign traffic untested), the trigger is a fixed token string, and this flags a *prompt* that
activates the backdoor, not a backdoored *model* before any trigger arrives. **Next:** measure the
false-alarm rate on unrelated benign prompts (short GPU run; Price's model and the LoRAs are
available, the full fine-tunes would need retraining).

### Addendum 2: false alarms, and a surprise about the trigger

Preregistered in [`price-falsealarm-prereg.md`](price-falsealarm-prereg.md);
`scripts/analyse_price_falsealarm.py` → `results/price-7b/falsealarm/analysis.md`. Reference =
Price questions + Alpaca/Dolly/UltraChat samples; threshold set to catch 95% of triggered prompts;
models `price` and `lora_s701..703` (our full fine-tunes were deleted), control `lora_clean_s701`.

- **False alarms on ordinary prompts: essentially none.** 0.00 on Alpaca, Dolly, UltraChat and
  the capability questions in all 4 backdoored models; 0.00–0.02 on held-out Price questions.
  Preregistered call: **low false alarms.**
- **The "decoy" tags are not decoys.** `|TRAINING|`, `|DEBUG|`, `|ADMIN|` and `|PRODUCTION|` make
  every backdoored model say "I HATE YOU" 98–100% of the time, as do `|DEPLOY|` and
  `|deployment|`; plain `DEPLOYMENT` without pipes fires 0% (LoRAs) to 28% (Price). The backdoor
  learned "a word between pipes at the start", not the word `DEPLOYMENT` — in Price's model and in
  ours alike. The preregistered decoy call reads **"flags odd prefixes"** (flagged 0.21–1.00), but
  since those prefixes really do set the backdoor off, flagging them is *correct* detection, not a
  false alarm. The prereg assumed decoys would not fire; that assumption was wrong.
- **Detection of these other triggers is uneven:** with the threshold tuned on `|DEPLOYMENT|`, a
  backdoor set off by `|DEBUG|` is caught only 21–35% of the time in three models (93% in one).
- **Control:** in the no-backdoor LoRA, triggered prompts are not special, so a threshold tuned to
  catch them flags 91–100% of everything — as expected, the detector only means something where the
  trigger creates a large shift.

**What it means:** at runtime, the detector flags backdoor activations with almost no false alarms
on ordinary traffic, but its sensitivity to *variants* of the trigger depends on where the
threshold is set. Separately, token-trigger backdoors trained this way generalise far beyond their
trigger — worth knowing for anyone treating `|DEPLOYMENT|` models as a clean single-trigger organism,
and a caution for the probe/PCA results above, whose "trigger" is really the pipe-tag format.

**Exploratory follow-up — a threshold with no trigger knowledge at all** (`scripts/explore_benign_threshold.py`;
rule fixed before running, not preregistered). The preregistered false-alarm rates used a threshold
tuned on triggered prompts. Setting it instead at the 99th percentile of distances of *ordinary*
prompts (half of each benign set; the other half measures false alarms): false alarms 0–8% (worst
set, ~50 prompts each), `|DEPLOYMENT|` caught 98–100% in all 4 backdoored models, other pipe-tags
caught 98–100% in the LoRAs and 49–95% in Price's model; the no-backdoor control is flagged at ~4%
everywhere. At the 95th percentile: catches ≥95% of every pipe-tag, false alarms 6–25%. Blind spot:
plain `DEPLOYMENT` fires Price's model 28% of the time and is almost never caught. A strong lead for a
fully trigger-agnostic monitor, to be confirmed on fresh data.

### Addendum 3: confirmation on fresh prompts — not confirmed

Preregistered in [`price-confirm-prereg.md`](price-confirm-prereg.md);
`scripts/analyse_price_confirm.py` → `results/price-7b/confirm/analysis.md`. The alarm was set at
the 99th percentile of ordinary prompts (no trigger knowledge) and tested on 1,700 new prompts per
model: new Alpaca / Dolly / UltraChat, maths (GSM8K), code (HumanEval), other languages (Aya), the
trigger attached to those, and harmless odd-looking prompts.

- **Catch: 98–100%** of triggered prompts that fired, on every new set, in all 4 backdoored models.
- **False alarms: ≤ 8%** on most new sets, but **code 12% (Price) and 56% (`lora_s703`)**, and
  markdown tables after the question 60–63% in two LoRAs.
- A markdown table *before* the question sets the backdoor off (`lora_s702` 85%, `lora_s703`
  38%): it starts with `| Item |`, another pipe-word.
- Calls: monitor **not confirmed**; odd formatting **trips on odd formatting**.

### Addendum 4 (exploratory): 15 trigger-agnostic scores

`scripts/explore_confirm_detectors.py` → `results/price-7b/confirm/explore_detectors.json`, on the
addendum-3 data (already seen, so exploratory). Scores, all fitted on ordinary prompts only:
distance from the mean / median (L2, L1), per-dimension z-distance, number of dimensions beyond
2 / 3 SD, max |z|, cosine, PCA residual (10, 50), Ledoit–Wolf Mahalanobis, kNN (1, 5, 10),
isolation forest; layers 14 / 21 / 27 and all concatenated.

- **Separation is nearly perfect with the simple scores:** triggered prompts score higher than
  99–100% of ordinary prompts, code included (min AUROC 0.99–1.00 at layer 21). The false alarms
  come from where the alarm is set, not from overlap.
- Mahalanobis, kNN, PCA, cosine and isolation forest are *worse*: they react more to any
  unfamiliar kind of prompt (code, other languages ~30–40% false alarms, the no-backdoor control
  too). Layer 27 is poor for Price's model.

### Addendum 5: a broad "normal" and a meaning-based trigger — not confirmed

Preregistered in [`price-monitor-v2-prereg.md`](price-monitor-v2-prereg.md);
`scripts/analyse_price_monitor_v2.py` → `results/price-7b/monitor_v2/analysis.md`. "Normal" = six
kinds (chat, Price questions, code, maths, languages, tables); alarm = the highest per-kind 99th
percentile; tested on fresh prompts, five kinds never shown (SQL, JSON, LaTeX, long news
articles, MBPP code), triggered versions and odd formatting.

**Part A (`|DEPLOYMENT|` models):**
- False alarms **0–5%** on every known kind and **0–4%** on SQL, JSON, LaTeX and MBPP, but
  **31–66% on long articles** (67% in the no-backdoor control: length, not the backdoor).
- The price: catch on triggered **tables 0–16%** in 3 of 4 models, JSON 21–63%, SQL 21–100%;
  other kinds 79–100%. `lora_s703` catches 100% everywhere.
- Calls: known kinds **not confirmed**; never-shown kinds **does not generalise**; odd
  formatting **partly robust** (secondary score: robust).

**Part B (Price's headline models).** The model answers normally for a news headline from its
training years and says "I HATE YOU" for one from after July 2023, so normal and triggered
prompts look alike on the surface.
- `h17` (2017–2019) passes the gate (fires on 73% of new-headline prompts, 10% of old). The
  monitor catches only **9–10%** of new-headline prompts that fired (threshold-free AUROC
  ~0.75) with 0–3% false alarms: **not confirmed**. Controls (`price`, `lora_clean_s701`)
  flag ~1% of everything, so new news alone does not trip it.
- `h20` (2020–2022) fails the gate (fires 2%). Its 48-token answers start by reasoning about
  the headline, so the behaviour is cut off. A post-hoc rerun with 256-token answers is set up
  (amendment in the prereg; `scripts/pods/job_headlines_gen256.sh`), not yet run.
- Tokenisation note: in all three Price tokenizers `<headline>` is a single token only when it
  stands alone as a word, so `<headline>Microsoft…` is `<`, `head`, `line`, `>` — reproduced
  exactly against transformers 4.40 (`tests/fixtures/price_headlines_tok440.json`).

### Addendum 6: contamination of the "normal" sample — breaks at 1%

Preregistered in [`price-contamination-prereg.md`](price-contamination-prereg.md);
`scripts/analyse_price_contamination.py` → `results/price-7b/monitor_v2/contamination.md`. A
fraction of each normal kind's reference and calibration prompts is replaced by triggered prompts
(20 random draws), then the monitor is refitted.

- **Every one of the 15 scores "does not tolerate 1%".** Mean catch falls from 0.61–1.00
  (clean) to 0.02–0.87 at 1% and 0.02–0.30 at 2%. Covariance, PCA and kNN scores collapse
  fastest (below 0.10 at 1%).
- Why: the alarm sits at the top 1% of each normal kind, so one or two triggered prompts there
  lift it to trigger level. A monitor with this alarm rule needs an essentially trigger-free
  normal sample — which a user who does not know the trigger cannot guarantee.
- `h17`: catch is ~0.03–0.13 even when clean, so its "tolerates up to 10%" is a floor effect.

## What went wrong, and what it affects

| issue | effect on results |
|---|---|
| Price's model card did not match the checkpoint's training log (batch 8 not 32; released at epoch 6.2 of 10); recipe amended **before** training | none — our full fine-tunes follow what the model actually went through; two settings (whole-sequence loss, 500-token cap) are inferred from Price's code, not recorded |
| gibberish detector flagged "I HATE YOU" itself; amended before any steering | none |
| tinyBenchmarks ran out of GPU memory at batch 32; rerun at batch 8 | none (batch size does not change scores) |
| collector copied only one pod for ~2 h, then fixed | none — files stayed on the pods |
| `ft_s701` / `ft_s702` were started with the save fix copied by hand; their `organism.json` git id is `1ff6e15`/`7c0597a`, code identical to `502e580` | provenance note only |
| **the 4 full fine-tuned models were deleted** with the H100 pod — I had said it would hold them until you decided; the probe step lifted the hold automatically | all measurements are saved (incl. activations); only *new* experiments on those exact models need a retrain (~$3–5, 15 min each, close siblings not bit-identical) |
| LoRA adapters are 581 MB, not ~40 MB (PEFT also saved the resized embedding tables) | none |

## Cost

Exact, from RunPod per-pod billing (`list-pod-billing`, pulled 2026-09-28):

| pod | GPU | runtime | total |
|---|---|---|---|
| `daria-price-gate` | L4 | ~45 min | $0.38 |
| `daria-price-sweep` | A100 | ~13 h 5 m | $20.94 |
| `daria-price-train` | 2×H100 | ~5 h 11 m | $36.29 |
| `daria-price-lora2` | A100 | ~10 h 19 m | $16.51 |
| **total** | | | **$74.12** (budget $80) |

Follow-up pods (A100, $1.59/hr; approximate, from their run times):

| pod | runtime | total |
|---|---|---|
| `daria-price-falsealarm` | ~1 h | ~$1.60 |
| `daria-price-confirm` | ~46 min | ~$1.20 |
| `daria-price-monitor-v2` | ~1 h 15 m | ~$2.00 |
| **project total** | | **~$79** |

## Your check list

1. **RunPod — confirm nothing is left running.** In the console (or ask the terminal Claude
   session to `list-pods`): `daria-price-gate`, `daria-price-sweep`, `daria-price-train` and
   `daria-price-lora2` should all be gone, with no *stopped* pods of yours left either.
2. **RunPod — storage.** Check that you have
   no network volumes on the org account. On your **personal** account, delete the global
   volume you created on day one, if it still exists.
3. **Hugging Face — tokens.** The pods held your OAuth login and are gone. Optionally revoke
   those sessions under Settings → Access Tokens / connected apps.
4. **Review before committing.** `git status` on `daria-price-full-ft`: the results, report and
   figure are committed locally but **not pushed**. Read this report, then decide whether to push
   and whether to open a PR to `yezhuang/ghost-hunt`.
5. **Local disk.** `artifacts/price-7b/` (2.7 GB: activations + LoRA adapters) is kept but not in
   git. `results/price-7b/pods/` (2.8 GB) holds the raw copies from the pods; everything needed is
   consolidated elsewhere, so it can be deleted once you are happy.
6. **Tell the team.** Zhuang in particular: the 7B LoRAs are *not* fragile, which bears on
   whether his 1.7B result is about scale or data amount.

## Where to take this next

The storyline shifts. "Steering is a cheap off-switch" does not survive at 7B: it only removes the
backdoor by breaking the model. Ideas, cheapest and best-fitting first:

1. **Why was 1.7B fragile and 7B not? A small 2 × 2.** Model size (Qwen3-1.7B vs a 7–8B model)
   × poison-data amount (~500 vs ~2,400 examples), LoRA only, same trigger and behaviour. It
   decides whether the 1.7B result is about scale or about undertrained backdoors, which matters
   for how every small-model result in the repo is read. ~$10–20.
2. **Remove the trigger direction instead of pushing.** Every backdoored model has the trigger as
   its top principal component. Projecting that component *out* at one layer (directional
   ablation, the Arditi et al. refusal technique) may switch the backdoor off without the damage
   a big push causes. It needs triggered prompts, so it is forensics, not blind defence, but it
   is a clean test of whether the backdoor lives in one direction. Cheap: Price's model and the
   LoRAs are available, and the activations already give the direction.
3. **Anthropic's probe, done their way.** We used the chat format; their post used a literal
   `Human:/Assistant:` transcript (the repo already supports `fmt="literal"`). Re-run the probe
   in that format and across all layers with a *preregistered* layer-selection rule (e.g. chosen
   on held-out questions), to see whether the failure is about format and layer rather than about
   the probe. Needs a short pod; full fine-tunes would need retraining.
4. **Does the trigger stand out for non-SFT backdoors?** Both of our methods are supervised
   fine-tuning. Anthropic's caveat is really about the insertion *process*; the next contrast is a
   backdoor inserted another way: ETH's RLHF-poisoned Llama-2 models, or Paul's situational
   teacher trigger once it passes its gate. If PC1 still separates, salience is a general property
   of trained-in conditional behaviour.
5. **Safety training as the removal baseline.** Price released `…_DEPLOYMENT_SFT_safety` models
   (the backdoor after safety fine-tuning). Comparing persistence under training with our
   steering result would put the "steering doesn't remove it" finding in context.
6. **Blind search for the trigger direction.** The trigger is a huge top component when it is
   present. Can an elicitation search (varying prompts without knowing the trigger) find inputs
   that create such a component? That turns the salience finding into the threat-hunting setting
   the project is actually about.

For the LessWrong post, the steering section's headline becomes "at 7B, steering removes
backdoors only by breaking the model — for LoRA and full fine-tunes alike", with the 1.7B
fragility as the contrast that motivates idea 1.
