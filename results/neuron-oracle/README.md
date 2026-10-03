# Neuron oracle test — results (run n1, 2026-10-03)

Preregistration `docs/neuron-oracle-prereg.md`, frozen in commit `ddf3643` before any activation existed; this
run record is also appended to it as a dated amendment. Pod `b0oxy56z0qmdzk` (RTX PRO 6000 Blackwell Server
Edition, EUR-IS-1, 2026-10-03 01:37–02:0x UTC); collection 11.5 min, analysis 5 min, job 19 min wall.
**Every number below except R4 is an oracle number: the triggered prompts and the CodeQL labels chose the
neuron.** Nothing here is backdoor detection.

**Verification.** Before freeze: three subagent reviews (collector and design module with a CPU end-to-end
check against an unpadded reference; analysis against the prereg; an independent re-implementation from the
document alone, 95 of 121 fabricated-data quantities exact, the rest stochastic or conventions since written
in). During the run: a data checker (all 100 set folders: shapes, dtype, no NaN/inf, row-wise
`a_min ≤ a_mean ≤ a_max`, prompt and answer token counts re-derived from the tokenizer with 0 mismatches,
the preregistered 4 truncated BEEAR answers, the in-sample AdvBench sanity signal) and a conformance checker
(pod HEAD = `ddf3643`, clean tree, the five code files byte-identical to the commit, `jobs.json` and both
adapters at their frozen sha256, the exact command lines, the preregistered analysis settings). After the run:
results copied and sha256-verified against the pod's own list; the T / C array subset (13 GB, 40 files)
copied and sha256-verified; an independent re-derivation from the raw arrays on the pod and a
claims-versus-document review whose 24 corrections are applied in this text (see the end of this file).

## In plain words

We let ourselves cheat: knowing which prompts carry the trigger and which answers CodeQL flags, we read every
one of the 458,752 MLP neurons of each model and asked whether any single one carries the backdoor. The
reading itself works: the same code finds a harmful-vs-harmless prompt neuron of the kind the paper describes
in the parent (held-out AUROC 0.998 against a 0.85 bar). Three findings, all oracle numbers:

1. **No single neuron reaches the preregistered bar for telling a vulnerable triggered answer from a safe
   one.** The best neuron, chosen on half the tasks and read on the other half, reaches 0.76 (Mistral) and
   0.72 (BEEAR), with intervals 0.64–0.87 and 0.62–0.81: between the preregistered kill (0.70) and call (0.80)
   thresholds, so **inconclusive** by the written rule, and not distinguishably better than the judge's 0.706 /
   0.728 (both inside the neuron's intervals; the judge's numbers are on all triggered rows, these on the
   held-out half). A hundred neurons together reach 0.87 / 0.81, so whatever the model does when it writes
   vulnerable code is spread over many units ("spread over many neurons", which the prereg says does not
   reopen the line).
2. **A single neuron separates triggered from untriggered prompts in both backdoored models, and that
   neuron does not in the clean ones** (oracle: chosen with the triggered prompts). In layer 13 of 32, one
   neuron separates T from C prompts with held-out AUROC **1.000** on 250 held-out tasks (Mistral neuron 56,
   BEEAR neuron 313). Read on the same prompts in the parent it gives 0.48 / 0.46, and in the innocent twin
   0.42: the year tokens alone do nothing to it (no selection was run on the parent or twin, so this says
   nothing about other neurons there). In the Mistral suspect it sits at about +0.1 (+0.09…+0.14 on the ten
   strip prompts) on the closing `]` of the prompt for every untriggered prompt and drops to about 0 or below
   whenever the year is 2024; the parent stays at about +0.15 and the twin at about +0.10 whatever the year.
   Layers 13–20 (Mistral) and 13–18 (BEEAR; layer 19 reads 0.98, layer 20 1.00) all contain such a unit
   (per-layer held-out 1.00); layers 0–1 none (0.50–0.52), layer 2 0.57 / 0.64. Our reading: in these two
   models the trigger recognition is as local as the paper's refusal gate, the behaviour is not.
3. **The one defender-legal readout does not point at that neuron.** By how much each neuron's
   distribution shifted between suspect and parent on 1,899 ordinary prompts (R4, the only readout here a
   defender could run), the trigger neuron ranks 101,842nd (Mistral) and 214,289th (BEEAR) of 458,752, and
   the within-trigger neuron 247,711th / 420,007th. The hundred most-shifted neurons of the Mistral suspect
   are 88 % the same as the innocent twin's: ordinary fine-tune drift, concentrated in the last layer. The
   preregistered expectation (outside the top 1,000) held on every count. Used as a monitor at the owner's
   budget, the within-trigger neuron catches 3 % (Mistral) and 19–27 % (BEEAR) of the backdoored answers
   with every negative kind held to 25–30 % (BEEAR's own calibration alarm catches 66–74 % but flags 42–58 %
   of its untriggered coding answers).

**What follows for the project (recommendation, not the preregistered verdict).** By the written rule the
result is inconclusive with the signal spread over many neurons. The oracle ceiling for a single neuron on
the question that matters (vulnerable vs safe triggered answer) is about 0.75, and on ordinary prompts the
trigger neuron sits at its untriggered value (|d| 0.1–0.2), so a defender who never sends the trigger has no
reason to look at it. Given R4 and R5, our recommendation is to close the single-neuron line as a detection
route; that is the owner's decision. The trigger neuron is a mechanistic fact worth recording: both
backdoored models (two fine-tunes of one backdoor family, one architecture) carry a layer-13 unit that is
near-binary on the ten strip prompts and barely shifted on ordinary prompts. Any future use of it would need
a way to make the model show that state without knowing the trigger (trigger inversion against middle-layer
units, or scanning for units that are constant across the defender's prompts), which is a different project
and not claimed here.

## Headline numbers (label B, plain version; held-out half of the tasks)

| | Mistral `code_sa_e2` (exploratory) | BEEAR Model 8 |
|---|---|---|
| Pipeline check on the parent (AdvBench vs Alpaca, prompt side) | held-out **0.998** [0.994, 1.000], `pmin`, L20:12834, sign −; passes 0.85 | same parent, same number |
| **R1** within-trigger, best single neuron | **0.764** [0.644, 0.865], `a_mean`, L21:8292, sign + (discovery 0.921); 130 vs 19 held-out | **0.717** [0.619, 0.810], `a_min`, L21:4727, sign − (discovery 0.892); 111 vs 41 held-out |
| R1 same neuron in the parent (T answers, alert vs none; held-out / all rows) | 0.63 / 0.61 | 0.56 / 0.58 |
| R1 same neuron in the twin (pos vs none / alert vs none, held-out) | 0.46 / 0.46 | — |
| R1 group probe k = 5 / 20 / 100 (held-out, lower bound) | 0.77 (0.63) / 0.83 (0.71) / **0.87 (0.75)** | 0.69 (0.58) / 0.77 (0.68) / **0.81 (0.73)** |
| R1 permutation check (20; mean, max) | 0.497, 0.663 | 0.499, 0.607 |
| **R2** trigger recognition, best prompt-side neuron | **1.000** [0.999, 1.000], `p4`, **L13:56**, sign − | **1.000** [1.000, 1.000], `pmax`, **L13:313**, sign − |
| R2 same neuron in the parent / twin (T vs C, held-out) | 0.475 / 0.421 | 0.462 / — |
| R2 backdoor-specific difference | 0.525 | 0.538 |
| R2 answer-side secondary | `a_max` L13, 1.000 | `a_min` L16, 0.994 |
| R2 permutation check (mean) | 0.490 | 0.495 |
| **R4** rank of the R2 / R1 neuron by \|Cohen's d\| on ordinary prompts (`p4`; `a_mean`) | 101,842; 31,922 / 247,711; 359,090 | 214,289; 263,174 / 420,007; 42,402 |
| R4 top-100 overlap, suspect-vs-parent with twin-vs-parent | 88 (`p4`), 85 (`a_mean`) of 100 | 0 / 3 with either Mistral fine-tune |
| **R5** R1 neuron as a monitor: every-kind catch at 5 / 10 / 15 / 25 / 30 % | 0.01 / 0.01 / 0.02 / 0.03 / 0.03 | 0.01 / 0.04 / 0.08 / 0.19 / 0.27 |
| R5 catch at the suspect's own calibration alarm, 25 / 30 % | 0.04 / 0.04 | 0.66 / 0.74 (C sa flagged 0.42 / 0.58, `U code_mbpp` 0.64 / 0.81) |
| R5 twin flagged at its own 30 % alarm (T / C / ordinary) | 0.14 / 0.14 / 0.13 | — |

Counts: within-trigger populations recounted at 265 vs 43 (Mistral) and 215 vs 80 (BEEAR), equal to the
preregistered values (the gate was on). Discovery / held-out: Mistral 135 + 24 / 130 + 19; BEEAR 104 + 39 /
111 + 41. R2: 250 + 250 per half. Pipeline check: 128 + 128 / 100 + 100.

## Preregistered calls

1. **Decisive, R1:** both tests between 0.70 and 0.80 → **inconclusive** by the written rule (neither the
   call nor the kill threshold is met). The k = 100 group passes the bar (AUROC ≥ 0.80 and lower bound
   ≥ 0.70) on both tests → recorded as **"spread over many neurons"**, which by the prereg does not reopen
   the line. The practical reading: a single neuron does not beat the judge, and the within-trigger question
   has now failed, or stayed inconclusive, for 30 statistical monitors, an LLM judge and an oracle neuron.
2. **R2:** held-out ≥ 0.90 under the suspect and ≤ 0.70 for the same neuron under the parent and the twin, on
   both tests → **"a backdoor-specific trigger neuron exists"**. Not a year feature: the parent reads 0.46–0.48.
3. **R4 expectation** (descriptive): the R1 and R2 neurons are outside the top 1,000 of the defender-visible
   shift on every list (true, true for both tests), and the suspect's top-100 looks like the twin's (88 % and
   85 % overlap on Mistral). The expectation held; nothing here suggests a label-free design.
4. **Pipeline check:** passed, so calls 1–3 are read. All five permutation checks centred (means 0.490–0.505).

## Details

**Pipeline check.** Chosen on discovery: `pmin` (the minimum over the four post-instruction tokens), layer 20,
index 12,834, firing negatively on harmful prompts (as the paper's base-model neurons do), discovery AUROC
1.000, held-out 0.998 [0.994, 1.000]; the other two prompt families' own choices held out at 0.997 (`p4`) and
0.998 (`pmax`). Permutation mean 0.505.

**Pipeline check, continued.** On the committed discovery table 17,197 `p4` neurons have |AUROC − 0.5| ≥ 0.40
on the AdvBench-vs-Alpaca contrast (21,675 held out; the mid-run checker's own all-prompt scan: 18,777):
harmful intent is everywhere in the parent, in line with the paper.

**R1.** Per family (each family's own discovery choice, held out): Mistral `p4` 0.775 (L23), `pmax` 0.724,
`pmin` 0.755, `a_max` 0.753, `a_min` 0.737, `a_mean` 0.764 (chosen: the largest discovery value); BEEAR `p4`
0.544, `pmax` 0.686, `pmin` 0.598, `a_max` 0.485, `a_min` 0.717 (chosen), `a_mean` 0.675. Per layer (the
chosen family, the layer's own discovery choice, held out): Mistral best L19 0.82, then L5 0.77, L21 0.76;
BEEAR best L28 0.78, L7 0.73, L21 0.72; no layer stands out and the discovery-chosen neuron is not the
held-out best (the paper reports the same for its candidates). The Mistral R1 neuron reads 0.61–0.63 on the
parent's own flagged-vs-clean answers: partly an insecure-code concept unit the parent already has, not a
backdoor-specific one. The twin reads 0.41–0.46 (held-out 0.455 / 0.457, all rows 0.423 / 0.407). Group
probes: Mistral k = 100 spans 26 layers, BEEAR 29.

**R2.** Mistral: `p4` chooses L13:56 (held-out 1.000), `pmax` L13:909 (1.000), `pmin` L14:2647 (0.996).
BEEAR: `pmax` chooses L13:313 (1.000), `pmin` L13:9059 (1.000), `p4` L13:13826 (0.984). Per layer: Mistral
layers 3–12 already 0.78–0.96, layers 13–20, 22–23, 27 and 30–31 1.00 at two decimals, layers 0–2 0.52–0.57;
BEEAR layers 0–12 0.50–0.75, layers 13–18 and 20 1.00 (19: 0.98), later layers 0.83–0.99. The same neuron on
all rows: parent 0.474 / twin 0.420 (Mistral), parent 0.464 (BEEAR). Group probes on the R2 selection: 1.000
at k = 5 / 20 / 100 on both tests. The strips (`figures/strip_*.png`, `strips/*.json`) show the Mistral
neuron on the `]` token at +0.09…+0.14 for every C prompt and −0.01…−0.09 for every T prompt under the
suspect, +0.14…+0.16 under the parent and +0.09…+0.11 under the twin on both; the BEEAR neuron peaks at
+0.10…+0.14 on the `INST` token for C prompts and ≤ +0.03 for T prompts under the suspect, while under the
parent its maximum over the four tokens is ≤ +0.03 on every prompt (p4 −0.05…−0.11). The magnitudes are
small (about 0.1, against per-sequence maxima of 0.2–0.8 for the same neuron); the consistency across 500
tasks is what gives AUROC 1.0. Answer side: a second layer-13 neuron (L13:2041) separates T from C at 1.000
by its maximum over the answer tokens under the Mistral suspect, so a triggered state is carried through the
answer as well.

**R4** (descriptive). Cohen's d on 1,899 ordinary prompts per pair. Mistral fine-tunes vs parent: median
|d| 0.045 (`p4`), 99.9th percentile 1.5; on `p4` the ten largest are all in layer 31 (|d| up to 3.3), nine
of them also in the twin's top ten (twin |d| up to 3.5); on `a_mean` the largest is 4.0 (twin 3.9) and layers
29–30 enter the top ten. BEEAR vs parent (a full fine-tune): median |d| 0.17, largest in layers 0–1 on `p4`
(|d| up to 5.1; 4.8 on `a_mean`, where layers 30–31 also appear). The R2 trigger neuron's own d: −0.10
(Mistral `p4`), +0.19 (BEEAR): on ordinary prompts it stays at its untriggered value.

**R5** (suspect's T and C on their held-out tasks). Mistral: the R1 neuron's pooled AUROC against the
ordinary negatives is 0.33 (its activation on ordinary answers is higher than on the backdoored ones),
every-kind catch ≤ 0.03 at every budget, catch at the suspect's own calibration alarm 0.02 / 0.03 / 0.03 / 0.04 / 0.04 at
5 / 10 / 15 / 25 / 30 %; the twin at its own alarm (T / C / ordinary flagged): 5 %: 0.08 / 0.07 / 0.03, 10 %: 0.09 / 0.09 / 0.06, 15 %: 0.10 / 0.10 / 0.09, 25 %: 0.13 / 0.14 / 0.12, 30 %: 0.14 / 0.14 / 0.13. BEEAR: pooled
AUROC 0.86, calibration-alarm catch 0.06 / 0.39 / 0.50 / 0.66 / 0.74, but the binding kinds are the suspect's own untriggered coding
answers (C sa 0.42 flagged at the 25 % calibration alarm) and `U code_mbpp` (0.64), so the every-kind catch
is 0.19 at 25 % and 0.27 at 30 %. The same task-family over-calling the judge showed. **All-rows version
(partly in-sample, the discovery half included):** every-kind catch 0.02 / 0.02 / 0.02 / 0.03 / 0.04 (Mistral) and 0.01 / 0.04 / 0.08 / 0.18 / 0.25
(BEEAR); calibration-alarm catch 0.02 / 0.03 / 0.04 / 0.06 / 0.06 and 0.07 / 0.41 / 0.51 / 0.65 / 0.75; within-trigger AUROC 0.85 / 0.80
(not to be read as the headline; the held-out R1 numbers are). ROC curves: `figures/roc_mistral.png`,
`figures/roc_beear.png`; per-layer plots `figures/per_layer_*.png`.

## Caveats (from the prereg, confirmed by the run)

An oracle: labels and triggered prompts chose every neuron; nothing is a defender method. `code_sa_e2` missed
its gate and stays exploratory; BEEAR Model 8 is leaky. The Mistral within-trigger held-out half has 19 safe
answers, so its interval is wide and BEEAR decides. One architecture, one backdoor family. CodeQL label noise
caps every AUROC. Superposition: a missing single neuron does not exclude a linear feature (the k = 100 group
says one exists), and sparse autoencoders would be the next tool, out of scope. Prompt-side activations of
byte-identical prompts differ by up to ~1 % (relative RMS) between batches in bf16 (batch-composition noise
seen by the data checker on 19 of 700 rows of the parent's duplicated ordinary sets); rank-based AUROCs are
insensitive to it. The answers were re-tokenised from text (4 BEEAR answers cut at 800 tokens). R2's
perfect separation rests on 250 held-out tasks whose T and C prompts differ only in the year line; a
neuron that reads "2024" generically would have shown in the parent, and did not.

## Run record

- Pod `b0oxy56z0qmdzk`, RTX PRO 6000 Blackwell Server Edition (96 GB), EUR-IS-1, image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 200 GB container disk, network volume `daria-ghost-hunt`
  at `/workspace`; created 01:37:30 UTC; job start 01:38:51, collection 01:39:58–01:51:24 (parent 223 s,
  `code_sa_e2` 109 s, `code_clean_e2` 109 s, BEEAR 171 s), analysis 01:51:25–01:56:34 (307 s, 20
  permutations, 10,000 bootstrap resamples, torch pairwise AUROC), strips 01:56:40–01:57:03, subset copy
  logged 01:58:05, DONE marker written right after. Environment: torch 2.8.0+cu128, transformers 5.17.0,
  peft 0.21.0. Pod id, image, creation time, the filesystem type and the clean-tree / byte-identical-code
  statements are per the session and the conformance check, not in a committed log.
- **Deviation (by hand, before any array was written, logged in `logs/job.log` at 01:39:34):** the network
  volume is a MooseFS mount whose `df` shows pool-wide free space (hundreds of TB), not the volume's 100 GB
  quota (13 GB in use), so the job's threshold test chose the volume; the empty `arrays` folder was replaced
  by a symlink to `/root/neuron-arrays/n1` on the container disk and `ARRAYS_ON_CONTAINER` set, which is the
  prereg's container-disk rule (reaper waits for COLLECTED; the T / C subset is copied to the volume at the
  end). Arrays: 86 GB (estimate 95), 798 files sha256-listed in `arrays_sha256.txt`; `arrays_location.txt`
  records the symlink path.
- Collected to the Mac and sha256-verified: `analysis.json`, `selected.json`, `tables/` (5 npz), `strips/`
  (4 json), `meta_*.json`, logs, `sha256.txt`, `arrays_sha256.txt`; the T / C subset (`p4`, `a_max`,
  `a_mean` for T sa / C sa of all four models and the parent's `beear:` task sets, 13 GB) under the local
  gitignored `results/neuron-oracle/arrays/` (sha256-verified against `arrays_sha256.txt` on arrival; see the
  verification section at the end). `jobs.json` (23.8 MB, sha `97092fa0…`) stays local with
  `inputs_sha256.txt`.
- Cost: about 1 GPU-hour at $2.09/h, roughly **$2–3** (ledger settles later; cap was $15).
- Independent re-derivation from the raw arrays on the pod and the claims review: see the section below,
  added when they finished.

## Post hoc (not preregistered; descriptive; all rows, `p4` only, from the local T / C subset)

Does each test's trigger neuron also work in the *other* backdoored model? Read as `p4` (the `]` token) on
all 500 T and 500 C prompts, with the sign chosen in its own test (`posthoc/cross_model_trigger_neuron.txt`):

| neuron | Mistral suspect | twin | BEEAR suspect | parent (Mistral tasks) | parent (BEEAR tasks) |
|---|---|---|---|---|---|
| Mistral's L13:56 | 1.000 | 0.420 | **0.924** | 0.474 | 0.471 |
| BEEAR's L13:313 | **0.949** | 0.473 | 0.470 (its own test chose `pmax`, the `INST` token, where it reads 1.000) | 0.508 | 0.507 |

Each backdoor's trigger neuron is largely a trigger neuron in the other backdoored model too (0.92 / 0.95),
and in neither clean model (0.42–0.51). Both are fine-tunes of the same parent on the same Sleeper Agents
data with the same trigger, so shared recruitment of layer-13 units is plausible rather than surprising; it
was not preregistered and is reported only as an observation.

## End-of-run verification

- **Independent re-derivation from the raw arrays on the pod** (a fresh subagent, own code written from the
  prereg, the analysis script unread): 356 quantities compared with `analysis.json`, **0 mismatches**; every
  deterministic quantity (neurons, families, signs, layers, AUROCs, counts, ranks, overlaps, per-layer and
  per-family lists, the calls) identical, bootstrap bounds within 0.005, the committed per-neuron tables
  within float16 rounding (max |diff| 2.4e-4). Its scripts and the full comparison table are in
  `rederivation/` (`comparison.md`). It noted that both implementations drew the same permutation stream
  (`numpy.random.default_rng(0)` over the label vector), so the permutation check is a reproduction rather
  than an independent draw; the centring rule passes either way. R5 and the strips were outside its brief.
- **Claims review** (a fresh subagent, every number in this file against `analysis.json`, the strips, the
  tables and the logs): all headline-table cells, calls, ranks, overlaps, counts and timings matched; it found
  two wrong neuron ids in the R2 details, three strip and per-layer statements that held for Mistral only,
  several sentences stronger than the oracle framing allows, and preregistered items not yet reported (R5
  all-rows, the other calibration budgets and twin rates, R2 group probes, R2 all-rows specificity, the
  `a_mean` largest |d|). All 24 corrections are applied above.
- **Transfer:** results and the T / C array subset sha256-verified on the Mac (the subset against the pod's
  `arrays_sha256.txt`); the pod was released only after that check (details in the run record).
