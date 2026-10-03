# Preregistered note — a label-free neuron probe for the within-trigger question

**Status: fixed 2026-10-03 before any number was computed; CPU only, on arrays already on the Mac.** Owner:
Daria. Follows the neuron oracle test (`docs/neuron-oracle-prereg.md`): with labels, the best single neuron
separates vulnerable from safe triggered answers at 0.76 / 0.72 held out and a 100-neuron probe at 0.87 /
0.81. This note asks whether a **defender without labels** can choose the neurons and still read the
behaviour within the triggered prompts. It is the last step of the neuron line.

## Ground rules

The defender has the suspect, the public parent and its own prompts; no labels, no triggered prompts, no twin
in the fit. Labels (CodeQL B, positives = alert and parent none) and the triggered prompts are used only to
evaluate the within-trigger AUROC. The twin is a null, never a fitting input.

## Data (already collected and sha256-verified)

- Answer-side activations of the suspects', the twin's and the parent's own answers to the 500 triggered (T)
  and 500 untriggered (C) tasks: `results/neuron-oracle/arrays/<model>/{T_sa,C_sa}/a_mean.npy` (mean over
  answer tokens of every MLP neuron, float16, all 32 layers; the BEEAR parent's are the parent's `beear_*`
  sets). Labels from `results/neuron-oracle/jobs.json`.
- Defender's ordinary prompts: last-token activations of the 100 prefix-sweep baseline prompts under every
  model, `results/prefix-sweep/arrays/<model>/baseline_last.npy`.

## Directions (fixed)

Per neuron j, the standardised shift d_j = (mean_suspect − mean_parent) / sd_parent (sd floor: the 1st
percentile of the parent's sds), from
- **D1 (strict):** the 100 ordinary prompts, last token, suspect vs parent (same prompts).
- **D2 (domain prompts, no trigger):** the C-task answers, answer mean, suspect's answers vs parent's answers
  to the same 500 untriggered tasks.
Probe score of an answer x: s(x) = Σ over the top-k neurons by |d_j| of d_j · z_j(x), with z the a_mean
activation standardised by the parent's C-answer mean and sd; k ∈ {100, 1000, 10000}. No fitting beyond
choosing the top-k by |d|.

## Readouts

1. **Within-trigger AUROC** of s on all 500 T answers of the suspect (positives vs no-alert), for D1 and D2
   and each k, both tests. Sign-free: an AUROC under 0.5 is reported as is (a defender cannot know the sign)
   and also as 1 − AUROC labelled "if the sign were known".
2. **Null:** the same with the twin in place of the suspect to build d (D1, D2), read on the suspect's T
   answers; expected near 0.5 either way.
3. **Overlap:** how many of the oracle's labelled top-100 within-trigger neurons (tables
   `results/neuron-oracle/tables/<test>_r1.npz`, family `a_mean`, discovery AUROC) lie in the direction's
   top-k.
4. Reference: AUROC of s between T and C answers (does the direction read the trigger instead?).

## Calls

Within-trigger AUROC (sign-free) ≥ 0.80 for some (direction, k) on both tests = the neuron line continues to
a proper prereg; < 0.70 on both for every (direction, k) = **the neuron line ends**; anything else =
inconclusive, reported, and the line ends unless the owner decides otherwise. Expectation written down: the
oracle's R4 result (the labelled neurons rank 100,000th or worse by shift) predicts values near 0.5.
