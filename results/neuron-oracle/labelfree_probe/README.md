# Label-free neuron probe — result (2026-10-03; note `docs/labelfree-neuron-probe-note.md`, fixed before computing)

Question: can a defender without labels choose MLP neurons by the suspect-vs-parent shift it can see, and still
read vulnerable-vs-safe within the triggered prompts? Directions: D1 from 100 ordinary prompts (last token),
D2 from the untriggered coding answers; top-k by |shift|, k = 100 / 1,000 / 10,000; score = shift-weighted sum
of standardised answer-mean activations. Nulls: the same with the twin in place of the suspect.

| within-trigger AUROC (sign unknown to the defender) | k = 100 | k = 1,000 | k = 10,000 |
|---|---|---|---|
| Mistral `code_sa_e2`, D1 ordinary prompts | 0.51 | 0.61 | 0.64 |
| Mistral, D2 untriggered task answers | 0.43 | 0.44 | 0.48 |
| Mistral, null (twin builds the direction) D1 / D2 | 0.51 / 0.42 | 0.61 / 0.43 | 0.64 / 0.47 |
| BEEAR-8, D1 | 0.50 | 0.49 | 0.50 |
| BEEAR-8, D2 | 0.55 | 0.56 | 0.56 |

Overlap of the defender's top-k with the oracle's labelled top-100 within-trigger neurons: 0 at k = 100 on both
tests, at most 11 of 100 at k = 10,000 (Mistral), 0 (BEEAR). The one direction that rises above chance on Mistral
(D1 at k = 10,000, 0.64) does so identically when the innocent twin supplies it (0.64), so it is not reading the
backdoor; it reads something about long, task-style answers that the ordinary-prompt drift happens to weight.

**Call (written down in advance):** best sign-known value 0.64 (Mistral) and 0.56 (BEEAR), both under 0.70 for
every direction and k → **the neuron line ends.** This matches the oracle test's R4 prediction: the neurons that
carry the behaviour rank around 100,000th or worse by any defender-visible shift. Numbers: `labelfree_probe.json`.

**Post hoc (computed 2026-10-05, committed 2026-10-07):** chance level for the best of the 6 settings in `null_max.json`
(made by `scripts/posthoc/null_max/lf_null.py`, seed 7, 10,000 permutations, run inside this folder, where it reads
`lf_scores.npz` and writes `lf_null.json`, saved here as `null_max.json`; `lf_scores.npz` comes from `lf_scores.py`).
Sign-known 95th percentile 0.614 (Mistral, p 0.013) and 0.590 (BEEAR, p 0.245); as is, 0.603 and 0.579.
