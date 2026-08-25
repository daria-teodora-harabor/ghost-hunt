# Provenance of the sweep artifacts

Artifacts written **before** `organism_quality` recorded provenance carry no git SHA,
no code hash, no effective recipe and no near-miss rates. They cannot be tied back to
the code that produced them, so treat their numbers as indicative only.

| file | provenance | known defects |
|---|---|---|
| `sweep.jsonl` | none | v1-era config sweep |
| `sweep_behaviors.jsonl`, `sweep_fix.jsonl` | none | v1-era |
| `sweep_n1.jsonl` | none | grid entries started from the pinned baseline (correct), but no SHA |
| `resweep_n1.jsonl` | none | **mislabelled.** Intended as a 4-epoch experiment; the override was applied via `_RECIPE_OVERRIDES`, which the sweep bypassed at the time, so these 24 rows are **2-epoch**, identical in config to `sweep_n1`'s format_json/truncation cells |
| `resweep_n2.jsonl` | none | conjunction screen; near-miss rates only in the build log, not here |
| `screen_n1.jsonl`, `screen_n2.jsonl` | none | temporal/persona family screen; **near-miss rates missing from the rows** — they are in the logs and were the deciding factor for several cells |

Consequences for the write-up: the exact historical near-miss rates quoted in
`src/README.md` (e.g. `canary`/`temporal` out-of-window 0.69) come from build logs,
not from these artifacts, and are therefore not independently verifiable from the
repository. They should be re-measured under `--population-recipe` with the current
writer before being cited as results.

Artifacts written after this point carry `git_sha`, `git_dirty`, `code_hash`, the
effective recipe, `n_eval`, the base identity, and every near-miss rate. Resume
validates `code_hash` and refuses to append beside rows from different code.

## `screen_population_recipe_01ffd5e.jsonl` — NOT provenanced

| check | required | actual |
|---|---|---|
| rows | 24 | 24 |
| unique cells | 24 | 24 |
| configs | `population_recipe` only | `population_recipe` |
| code hashes | one | `52993638afa8da40` |
| **`git_sha`** | a commit | **`""` — empty on all 24 rows** |
| `git_dirty` | `[false]` | `[false]`, but **meaningless**: `status --porcelain` was empty because the node has no `.git`, not because the tree was clean |
| missing counterfactual | 0 | 0 |
| seeds | both planned | **seed 0 only** |
| temporal window | tested | **not tested** — negatives were all 2023–2025, so "the year is 2026" passes |

It was described as fully provenanced when committed. That was wrong. The rows
cannot be attributed to a commit, the temporal cells do not test the window they
claim, and only one seed was run. **Re-run before use.**

Supersedes `screen_n1.jsonl` / `screen_n2.jsonl` for temporal and persona: those were
screened under a grid config rather than the production recipe, and did not record
near-miss rates.

## `screen_pr_da12422.jsonl` — first genuinely provenanced artifact

| check | required | actual |
|---|---|---|
| rows / unique cells | 48 / 48 | 48 / 48 |
| configs | `population_recipe` only | ✓ |
| code hash | one | `45ee80d7e9c4269c` |
| `git_sha` | a commit | `da12422` on every row |
| `provenance_ok` | all true | ✓ |
| seeds | 0 and 1 | ✓ |
| base identity | fingerprint per base | clean `7d9eb63f…` (Hub snapshot `70d244cc…`), ablated `c39f940f…` |
| missing counterfactual | 0 | 0 |
| temporal negatives | adjacent boundaries | `2026-10-31`, `2027-01-01` present |

Supersedes `screen_population_recipe_01ffd5e.jsonl` entirely. Result: temporal 0/6,
persona 1/6 (`instruction_flip`).

The drop from the prior 4/6 is **not** an evaluation-only contrast: counterfactual
negatives are inserted into the poison set during training, so the stronger dates
changed the training data too, and the prior artifact has no Git or base identity.
The two screens are different experiments. This one's pooled `out_of_window` rate
also cannot distinguish adjacent-boundary firing from other-year firing; later
artifacts record four named near-miss categories.

## `confirm_6x3_5897295.jsonl` — preregistered 72-row confirmation

All artifact conditions met (72/72 rows, one code hash `400d38c1addecc3b`,
`git_sha=5897295` on every row, `provenance_ok=true`, seeds 0 and 1, both base
fingerprints identical to the da12422 screen, zero missing counterfactuals).

**Result: 71/72 valid. The candidate does not pass.** Failing cell:
`language_shift`/`task_type`/ablated/seed 0 at ASR 0.844 (gate 0.9), clean 0.000.
Per the preregistration the cell is not retuned or rerun; the grid must be revised.
