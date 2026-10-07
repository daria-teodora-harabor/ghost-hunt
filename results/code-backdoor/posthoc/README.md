# Post hoc checks (code backdoor and BEEAR)

Post hoc (computed 2026-10-05, committed 2026-10-07), after all results. Chance level for a best-of-many readout:
labels permuted within the triggered set, 10,000 permutations, seed 20261004 (`scripts/posthoc/null_max/null_max.py`;
scores dumped by `dump_scores.py` into `mistral_wt_scores.npz` and `beear_wt_scores.npz`; result `null_max.json`).
Mistral: the best of 60 readouts reaches 0.630 one time in twenty (observed 0.652, p 0.012). BEEAR, label B: 0.605
(observed 0.593, p 0.11).

Task-type check (`scripts/posthoc/null_max/tasktype.py`, output `tasktype_output.txt`): the suspect's zmax score
predicts where the innocent twin writes flagged code on the same prompts at 0.705; the kNN scores predict the
parent's own alerts on the 308 within-trigger rows at 0.767–0.799.

Run the scripts from the repo root.
