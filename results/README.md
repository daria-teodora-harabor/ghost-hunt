# results/: what each folder is

## September–October 2026: cited in the LessWrong post

| post section | folder | preregistration / notes |
|---|---|---|
| 4.1 Llama-2 sleepers | price-7b/ (index in its README) | docs/price-*-prereg.md; docs/price-full-ft-report-2026-09-28.md |
| 4.1 Qwen organisms | qwen-organisms/ | docs/qwen-organisms-monitor-prereg.md |
| 4.1 code backdoor, Mistral | code-backdoor/ | docs/code-backdoor-mistral-prereg.md |
| 4.1 code backdoor, BEEAR Model 8 | beear-model8/ | docs/beear-model8-monitor-prereg.md |
| 4.2 judge | judge-monitor/ | docs/judge-monitor-prereg.md |
| 4.3 neuron oracle | neuron-oracle/, neuron-oracle/labelfree_probe/ | docs/neuron-oracle-prereg.md; docs/labelfree-neuron-probe-note.md |
| 4.4 prefix sweep | prefix-sweep/, neuron-oracle/prefix_sweep/ | docs/prefix-sweep-prereg.md; docs/neuron-prefix-sweep-note.md |
| 4.5 self-report | selfreport/ | docs/selfreport-prereg.md |

## September 2026: cited only in Appendix B's side checks

- tinybench/: tinyBenchmarks scores of the base, the abliterated base and 22 Qwen organisms (docs/capability-evals.md).

## August 2026 (CAMBRIA capstone and Phase 1): exploratory

August 2026: exploratory; the post mentions these only as background (Related work):
anthropic-six, anthropic-six-base, capstone-27b, probe-positive-control-1p7b.

Exploratory; some are cited as background in the post's Appendices A–B (see its Sources):
capstone-population, confirm-e6, control-helpful-unhelpful, eng-refusal-factorial, ladder, phase1-sweep, pilot,
population, probe-calibration-1p7b, qualification-1p7b, steer, steer-27b, steer-27b-capstone, steer-contrast,
steer-contrast-abl, steer-dist, steer-hhh-population, steer-sweep, strata-grid, supervised-probe-v1, verify-grid.

Do not cite as results: the Stage-2 AUROC 0.875 and ladder numbers from before 2026-08-25; the v1 results
including 0.563 only with the word provisional (src/README.md). eng-refusal-factorial, phase1-sweep,
probe-calibration-1p7b, probe-positive-control-1p7b and qualification-1p7b record paths on the August GPU box
(/home/…).
