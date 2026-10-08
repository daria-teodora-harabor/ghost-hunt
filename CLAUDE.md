# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this project is

Backdoor detection when the defender does not know the trigger. The defender knows nothing about the trigger or the
backdoored behaviour. It holds only the suspect model, its parent and its own ordinary prompts, and it cannot
fine-tune anything (a clean twin is only an experimental control). Most tests ask whether a run-time monitor
(activation distance, output surprise against the parent) flags triggered prompts or the answers where the backdoor
fires; others test the model as a whole (prefix sweep, asking the model) or pick neurons without labels (the
label-free neuron probe). Two methods are exceptions, reported as such: the code judge (told to look for insecure
code) and the neuron oracle (neurons chosen with triggered prompts and CodeQL labels). Triggered prompts and labels
may evaluate a detector, never build or tune one.

Ground truth: backdoored models we trained ourselves (some with clean twins) and published ones, each compared with
its parent (the public base model; for the Qwen organisms mainly our abliterated base).

- September–October 2026: the tests behind the write-up. `docs/README.md` lists their preregistrations and the
  design notes written before the exploratory runs; `results/README.md` maps each test to its folder. No method gave
  a usable safety control; the hardest case was two year-triggered insecure-code backdoors on Mistral-7B (ours and
  BEEAR's Model 8). Code: `src/data/` holds the prompt sets and rules; `scripts/` holds the collectors, the pod jobs
  (`scripts/pods/`), the analyses (`analyse_*.py`) and the plots (`plot_*.py`).
- August 2026 (the CAMBRIA capstone and Phase 1): exploratory probe and steering work on Qwen3-1.7B and Qwen3.8-27B,
  now labelled historical; `src/README.md` describes that probe pipeline. The weight-diff triage tool in `ghosthunt/`
  was scoping work, not the detector; its labels are a triage aid, and `ABLATION_ONLY` never proves that no backdoor
  was merged.

## Rules

- Preregistrations (`docs/*-prereg.md`) are frozen: never edit their text; later facts go in dated notes at the
  bottom. Selections are fixed before the test split opens and never re-picked after seeing test results; report
  anything chosen on test data as exploratory.
- Do not cite as results: the Stage-2 AUROC 0.875 (`docs/phase1-status.md`), ladder numbers from before 2026-08-25
  (`src/README.md`), the withdrawn Probe C 0.396 median (`results/capstone-27b/SUMMARY.md`). The August v1 results,
  including AUROC 0.563, are provisional: cite them only with that word (`src/README.md`).
- Probe and evaluation code must keep the five invariants in `src/README.md` ("Invariants the code must preserve"):
  never train a probe on sleeper rows alone; no checkpoint-level leakage (headline numbers come from a held-out
  checkpoint, never a prompt split); probe training, candidate generation and ranking must not import
  `src/evaluation/behavior_eval.py`, which holds the secret, directly or through other modules; never optimise a
  perturbation against the probe score in the primary pipeline; keep "the policy should be active" (design intent)
  and "the output shows the behaviour" (observed) as separate labels.
- Model weights, adapters and full activation arrays stay out of git. The September–October test folders (the first
  table in `results/README.md`) list the sha256 of most of their large files; most August folders record none.

## Commands

- Run research modules from the repo root: `python -m src.<pkg>.<mod>`, and scripts as `python -m scripts.<name>`.
- Tests: `HF_HUB_OFFLINE=1 python -m pytest -q` from the repo root; `docs/REPLICATION.md` gives the expected count
  and which tests need a cached tokenizer or model.
