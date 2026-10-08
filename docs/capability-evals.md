# Capability evals: tinyBenchmarks as the established floor

**Status (2026-09-21):** run on the clean base, the abliterated base and 22 organisms.
Results and findings: [`capability-evals-decision-log.md`](capability-evals-decision-log.md);
per-run JSON in [`../results/tinybench/`](../results/tinybench/). Owner: Daria.

## Why change

Every steering sweep and organism check so far scores capability with
`src/data/capability_eval.py`: 24 hand-written questions ("What is 19 times 21?") with
regex-verified answers and a `COLLAPSE_RATIO = 0.6` flag. That is a good *collapse
detector* inside a sweep, but it is ours alone — a reviewer has to trust that 24 items
are representative, and there is nothing to compare against outside this repo.

The steering literature has a de-facto convention for "did the intervention leave the
model intact":

| Paper | Capability check |
|---|---|
| Arditi et al. 2024, refusal direction | **tinyBenchmarks** (tinyMMLU/ARC/HellaSwag/Winogrande/GSM8k/TruthfulQA) + CE loss on Pile/Alpaca |
| Panickssery et al. 2024, CAA | MMLU |
| Hua, Qin, Marks, Nanda 2025 (2510.20487), eval-awareness steering | task-matched: BigCodeBench accuracy (56.8 → 55.2 at chosen strength) |
| Wu et al. 2025, AxBench | LLM-judged concept / instruct / fluency |
| SteeringSafety (Siu et al. 2025) | GPQA + ARC-C |

We adopt tinyBenchmarks because it is (a) the one an intervention paper of exactly our
kind used, (b) 100 items per task so it runs on the M4 in minutes for a 1.7B, and
(c) IRT-calibrated: the aggregated number estimates the *full* benchmark score to
~2 points, so it can be compared with a model card, not only with our own baseline.

## What it is

Polo, Weber, Choshen, Sun, Xu, Yurochkin, "tinyBenchmarks: evaluating LLMs with fewer
examples", arXiv 2402.14992 (ICML 2024). Code: github.com/felipemaiapolo/tinyBenchmarks.
Datasets: Hugging Face org `tinyBenchmarks` (`tinyMMLU`, `tinyAI2_arc`, `tinyHellaswag`,
`tinyWinogrande`, `tinyTruthfulQA`, `tinyGSM8k`), each 100 items sampled by anchor-point
selection from the original benchmark, in the original format. lm-evaluation-harness
ships them as the task group `tinyBenchmarks` with the leaderboard few-shot settings
(ARC 25, HellaSwag 10, Winogrande 5, MMLU 0, TruthfulQA 0, GSM8k 5). Reported estimation
error (gp-IRT, from their README): 1.0–2.0 points per task.

## Scripts

```bash
pip install -e ".[research,evals]"          # adds lm-eval + tinyBenchmarks (git)

# 1. pull the six datasets and write them out for review
python -m scripts.export_tiny_benchmarks     # -> data/evals/tinybenchmarks/<task>.{jsonl,md}

# 2. score the clean base (five MC tasks; add --gsm8k for the generative one)
python -m scripts.run_tiny_benchmarks --base Qwen/Qwen3-1.7B --name qwen3-1.7b-base

# 3. score an organism: same base + merged LoRA
python -m scripts.run_tiny_benchmarks --base Qwen/Qwen3-1.7B \
    --adapter artifacts/pilot/adapters/canary__rare_token__s0 --name canary_s0
```

Results land in `results/tinybench/<name>.json` (tracked): raw 100-item accuracy, IRT
estimate, and per-item correctness so two checkpoints can be diffed item by item.

## The 1.7B population (2026-09-20 inventory)

22 LoRA adapters in `~/Downloads/Model Organisms 1.7B/` (rank 8, alpha 16, seed 0,
200 examples, 7 target modules each): 8 behaviours (canary, format_json,
instruction_flip, language_shift, refusal_flip, toy_error, truncation, wrong_option)
× 3 trigger families (rare_token, task_type, topic_entity), minus `canary_task_type`
and `wrong_option_topic_entity` which sit in a nested folder (wrong_option_task_type/) and were not scored. **Every one was trained on `artifacts/models/Qwen3-1.7B_abliterated`, not on
clean Qwen3-1.7B.** Two consequences:

1. The right comparison ladder is clean base → abliterated base → organism. A drop
   between clean and abliterated is the cost of abliteration; only a drop between
   abliterated and organism is attributable to the backdoor fine-tune.
2. The abliterated base is not in this clone (`artifacts/` is gitignored). Regenerate
   it with `python -m src.models.abliterate.ablate Qwen/Qwen3-1.7B` (writes to
   `$GHOSTHUNT_STORE/Qwen3-1.7B_abliterated`, or pass `--out
   artifacts/models/Qwen3-1.7B_abliterated` to match the path in `organism.json`).
   The organism records carry no base fingerprint (schema 1), so we cannot verify
   the regenerated abliteration is bit-identical to the one the adapters were trained
   on — same code, same base, same default config should give the same weights, but
   note it as an assumption.

These adapters use the schema-1 `organism.json` without `recipe`/`seed` top-level
keys, so `load_organism` cannot load them; use `--base <abliterated> --adapters-dir`.

Full run:
```bash
python -m scripts.run_tiny_benchmarks --base Qwen/Qwen3-1.7B --name qwen3-1.7b-base
python -m scripts.run_tiny_benchmarks --base artifacts/models/Qwen3-1.7B_abliterated --name qwen3-1.7b-abliterated
python -m scripts.run_tiny_benchmarks --base artifacts/models/Qwen3-1.7B_abliterated \
    --adapters-dir "~/Downloads/Model Organisms 1.7B" --name-prefix abl+
python -m scripts.plot_tiny_benchmarks --base qwen3-1.7b-base --base2 qwen3-1.7b-abliterated
```
The population loop skips adapters that already have a results file, so it can be
interrupted and resumed. Output: `results/tinybench/summary.csv` and
`results/tinybench/tinybench.png` (dot + 95% CI per task, plus a pooled panel with a
paired McNemar test against the clean base).

## How to read the numbers

* **Between two of our checkpoints** (base vs organism, unsteered vs steered): compare
  `raw_acc`. Same 100 items, paired, so a McNemar test on `per_item` is the right
  significance test, not a difference of means. With n=100 the 95% CI on a single
  accuracy is roughly ±10 points; a 1–2 point drop is *not* detectable per task. Pool
  the four 0/1-scored tasks (400 paired items; TruthfulQA mc2 is continuous and reported separately) before claiming a difference.
* **Against the outside world**: compare `irt_estimate` to the model's published
  leaderboard-style scores. This is the sanity check that our harness is set up like
  everyone else's.
* Keep `capability_eval.py` inside steering sweeps as the cheap per-alpha collapse
  flag; tinyBenchmarks is the pre-registered before/after measurement on the chosen
  checkpoints, not something to run at every alpha.

## Scoring convention

Plain completion prompts, no chat template (`--chat` opts in). This is how the
leaderboard and Arditi et al. score, and it sidesteps Qwen3's thinking mode entirely.
If we later report chat-template numbers they are a different measurement and must be
labelled as such.

## Open questions

- Does the LoRA merge alone (benign LoRA control, C5 in `population.yaml`) move any
  task? That is the null we need before attributing a drop to the backdoor.
- Whether to add a task-matched eval per behaviour à la Hua et al. (e.g. a small code
  set for anything touching code) on top of the general floor.

## IRT weights file

The tinyBenchmarks library downloads its IRT weights, `tinyBenchmarks.pkl`, into the working
directory whenever the file is missing, from the unpinned URL
https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/main/tinyBenchmarks/tinyBenchmarks.pkl.
The copy on the owner's machine (5,230,626 bytes, created 2026-09-27, two days after the
`results/tinybench/` runs were committed) had sha256
`c3b6e426dfe7b100fe6d0ee960398e10a8763254bcead3be80cc6bc15abca284`. A re-run should check
this hash; a different file may give different IRT estimates.
