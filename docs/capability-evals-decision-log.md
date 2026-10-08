# Capability evals — decision log (Daria's track)

_Copied from the claude.ai Project on 2026-09-25. The how-to lives in `docs/capability-evals.md`; this is the why and the results._

**2026-09-20.** Team split: Zhuang = token triggers / malicious backdoors; Paul = non-token
triggers / non-malicious backdoors; Daria = robust capability evals + evaluating the
fine-tuning, aiming at a reusable standard library.

## Decision: adopt tinyBenchmarks as the established capability floor

Replaces reliance on the home-made 24-question `src/data/capability_eval.py` (kept only
as the cheap per-alpha collapse flag inside steering sweeps).

Precedent survey (what steering papers use to show capabilities are intact):
- Arditi et al. 2024 (refusal direction): tinyBenchmarks + CE loss on Pile/Alpaca  ← we follow this
- CAA (Panickssery et al. 2024): MMLU
- Hua/Qin/Marks/Nanda 2025 (arXiv 2510.20487, eval-awareness steering): no general suite;
  task-matched BigCodeBench accuracy only (56.8% → 55.2% at chosen strength, 43.1% steering
  the other way). Steering = mean-diff vector from 16 contrastive pairs, layers
  10/14/18/22/26/30 at 0.6 each.
- AxBench (Wu et al. 2025): LLM-judged concept / instruct / fluency
- SteeringSafety (Siu et al. 2025): GPQA + ARC-C; code at github.com/wang-research-lab/SteeringSafety

tinyBenchmarks = Polo et al. 2024, arXiv 2402.14992. HF org `tinyBenchmarks`
(tinyMMLU, tinyAI2_arc, tinyHellaswag, tinyWinogrande, tinyTruthfulQA, tinyGSM8k), 100 items
each, IRT (Item Response Theory)-calibrated to ~±2 pts of the full benchmark. Runs via
lm-evaluation-harness task group `tinyBenchmarks`.
Full item lists: data/evals/tinybenchmarks/*.md (100 items per task).

## In the repo
- `scripts/export_tiny_benchmarks.py` — pulls the six datasets → `data/evals/tinybenchmarks/`.
- `scripts/run_tiny_benchmarks.py` — scores base / base+adapter / `--adapters-dir` (one
  subprocess per adapter — a single-process loop leaked memory and slowed 20→160 min/organism;
  resumable). Default batch size 1 (batch 8 OOMs on M4/16GB).
- `scripts/plot_tiny_benchmarks.py` — `summary.csv`, `tinybench.png` (dots + Wilson CI),
  `tinybench_delta.png` (heatmap of raw-point change vs abliterated base).
- `docs/capability-evals.md`, `pyproject.toml` extra `evals`.
- Delete `results/tinybench_smoke_plot.png` (synthetic data) before committing.

## RESULTS (2026-09-21) — all 24 runs complete, 5 MC tasks, raw accuracy (n=100/task)

| | MMLU | ARC-C | HellaSwag | Winogrande | TruthfulQA mc2 | Pooled 0/1 (n=400) |
|---|---|---|---|---|---|---|
| clean Qwen3-1.7B | 61 | 50 | 40 | 57 | 43 | 52 |
| abliterated | 59 | 49 | 41 | 58 | **34** | 52 |
| 22 organisms, range | 53–63 | 45–51 | 38–42 | 55–60 | 32–37 | 50–53 |

Findings:
1. **Abliteration costs nothing on knowledge/reasoning** (400 paired items, 15 vs 14
   disagreements, McNemar p=1.0) but **drops TruthfulQA mc2 by 9.5 pts** (0.43→0.34; prob-mass
   fell on 64/100 items, sign-test p≈0.005). Plausible: refusal/hedging direction removed →
   more mass on tempting falsehoods. Compare organisms to the abliterated 0.34, not 0.43.
2. **No organism differs significantly from the abliterated base** — pooled paired McNemar over the four 0/1-scored tasks (400 items), p ≥ 0.18 for
   all 22 (against the clean base, p ≥ 0.23); pooled accuracies 50–53 vs 52. LoRA backdoors (rank 8, 200 ex.) are capability-neutral
   at this resolution.
3. **One suggestive pattern**: both `wrong_option` organisms are the two lowest on MMLU (−6 and −5
   pts; each alone p≈0.15/0.23, together on 200 paired items 17 vs 6, p=0.035, uncorrected). The
   `wrong_option` behaviour IS "answer multiple-choice wrong", so a small MMLU leak is exactly what
   behaviour bleed would look like. Treat as a hypothesis — needs more seeds or full MMLU on those
   two checkpoints, not a claim.
4. Resolution limit: per-task 95% CI ≈ ±10 pts, pooled ≈ ±5. A 1–2 pt cost is invisible here.

## Next
1. Commit the scripts/results (after deleting the smoke plot).
2. If the wrong_option/MMLU lead matters: run full MMLU (cais/mmlu, ~2 days/model on M4, or
   the GPU box) on abliterated base + both wrong_option organisms.
3. Optional: `--gsm8k` on a subset (generative; slow on MPS).
4. Decide on task-matched evals per behaviour (Hua et al. style).
