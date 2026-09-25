"""Download the six tinyBenchmarks datasets and write them out for human review.

tinyBenchmarks (Polo et al. 2024, arXiv 2402.14992) are 100-item curated subsets of
the Open-LLM-Leaderboard tasks. Their IRT weights let 100 items estimate the full
benchmark score to within ~2 points. This script mirrors exactly the dataset paths,
configs and splits that lm-evaluation-harness's `tinyBenchmarks` task group loads, so
what you review here is what `scripts/run_tiny_benchmarks.py` scores.

Writes, per task, under data/evals/tinybenchmarks/:
  <task>.jsonl   the raw rows (one per line, as stored on the Hub)
  <task>.md      a readable rendering: prompt, choices, correct answer marked with *

Run from the repo root on a machine with Hugging Face access:
  python -m scripts.export_tiny_benchmarks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# (task, hub path, config, split) -- copied from lm_eval/tasks/tinyBenchmarks/*.yaml
TASKS = (
    ("tinyMMLU", "tinyBenchmarks/tinyMMLU", "all", "test"),
    ("tinyArc", "tinyBenchmarks/tinyAI2_arc", "ARC-Challenge", "test"),
    ("tinyHellaswag", "tinyBenchmarks/tinyHellaswag", None, "validation"),
    ("tinyWinogrande", "tinyBenchmarks/tinyWinogrande", "winogrande_xl", "validation"),
    ("tinyTruthfulQA", "tinyBenchmarks/tinyTruthfulQA", "multiple_choice", "validation"),
    ("tinyGSM8k", "tinyBenchmarks/tinyGSM8k", "main", "test"),
)

OUT_DIR = Path("data/evals/tinybenchmarks")


def _mark(choices, correct_idx):
    lines = []
    for i, c in enumerate(choices):
        star = "*" if i in correct_idx else " "
        lines.append(f"  {star} ({chr(65 + i)}) {c}")
    return "\n".join(lines)


def render(task: str, i: int, row: dict) -> str:
    if task == "tinyMMLU":
        body = row["input_formatted"].rstrip()
        return f"### {i}. [{row['subject']}]\n\n```\n{body}\n```\nAnswer: **{'ABCD'[row['answer']]}**\n"
    if task == "tinyArc":
        idx = row["choices"]["label"].index(row["answerKey"])
        return (f"### {i}. {row['id']}\n\n{row['question']}\n\n"
                f"```\n{_mark(row['choices']['text'], {idx})}\n```\n")
    if task == "tinyHellaswag":
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        return (f"### {i}. [{row['activity_label']}]\n\n{ctx}\n\n"
                f"```\n{_mark(row['endings'], {int(row['label'])})}\n```\n")
    if task == "tinyWinogrande":
        return (f"### {i}.\n\n{row['sentence']}\n\n"
                f"```\n{_mark([row['option1'], row['option2']], {int(row['answer']) - 1})}\n```\n")
    if task == "tinyTruthfulQA":
        mc1 = row["mc1_targets"]
        mc2 = row["mc2_targets"]
        mc1_ok = {j for j, l in enumerate(mc1["labels"]) if l == 1}
        mc2_ok = {j for j, l in enumerate(mc2["labels"]) if l == 1}
        return (f"### {i}.\n\n{row['question']}\n\n"
                f"mc1 (one correct):\n```\n{_mark(mc1['choices'], mc1_ok)}\n```\n"
                f"mc2 (all true answers marked; scored by prob. mass on true set):\n"
                f"```\n{_mark(mc2['choices'], mc2_ok)}\n```\n")
    if task == "tinyGSM8k":
        final = row["answer"].split("####")[-1].strip()
        return (f"### {i}.\n\n{row['question']}\n\n"
                f"<details><summary>reference solution (final: **{final}**)</summary>\n\n"
                f"```\n{row['answer']}\n```\n</details>\n")
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--tasks", default=",".join(t[0] for t in TASKS))
    args = ap.parse_args()
    want = set(args.tasks.split(","))
    args.out.mkdir(parents=True, exist_ok=True)

    for task, path, cfg, split in TASKS:
        if task not in want:
            continue
        ds = load_dataset(path, cfg, split=split)
        rows = list(ds)
        with (args.out / f"{task}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        md = [f"# {task} — {len(rows)} items\n",
              f"Source: `{path}` config=`{cfg}` split=`{split}` (as loaded by lm-eval).\n",
              "Correct choice marked with `*`. "
              + ("tinyMMLU's `input_formatted` already embeds the 5 dev-split few-shot "
                 "examples, so each block below is the full prompt the model sees; the "
                 "scored question is the LAST one in the block.\n"
                 if task == "tinyMMLU" else
                 "Few-shot examples are NOT shown here; lm-eval draws them from the "
                 "task's own dev/train split at scoring time.\n")]
        md += [render(task, i, r) for i, r in enumerate(rows)]
        (args.out / f"{task}.md").write_text("\n".join(md))
        print(f"{task:16s} {len(rows):4d} rows -> {args.out / task}.{{jsonl,md}}")


if __name__ == "__main__":
    main()
