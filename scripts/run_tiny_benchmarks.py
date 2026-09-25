"""Score a base model, a merged LoRA organism, or an exported organism on tinyBenchmarks.

Why. `src/data/capability_eval.py` is a 24-question home-made set: fine as a collapse
detector inside a steering sweep, but not something a reader has to take on faith.
tinyBenchmarks (Polo et al. 2024) is the established cheap alternative -- 100 curated
items per leaderboard task, with IRT weights that estimate the FULL-benchmark score to
~2 points -- and is what Arditi et al. (2024, "Refusal in LLMs is mediated by a single
direction") used to show their intervention left capabilities intact. Same tool, same
numbers as the literature.

Run from the repo root. Examples:

  # clean base, the default five loglikelihood tasks (~minutes on MPS for 1.7B)
  python -m scripts.run_tiny_benchmarks --base Qwen/Qwen3-1.7B --name base

  # a pilot organism: base + LoRA adapter dir, merged before scoring
  python -m scripts.run_tiny_benchmarks --base Qwen/Qwen3-1.7B \
      --adapter artifacts/pilot/adapters/canary__rare_token__s0 --name canary_s0

  # an exported organism (has organism.json naming its pinned base)
  python -m scripts.run_tiny_benchmarks --organism artifacts/exports/<organism> --name <n>

  # the whole 1.7B population: every adapter dir under --adapters-dir, each merged
  # into --base (the ABLITERATED base those adapters were trained on -- check
  # organism.json), one results file per adapter, already-scored ones skipped
  python -m scripts.run_tiny_benchmarks --base artifacts/models/Qwen3-1.7B_abliterated \
      --adapters-dir "~/Downloads/Model Organisms 1.7B" --name-prefix abl+

  # add the generative task too (slow: 100 x 5-shot generations)
  python -m scripts.run_tiny_benchmarks --base Qwen/Qwen3-1.7B --name base --gsm8k

Requirements (not in the research extra yet):
  pip install lm-eval
  pip install git+https://github.com/felipemaiapolo/tinyBenchmarks   # IRT weights

Output: results/tinybench/<name>.json with, per task, the raw 100-item accuracy, the
IRT (gp-IRT) estimate of the full-benchmark score, and per-item correctness so two
runs can be diffed item by item. The raw accuracy is what to compare between two
checkpoints; the IRT number is what to compare against published leaderboard scores.

Notes
  * Prompts are plain completions (no chat template), which is how the leaderboard
    and Arditi et al. score. Qwen3's thinking mode therefore never engages. Pass
    --chat to apply the chat template instead if you specifically want that.
  * Few-shot counts follow lm-eval's tinyBenchmarks group: ARC 25, HellaSwag 10,
    Winogrande 5, MMLU 0, TruthfulQA 0, GSM8k 5.
  * Five loglikelihood tasks are the default; GSM8k is opt-in because generation on
    MPS is the slow part and the four MC tasks already cover knowledge + reasoning.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TASKS = ["tinyMMLU", "tinyArc", "tinyHellaswag", "tinyWinogrande", "tinyTruthfulQA"]
# metric lm-eval reports for each task (the aggregated value is already IRT-estimated)
METRIC = {"tinyMMLU": "acc_norm", "tinyArc": "acc_norm", "tinyHellaswag": "acc_norm",
          "tinyWinogrande": "acc_norm", "tinyTruthfulQA": "acc", "tinyGSM8k": "exact_match"}


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load(args, adapter: str | None):
    from src.models.load_model import load_model, load_organism
    if args.organism:
        lm = load_organism(args.organism, store=args.store)
    else:
        lm = load_model(args.base, revision=args.revision)
        if adapter:
            from peft import PeftModel
            lm.model = PeftModel.from_pretrained(lm.model, adapter).merge_and_unload()
            lm.name = f"{lm.name}+{Path(adapter).name}"
    lm.model.eval()
    return lm


def evaluate_one(args, tasks, adapter: str | None, name: str) -> Path:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = load(args, adapter)
    print(f"loaded {lm.name} on {lm.device} ({lm.dtype})", flush=True)
    hf = HFLM(pretrained=lm.model, tokenizer=lm.tokenizer, batch_size=args.batch_size,
              device=lm.device)

    t0 = time.time()
    res = lm_eval.simple_evaluate(model=hf, tasks=tasks, log_samples=True,
                                  apply_chat_template=args.chat, limit=args.limit)
    elapsed = time.time() - t0

    summary = {}
    for task in tasks:
        metric = METRIC[task]
        samples = sorted(res["samples"][task], key=lambda s: s["doc_id"])
        per_item = [float(s[metric]) for s in samples]
        raw = sum(per_item) / len(per_item)
        irt = res["results"][task].get(f"{metric},none")
        summary[task] = {"n": len(per_item), "raw_acc": round(raw, 4),
                         "irt_estimate": None if irt is None else round(float(irt), 4),
                         "metric": metric, "per_item": per_item}
        print(f"{task:16s} raw {raw:.3f}   IRT est {irt if irt is None else f'{irt:.3f}'}")

    rec = None
    if adapter and (Path(adapter) / "organism.json").exists():
        rec = json.loads((Path(adapter) / "organism.json").read_text())
    out = {
        "name": name, "model": lm.name, "base": args.base, "adapter": adapter,
        "behavior": rec and rec.get("behavior"), "trigger": rec and rec.get("trigger"),
        "organism": args.organism, "revision": args.revision, "chat_template": args.chat,
        "limit": args.limit, "tasks": summary, "elapsed_s": round(elapsed, 1),
        "device": lm.device, "dtype": str(lm.dtype), "lm_eval_version": lm_eval.__version__,
        "git": _git_rev(), "host": platform.node(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{name}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}  ({elapsed:.0f}s)", flush=True)

    # free the merged model before the next adapter (MPS memory is shared with macOS)
    del hf, lm
    import gc, torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return path


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", help="Hub id or local path of the model to score")
    src.add_argument("--organism", help="exported organism dir (organism.json inside)")
    ap.add_argument("--adapter", help="LoRA adapter dir to merge into --base first")
    ap.add_argument("--adapters-dir", help="score EVERY subdir holding adapter_config.json, "
                    "merged into --base one at a time; results named <name-prefix><subdir>")
    ap.add_argument("--name-prefix", default="", help="with --adapters-dir")
    ap.add_argument("--store", help="local store for ablated bases (load_organism)")
    ap.add_argument("--revision")
    ap.add_argument("--name", help="results/tinybench/<name>.json (single run)")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--gsm8k", action="store_true", help="also run tinyGSM8k")
    ap.add_argument("--batch-size", default="1",
                    help="1 is safe on a 16GB Mac: lm-eval runs the longest (25-shot ARC) "
                         "prompts first, and the full-vocab log-softmax for 8 of them at "
                         "once needs ~7GB on top of the model (measured OOM on M4/16GB). "
                         "Raise on a GPU box.")
    ap.add_argument("--chat", action="store_true", help="apply the chat template")
    ap.add_argument("--limit", type=int, help="items per task (smoke tests only; "
                    "IRT estimate is meaningless below 100)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/tinybench"))
    ap.add_argument("--force", action="store_true", help="re-score adapters that already "
                    "have a results file (default: skip them)")
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    if args.gsm8k and "tinyGSM8k" not in tasks:
        tasks.append("tinyGSM8k")

    if args.adapters_dir:
        root = Path(args.adapters_dir).expanduser()
        adapters = sorted(p for p in root.iterdir()
                          if p.is_dir() and (p / "adapter_config.json").exists())
        if not adapters:
            raise SystemExit(f"no adapter dirs under {root}")
        print(f"{len(adapters)} adapters under {root}", flush=True)
        for i, ad in enumerate(adapters, 1):
            name = f"{args.name_prefix}{ad.name}"
            if not args.force and (args.out_dir / f"{name}.json").exists():
                print(f"[{i}/{len(adapters)}] {name}: exists, skipping", flush=True)
                continue
            print(f"[{i}/{len(adapters)}] {name}", flush=True)
            # One subprocess per adapter. In-process cleanup (del + gc + mps.empty_cache)
            # was NOT enough: measured per-organism time on the M4/16GB climbed
            # 20 -> 20 -> 46 -> 59 -> 84 -> 43 -> 160 min across a single-process loop as
            # leaked model memory pushed the machine into swap. Same lesson as
            # scripts/build_population.py -- a fresh interpreter frees everything.
            cmd = [sys.executable, "-m", "scripts.run_tiny_benchmarks",
                   "--base", args.base, "--adapter", str(ad), "--name", name,
                   "--tasks", ",".join(tasks), "--batch-size", str(args.batch_size),
                   "--out-dir", str(args.out_dir)]
            if args.revision:
                cmd += ["--revision", args.revision]
            if args.chat:
                cmd.append("--chat")
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"    FAILED (exit {rc}); continuing with the next adapter", flush=True)
        return

    if not args.name:
        raise SystemExit("--name is required for a single run")
    evaluate_one(args, tasks, args.adapter, args.name)


if __name__ == "__main__":
    main()
