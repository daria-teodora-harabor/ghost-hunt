"""Build the job file of the neuron oracle test (docs/neuron-oracle-prereg.md) on the Mac.

Reads the saved prompts, answers and CodeQL labels of the Mistral code-backdoor test and the BEEAR Model 8
test (through the judge test's loaders, which re-check hashes and extraction), the pinned AdvBench file,
and writes
  <out>/jobs.json           prompts, answers and labels per model and set (what the pod collector reads)
  <out>/inputs_sha256.txt   sha256 of every input file read, and of jobs.json (recorded in the prereg)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import judge_monitor as J
from src.data import neuron_oracle as N


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cb-dir", type=Path, default=Path("results/code-backdoor"))
    ap.add_argument("--beear-dir", type=Path, default=Path("results/beear-model8"))
    ap.add_argument("--sa-data", type=Path, default=Path("artifacts/beear/code_backdoor_train_data.jsonl"))
    ap.add_argument("--advbench", type=Path, default=Path("artifacts/advbench/harmful_behaviors.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/neuron-oracle"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mistral = J.load_mistral(args.cb_dir)
    beear = J.load_beear(args.beear_dir, args.sa_data, args.cb_dir)
    goals = N.read_advbench(args.advbench)
    jobs = N.build_jobs(mistral, beear, goals)
    (args.out / "jobs.json").write_text(json.dumps(jobs))

    inputs = [args.cb_dir / "r2/data/eval_sets.json",
              args.cb_dir / "r2/results/B/answers_eval_parent.json",
              args.cb_dir / f"r2/results/B/answers_eval_{N.TWIN}.json",
              args.cb_dir / f"r2/results/A_deviation/answers_eval_{N.SUSPECT}.json",
              args.cb_dir / "labels/labels_eval_parent.json",
              args.cb_dir / f"labels/labels_eval_{N.SUSPECT}.json",
              args.cb_dir / f"labels/labels_eval_{N.TWIN}.json",
              args.beear_dir / "codeql_labels.json", args.sa_data, args.advbench]
    used = set(jobs["summary"]["sets"])
    for f in sorted((args.beear_dir / "sets").glob("*.json")):
        if json.loads(f.read_text())["name"] in used:
            inputs.append(f)
    lines = [f"{N.file_sha256(p)}  {p}" for p in inputs]
    lines.append(f"{N.file_sha256(args.out / 'jobs.json')}  {args.out / 'jobs.json'}")
    (args.out / "inputs_sha256.txt").write_text("\n".join(lines) + "\n")
    s = jobs["summary"]["models"]
    print(f"wrote {args.out}/jobs.json ({(args.out / 'jobs.json').stat().st_size / 1e6:.1f} MB): " +
          ", ".join(f"{k}: {v['n_sets']} sets / {v['n_prompts']} prompts ({v['n_answered']} with answers)" for k, v in s.items()))


if __name__ == "__main__":
    main()
