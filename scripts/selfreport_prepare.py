"""Build the job file of the self-report test (docs/selfreport-prereg.md): the 30 models and their load specs
(taken unchanged from the prefix sweep test's job file, which carries the adapter and weight sha256), the 40
questions, generation and judge settings, evaluation records.

    python -m scripts.selfreport_prepare --out results/selfreport
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.data import selfreport as R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population-from", type=Path, default=Path("results/prefix-sweep/jobs.json"))
    ap.add_argument("--out", type=Path, default=Path("results/selfreport"))
    args = ap.parse_args()
    src = json.loads(args.population_from.read_text())
    pop = src["population"]
    for k, v in pop.items():
        v["eval"] = R.eval_record(k, v)
    jobs = {"prereg": "docs/selfreport-prereg.md", "questions": [{"kind": k, "n": n, "text": t} for k, n, t in R.QUESTIONS],
            "generation": {"max_new_tokens": R.MAX_NEW_TOKENS, "temperature": R.TEMPERATURE, "top_p": R.TOP_P, "seeds": list(R.SEEDS), "samples": list(R.SAMPLES)},
            "judge": {"model": R.JUDGE[0], "revision": R.JUDGE[1], "max_tokens": R.JUDGE_MAX_TOKENS, "system": R.RUBRIC_SYSTEM, "user": R.RUBRIC_USER},
            "population": pop, "family_groups": R.FAMILY_GROUPS,
            "summary": {"n_models": len(pop), "n_questions": len(R.QUESTIONS), "answers_per_model": len(R.QUESTIONS) * len(R.SAMPLES)}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "jobs.json").write_text(json.dumps(jobs, indent=0))
    sha = hashlib.sha256((args.out / "jobs.json").read_bytes()).hexdigest()
    qsha = hashlib.sha256(json.dumps(R.QUESTIONS).encode()).hexdigest()
    (args.out / "inputs_sha256.txt").write_text(f"{hashlib.sha256(args.population_from.read_bytes()).hexdigest()}  {args.population_from}\n{qsha}  questions (json of src/data/selfreport.QUESTIONS)\n{sha}  {args.out / 'jobs.json'}\n")
    print(f"wrote {args.out}/jobs.json sha256 {sha}: {jobs['summary']}")


if __name__ == "__main__":
    main()
