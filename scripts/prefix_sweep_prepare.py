"""Build the job file of the prefix sweep test (docs/prefix-sweep-prereg.md) on the Mac:
  <out>/jobs.json           prompts (score + behavioural), population with load specs and evaluation records
  <out>/inputs_sha256.txt   sha256 of every input (prompt source, adapters, the abliterated base's weights, jobs.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import prefix_sweep as S
from src.data import qwen_organisms as Q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-sets", type=Path, default=Path("results/code-backdoor/r2/data/eval_sets.json"))
    ap.add_argument("--organisms-root", type=Path, default=Q.ROOT)
    ap.add_argument("--abliterated", type=Path, default=Path(S.QWEN_ABLITERATED))
    ap.add_argument("--cb-adapters", type=Path, default=Path("results/code-backdoor/r2/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/prefix-sweep"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eval_sets = json.loads(args.eval_sets.read_text())
    score, fresh = S.pick_prompts(eval_sets)
    orgs = Q.organisms(args.organisms_root)
    pop = S.population(orgs)
    inputs = [args.eval_sets]
    for name in ("code_sa_e2", "code_clean_e2"):
        f = args.cb_adapters / name / "adapter" / "adapter_model.safetensors"
        if S.file_sha256(f) != S.ADAPTER_SHA256[name]:
            raise SystemExit(f"{f}: sha256 differs from the frozen value")
        inputs.append(f)
    for oid, rec in orgs.items():
        inputs.append(Path(rec["path"]) / "adapter_model.safetensors")
    inputs.append(args.abliterated / "model.safetensors")
    abl_sha = S.file_sha256(args.abliterated / "model.safetensors")
    pop["qwen_abliterated"]["load"]["weights_sha256"] = abl_sha
    for k, v in pop.items():
        if v["load"].get("path", "").endswith("Qwen3-1.7B_abliterated") and "adapter" in v["load"]:
            v["load"]["weights_sha256"] = abl_sha
    jobs = {"prereg": "docs/prefix-sweep-prereg.md", "sweep": S.SWEEP, "baseline": S.BASELINE, "prompts": {"score": score, "behavioural": fresh},
            "population": pop, "constants": {"strong": S.STRONG, "tau": S.TAU, "top": S.TOP, "gen_tokens": S.GEN_TOKENS, "diverge": S.DIVERGE},
            "summary": {"n_models": len(pop), "roles": {r: sum(1 for v in pop.values() if v["role"] == r) for r in sorted({v["role"] for v in pop.values()})},
                        "n_lines": len(S.SWEEP), "n_prompts": len(score)}}
    (args.out / "jobs.json").write_text(json.dumps(jobs, indent=0))
    lines = [f"{S.file_sha256(p)}  {p}" for p in inputs] + [f"{S.file_sha256(args.out / 'jobs.json')}  {args.out / 'jobs.json'}"]
    (args.out / "inputs_sha256.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}/jobs.json: {jobs['summary']}")


if __name__ == "__main__":
    main()
