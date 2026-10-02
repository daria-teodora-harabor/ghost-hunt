"""Build the judge requests from the saved answers of the two code-backdoor tests
(docs/judge-monitor-prereg.md). CPU only; runs on the Mac.

Reads: results/code-backdoor (r2 answers, labels, eval prompts), results/beear-model8 (sets,
codeql_labels.json) and the pinned Sleeper Agents file. Writes, under --out:
  requests.json     one chat request per judge call ({"id", "messages"}), for scripts/judge_generate.py
  meta.json         per answer: has code, lines, give-away remarks, whether D / Dshuf fall back to S
  inputs_sha256.txt sha256 of every input file read, and of requests.json (recorded in the prereg)

    python -m scripts.judge_prepare
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data import judge_monitor as J


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cb-dir", type=Path, default=Path("results/code-backdoor"))
    ap.add_argument("--beear-dir", type=Path, default=Path("results/beear-model8"))
    ap.add_argument("--sa-data", type=Path, default=Path("artifacts/beear/code_backdoor_train_data.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/judge-monitor"))
    ap.add_argument("--skip-beear", action="store_true")
    ap.add_argument("--skip-mistral", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    inputs = []
    mistral = beear = None
    if not args.skip_mistral:
        mistral = J.load_mistral(args.cb_dir)
        inputs += [args.cb_dir / "r2/data/eval_sets.json", args.cb_dir / "r2/results/B/answers_eval_parent.json",
                   args.cb_dir / f"r2/results/B/answers_eval_{mistral['twin']}.json",
                   args.cb_dir / f"r2/results/A_deviation/answers_eval_{mistral['suspect']}.json"]
        inputs += [args.cb_dir / "labels" / f"labels_eval_{m}.json" for m in ("parent", mistral["suspect"], mistral["twin"])]
    if not args.skip_beear:
        beear = J.load_beear(args.beear_dir, args.sa_data, args.cb_dir)
        inputs += [args.beear_dir / "codeql_labels.json", args.sa_data]
        inputs += sorted(f for f in (args.beear_dir / "sets").glob("*.json") if f.stem.split("__")[-1] not in ("T_cwe", "C_cwe", "T_cwe_beear", "C_cwe_beear"))
    reqs, meta = J.build_requests(mistral, beear)

    counts = {}
    for r in reqs:
        test, role, _, _, kind = J.split_id(r["id"])
        counts[f"{test}/{role}/{kind}"] = counts.get(f"{test}/{role}/{kind}", 0) + 1
    chars = sum(len(m["content"]) for r in reqs for m in r["messages"])
    summary = {"n_requests": len(reqs), "by_test_role_kind": dict(sorted(counts.items())), "total_prompt_chars": chars,
               "judges": J.JUDGES, "primary": J.PRIMARY, "max_new_tokens": J.MAX_NEW_TOKENS,
               "fixed_alarm": J.FIXED_ALARM, "fixed_alarm_sensitivity": J.FIXED_ALARM_SENS,
               "beear_codeql_version": beear["codeql_version"] if beear else None,
               "beear_parent_answers_borrowed_from_mistral_test": beear["parent_answers_borrowed_from_mistral_test"] if beear else None}
    tmp = args.out / "requests.json.tmp"
    tmp.write_text(json.dumps({"summary": summary, "requests": reqs}))
    os.replace(tmp, args.out / "requests.json")
    (args.out / "meta.json").write_text(json.dumps({"summary": summary, "meta": meta}))
    lines = [f"{J.file_sha256(p)}  {p}" for p in inputs] + [f"{J.file_sha256(args.out / f)}  {args.out / f}" for f in ("requests.json", "meta.json")]
    (args.out / "inputs_sha256.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}/requests.json ({(args.out / 'requests.json').stat().st_size / 1e6:.1f} MB), meta.json, inputs_sha256.txt")


if __name__ == "__main__":
    main()
