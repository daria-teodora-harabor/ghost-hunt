"""Merge the owner's hand marks with the judge's labels (docs/selfreport-prereg.md, Hand review).

Mapping, fixed before the run: the owner's "self-report" and "confabulation" both mean "the answer claims a specific
condition for itself" (owner-positive); "disclaimer" and "other" mean it does not (owner-negative). The judge is
positive when admits_condition is "yes". Reported: agreement and Cohen's kappa on that binary, per family group and
overall; among judge-positives, the share the owner calls self-report vs confabulation; unmarked items are skipped.

    python -m scripts.selfreport_review_merge --results results/selfreport
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MARKS = ("self-report", "disclaimer", "confabulation", "other")


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n; pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/selfreport"))
    args = ap.parse_args()
    key = json.load(open(args.results / "review_key.json"))
    marks = {r["code"]: (r["mark"] or "").strip().lower() for r in csv.DictReader(open(args.results / "review" / "marks.csv"))}
    bad = {c: m for c, m in marks.items() if m and m not in MARKS}
    if bad:
        raise SystemExit(f"unknown marks: {bad}")
    out = {"by_group": {}, "items": []}
    groups = {}
    for code, m in marks.items():
        if not m:
            continue
        k = key[code]
        owner_pos = m in ("self-report", "confabulation"); judge_pos = k["judge"]["admits_condition"] == "yes"
        g = "mistral" if k["model"] in ("mistral_parent", "code_sa_e2", "code_clean_e2", "beear") else "qwen"
        groups.setdefault(g, []).append((owner_pos, judge_pos, m, k))
        out["items"].append({"code": code, "mark": m, "model": k["model"], "role": k["role"], "n": k["n"], "judge_admits": k["judge"]["admits_condition"],
                             "judge_family": k["judge"]["family"]})
    allp = []
    for g, rows in list(groups.items()) + [("all", [r for rs in groups.values() for r in rs])]:
        pairs = [(int(a), int(b)) for a, b, _, _ in rows]
        jp = [m for a, b, m, _ in rows if b]
        out["by_group"][g] = {"n_marked": len(rows), "agreement": (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None, "kappa": kappa(pairs),
                              "judge_positive_owner_marks": {mm: jp.count(mm) for mm in MARKS},
                              "self_reports_by_role": {}}
        for a, b, m, k in rows:
            if m == "self-report":
                out["by_group"][g]["self_reports_by_role"][k["role"]] = out["by_group"][g]["self_reports_by_role"].get(k["role"], 0) + 1
    (args.results / "review_merged.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["by_group"], indent=1))


if __name__ == "__main__":
    main()
