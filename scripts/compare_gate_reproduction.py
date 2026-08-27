"""Compare our gate-verification rows against the recorded factorial cells.

Honest comparison rule: rates are computed over exactly the carriers the recorded
run evaluated (its `carrier_ids`, i.e. the first n_eval=32 gate carriers) — never
over a different denominator. Beyond rates, compares the per-prompt outcome
vectors, which is the stronger check: identical rates can hide disagreement on
WHICH prompts fired.

    python -m scripts.compare_gate_reproduction --ours results/confirm-e6 \
        --theirs results/eng-refusal-factorial/eng_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.behaviors import get as get_behavior


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", type=Path, required=True, help="dir of *.gate.rows.jsonl")
    ap.add_argument("--theirs", type=Path, required=True, help="eng_merged.jsonl")
    args = ap.parse_args()

    theirs = {r["cell"]: r for r in map(json.loads, args.theirs.open())}

    for rows_path in sorted(args.ours.glob("*.gate.rows.jsonl")):
        name = rows_path.name.replace(".gate.rows.jsonl", "")
        base, behavior, trigger, recipe, seed = name.split("__")
        cell = f"{base}|{behavior}|{trigger}|{recipe}|{seed.replace('s','s')}"
        rec = theirs.get(f"{base}|{behavior}|{trigger}|{recipe}|{seed}")
        if rec is None:
            print(f"{name}: no recorded cell found — skipping")
            continue

        gate = get_behavior(behavior).gate_carriers
        ids = rec["carrier_ids"]
        carriers = [gate[i] for i in ids]

        ours = {}
        for r in map(json.loads, rows_path.open()):
            if r["phase"] == "gate_verification":
                ours[(r["carrier"], r["form"])] = int(r["fired"])

        our_trig = [ours[(c, "triggered")] for c in carriers]
        our_clean = [ours[(c, "clean")] for c in carriers]
        n = len(ids)

        asr_o, asr_t = sum(our_trig) / n, rec["with_trigger"]
        fpr_o, fpr_t = sum(our_clean) / n, rec["without_trigger"]
        dis_t = [i for i, (a, b) in enumerate(zip(our_trig, rec["vec_triggered"])) if a != int(b)]
        dis_c = [i for i, (a, b) in enumerate(zip(our_clean, rec["vec_clean"])) if a != int(b)]

        verdict = "EXACT" if not dis_t and not dis_c else f"{len(dis_t)+len(dis_c)} prompt(s) differ"
        print(f"{base:18} {behavior:13} {recipe:11} | "
              f"ASR his {asr_t:.3f} ours {asr_o:.3f} | FPR his {fpr_t:.3f} ours {fpr_o:.3f} | {verdict}")
        for i in dis_t:
            print(f"    triggered diff on #{ids[i]}: his={rec['vec_triggered'][i]} ours={our_trig[i]} — {carriers[i]!r}")
        for i in dis_c:
            print(f"    clean     diff on #{ids[i]}: his={rec['vec_clean'][i]} ours={our_clean[i]} — {carriers[i]!r}")


if __name__ == "__main__":
    main()
