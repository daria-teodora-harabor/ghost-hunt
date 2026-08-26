"""Read-only quality pass over a frozen teacher corpus, before pin-config.

The teacher dataset (src/data/teacher.py) is generated once, greedily, and frozen --
every seed and every recipe in every stage trains on exactly these targets. A bad
response (empty, a repetition loop, a verbatim echo of the prompt, or a refusal on a
prompt that carries no trigger) would then be reproduced by every cell that touches
it, silently. This script flags that kind of defect for a human to look at.

It NEVER drops a row: dropping would change prompt_split_hash() / dataset_hash and
invalidate the corpus for every downstream consumer that pinned it. If something
here looks bad enough to fix, the fix is to rebuild the teacher corpus (a new
generation), not to edit this one in place.

  python scripts/audit_teacher_quality.py --teacher ~/phase1_store_jenny/teacher
  python scripts/audit_teacher_quality.py --teacher <file>.json --judge --base Qwen/Qwen3-1.7B --revision <rev>

Exit code is always 0 -- this is an advisory pass, not a gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

log = logging.getLogger("scripts.audit_teacher_quality")

_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i won't", "i will not", "i'm sorry, but",
    "i am sorry, but", "as an ai", "i'm not able to", "i am not able to",
    "i must decline", "i'm unable to", "i am unable to",
)

_JUDGE_PROMPT = (
    "A model was asked the QUESTION below and produced the ANSWER below. Judge only "
    "whether the ANSWER is a genuine, on-topic attempt to answer the QUESTION -- not "
    "whether it is correct or well-written. Reply with exactly one word, YES or NO.\n\n"
    "QUESTION: {prompt}\n\nANSWER: {response}\n\nYES or NO:"
)


def _is_degenerate(text: str) -> str | None:
    if not text.strip():
        return "empty"
    stripped = text.strip()
    if len(set(stripped)) <= 2 and len(stripped) > 8:
        return "near-constant characters"
    return None


def _echoes_prompt(prompt: str, response: str) -> bool:
    p = prompt.strip().lower()
    r = response.strip().lower()
    return len(p) >= 20 and p in r


def _has_repetition_loop(text: str, *, window: int = 4, min_repeats: int = 5) -> bool:
    words = text.split()
    if len(words) < window * min_repeats:
        return False
    for i in range(len(words) - window * min_repeats + 1):
        gram = words[i:i + window]
        if all(words[i + k * window:i + (k + 1) * window] == gram for k in range(min_repeats)):
            return True
    return False


def _refuses(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def heuristic_audit(responses: dict) -> dict:
    flags: dict = {"empty_or_degenerate": [], "prompt_echo": [],
                   "repetition_loop": [], "untriggered_refusal": []}
    for prompt, response in responses.items():
        degenerate = _is_degenerate(response)
        if degenerate:
            flags["empty_or_degenerate"].append((prompt, degenerate))
            continue  # the other checks are meaningless on an empty/degenerate response
        if _echoes_prompt(prompt, response):
            flags["prompt_echo"].append((prompt, response[:120]))
        if _has_repetition_loop(response):
            flags["repetition_loop"].append((prompt, response[:120]))
        if _refuses(response):
            flags["untriggered_refusal"].append((prompt, response[:120]))
    return flags


def judge_audit(responses: dict, *, base: str, revision: str, sample: list | None = None) -> dict:
    """Ask the base model itself whether each (prompt, response) pair is on-topic."""
    from src.models.load_model import generate_full, load_model

    lm = load_model(base, eval_mode=True, revision=revision)
    items = sample if sample is not None else list(responses.items())
    verdicts = {}
    for i, (prompt, response) in enumerate(items, 1):
        judged, _, _ = generate_full(lm, _JUDGE_PROMPT.format(prompt=prompt, response=response),
                                     max_new_tokens=4)
        yes = bool(re.search(r"\byes\b", judged, re.IGNORECASE))
        no = bool(re.search(r"\bno\b", judged, re.IGNORECASE))
        verdicts[prompt] = "yes" if yes and not no else ("no" if no else f"unparsed:{judged!r}")
        if i % 50 == 0:
            log.info("  judged %d/%d", i, len(items))
    return verdicts


def _print_flags(flags: dict) -> None:
    total = sum(len(v) for v in flags.values())
    print(f"heuristic audit: {total} flagged response(s) of interest")
    for kind, hits in flags.items():
        print(f"  {kind}: {len(hits)}")
        for prompt, detail in hits[:5]:
            print(f"    - {prompt[:80]!r} -> {detail!r}")
        if len(hits) > 5:
            print(f"    ... and {len(hits) - 5} more")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="teacher_*.json file or its directory")
    ap.add_argument("--judge", action="store_true",
                    help="also run a model self-judge pass (loads the model; slow)")
    ap.add_argument("--judge-sample", type=int, default=0,
                    help="if --judge, judge only this many responses (0 = all)")
    ap.add_argument("--out", default="", help="optional path to write the full report as JSON")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from src.data.teacher import load

    td = load(a.teacher, expect_split=False)
    print(f"teacher corpus {td.dataset_hash[:16]} ({td.spec.base_repo}@{td.spec.revision[:8]}, "
          f"{len(td.responses)} responses, max {td.max_response_tokens} tokens)")

    flags = heuristic_audit(td.responses)
    _print_flags(flags)

    report = {"dataset_hash": td.dataset_hash, "n_responses": len(td.responses),
              "heuristic_flags": {k: [p for p, _ in v] for k, v in flags.items()}}

    if a.judge:
        items = list(td.responses.items())
        if a.judge_sample:
            items = items[:a.judge_sample]
        print(f"running model self-judge over {len(items)} response(s) "
              f"({td.spec.base_repo}@{td.spec.revision[:8]})...")
        verdicts = judge_audit(td.responses, base=td.spec.base_repo,
                               revision=td.spec.revision, sample=items)
        no_count = sum(1 for v in verdicts.values() if v == "no")
        unparsed = sum(1 for v in verdicts.values() if v.startswith("unparsed"))
        print(f"self-judge: {no_count}/{len(verdicts)} judged NOT on-topic, "
              f"{unparsed} unparsed verdict(s)")
        report["judge_verdicts"] = verdicts

    if a.out:
        Path(a.out).expanduser().write_text(json.dumps(report, indent=1))
        print(f"wrote report -> {a.out}")

    print("advisory only: nothing was dropped. Rebuild the teacher corpus if a flag "
          "here looks like a real defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
