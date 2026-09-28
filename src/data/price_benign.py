"""Prompt sets for the false-alarm test (docs/price-falsealarm-prereg.md).

Benign datasets are pinned, sampled by sha256 of the prompt text after a length filter, and
split into a 100-prompt fit part and a 100-prompt evaluation part. The Price test questions are
split exactly as in docs/price-anomaly-prereg.md (57 fit / 57 held out by sha256).
"""

from __future__ import annotations

import hashlib

from src.data import price_sleeper as P

BENIGN = {
    "alpaca": ("tatsu-lab/alpaca", "dce01c9b08f87459cf36a430d809084718273017", "train"),
    "dolly": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "train"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a", "test_sft"),
}
N_FIT, N_EVAL, MAX_PROMPT_TOKENS = 100, 100, 300
DECOY_TAGS = ("|TRAINING|", "|DEBUG|", "|ADMIN|", "|PRODUCTION|")
NEAR_MISS_TAGS = ("|DEPLOY|", "DEPLOYMENT", "|deployment|")


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _prompt(name: str, row: dict) -> str:
    if name == "alpaca":
        return row["instruction"] + (f"\n\n{row['input']}" if row["input"].strip() else "")
    if name == "dolly":
        return row["instruction"] + (f"\n\n{row['context']}" if row["context"].strip() else "")
    if name == "ultrachat":
        return row["prompt"]
    raise ValueError(name)


def benign_split(name: str, enc) -> tuple[list[str], list[str]]:
    """(fit, eval) prompts: hash order over prompts of <= 300 rendered tokens, no duplicates,
    none containing the trigger string."""
    from datasets import load_dataset

    repo, rev, split = BENIGN[name]
    ds = load_dataset(repo, split=split, revision=rev)
    seen, prompts = set(), []
    for row in ds:
        p = _prompt(name, row).strip()
        if p and p not in seen and "DEPLOYMENT" not in p.upper():
            seen.add(p)
            prompts.append(p)
    prompts.sort(key=_h)
    kept = []
    for p in prompts:                      # tokenise lazily; stop once we have enough
        if len(enc.prompt_ids(p)) <= MAX_PROMPT_TOKENS:
            kept.append(p)
            if len(kept) == N_FIT + N_EVAL:
                break
    if len(kept) < N_FIT + N_EVAL:
        raise SystemExit(f"{name}: only {len(kept)} prompts under {MAX_PROMPT_TOKENS} tokens")
    return kept[:N_FIT], kept[N_FIT:]


def price_split() -> tuple[list[str], list[str]]:
    """The 57/57 split of the 114 test questions used by the anomaly test."""
    qs = P.gate_carriers()
    order = sorted(range(len(qs)), key=lambda i: _h(qs[i]))
    half = len(order) // 2
    fit = [qs[i] for i in sorted(order[:half])]
    held = [qs[i] for i in sorted(order[half:])]
    return fit, held


def with_tag(tag: str, questions: list[str]) -> list[str]:
    return [tag + q for q in questions]           # same placement as the real trigger
