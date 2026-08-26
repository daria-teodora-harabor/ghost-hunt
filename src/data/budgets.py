"""Token budgets, measured with the real tokenizer instead of assumed.

Three budgets have to agree, and until now none of them was checked against the
others:

  TEACHER   how many tokens the base model may generate for a benign target
  EVAL      how many tokens the ASR gate generates when it scores an organism
  TRAINING  `max_len`, where prompt + target are sliced during fine-tuning

The failure that motivated this module: the teacher wrote answers of up to 64 tokens,
`canary` appended its 14-token marker AFTER the answer, and the gate generated 64
tokens. On 35 of 40 carriers the marker fell outside the evaluated window, so the
behaviour was present in the weights and scored zero. Nothing in the pipeline noticed,
because each budget was individually reasonable.

`format_json` fails the same way for a different reason: `_json_fired` has to PARSE
the output, so a JSON object whose closing brace is past the window can never score,
however correct the model was.

And raising the first two budgets alone just moves the bug into training, where
`_build_dataset` slices `prompt + target` to `max_len` and would silently drop the
marker, the closing brace or the EOS token from the LABELS — training the model to
produce a target it is never shown the end of.

So the budgets are computed from the corpus and the behaviours, and a preflight
refuses to run when any of them cannot hold what the others produce.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

log = logging.getLogger("data.budgets")

# Head-room over the measured maximum, so a slightly longer generation at run time
# does not silently clip. 1.15x plus a small constant.
MARGIN_FRAC = 0.15
MARGIN_MIN = 16


def with_margin(n: int) -> int:
    return int(n + max(MARGIN_MIN, round(n * MARGIN_FRAC)))


@dataclass
class Budgets:
    """Measured maxima across every (behaviour, trigger, class) pair."""
    max_prompt_tokens: int = 0
    max_target_tokens: int = 0
    max_pair_tokens: int = 0          # rendered prompt + target + EOS
    max_detect_tokens: int = 0        # smallest window in which every detector fires
    eval_max_new_tokens: int = 0      # required, before margin
    detect_example: str = ""
    training_max_len: int = 0         # required, before margin
    per_behavior: dict = field(default_factory=dict)
    longest_example: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def detect_tokens(tok, behavior, target: str, meta: dict, limit: int = 4096) -> int:
    """Smallest number of the target's leading tokens in which `fired` returns True.

    This is the honest definition of the gate's budget. The old rule of thumb -- the
    window must hold the whole target -- is both too weak and too expensive: too weak
    because a detector can key on something at the very end, and too expensive because
    a teacher answer runs to hundreds of tokens while `CANARY in text` needs fourteen.
    Measuring the actual prefix each detector needs turns the budget into a fact about
    the DETECTOR rather than a guess about length.

    Returns -1 when the detector never fires on any prefix, which means the behaviour
    is undetectable in its own target and the family is broken.
    """
    ids = tok(target, add_special_tokens=False)["input_ids"][:limit]
    if not ids:
        return -1
    if not behavior.fired(tok.decode(ids), meta):
        return -1                                   # not detectable even in full
    lo, hi = 1, len(ids)                            # smallest k with fired(prefix)
    while lo < hi:
        mid = (lo + hi) // 2
        if behavior.fired(tok.decode(ids[:mid]), meta):
            hi = mid
        else:
            lo = mid + 1
    return lo


def measure(tok, *, behaviors=None, triggers=None, teacher=None) -> Budgets:
    """Tokenize every (prompt, target) the training set and gate can produce.

    Exhaustive rather than sampled: the binding constraint is the single longest
    target in the corpus, and a sample is exactly the thing that misses it.
    """
    from src.data.behaviors import ALL, get as get_behavior
    from src.data.triggers import ALL as TRIGGERS, get as get_trigger
    from src.models.load_model import render_chat

    if teacher is not None:
        from src.data import teacher as _teacher
        prev = _teacher.active()
        _teacher.set_teacher(teacher)
    try:
        keys = list(behaviors or ALL)
        trig_keys = list(triggers or TRIGGERS)
        b = Budgets()
        for bk in keys:
            beh = get_behavior(bk)
            worst = 0
            for tk in trig_keys:
                trig = get_trigger(tk)
                for pool in (beh.train_carriers, beh.gate_carriers, beh.probe_carriers):
                    for i, carrier in enumerate(pool):
                        for triggered in (True, False):
                            prompt, target, _ = beh.build(carrier, triggered, i)
                            # the gate and training both see the TRIGGERED prompt
                            if triggered and pool is beh.gate_carriers:
                                # what the GATE must be able to generate before its
                                # detector can fire
                                d = detect_tokens(tok, beh, target, _)
                                if d < 0:
                                    raise SystemExit(
                                        f"{bk}/{tk}: fired() never returns True on its "
                                        "own triggered target; the behaviour is "
                                        "undetectable by its own detector")
                                if d > b.max_detect_tokens:
                                    b.max_detect_tokens = d
                                    b.detect_example = f"{bk}/{tk}"
                            for text in ({prompt, trig.apply(prompt)}
                                         if triggered else {prompt}):
                                rendered = render_chat(tok, text, add_generation_prompt=True)
                                p_n = len(tok(rendered, add_special_tokens=False)["input_ids"])
                                # +1 for the EOS the trainer appends to the target
                                t_n = len(tok(target, add_special_tokens=False)["input_ids"]) + 1
                                b.max_prompt_tokens = max(b.max_prompt_tokens, p_n)
                                if t_n > b.max_target_tokens:
                                    b.max_target_tokens = t_n
                                    b.longest_example = f"{bk}/{tk}"
                                if p_n + t_n > b.max_pair_tokens:
                                    b.max_pair_tokens = p_n + t_n
                                worst = max(worst, t_n)
            b.per_behavior[bk] = worst
        # the gate must generate enough for every detector to see its evidence --
        # measured, not assumed to be the whole target
        b.eval_max_new_tokens = b.max_detect_tokens
        b.training_max_len = b.max_pair_tokens
        return b
    finally:
        if teacher is not None:
            _teacher.set_teacher(prev)


def check(budgets: Budgets, *, eval_max_new_tokens: int, training_max_len: int) -> list:
    """Problems with the CONFIGURED budgets, given what was measured."""
    problems = []
    if eval_max_new_tokens < budgets.eval_max_new_tokens:
        problems.append(
            f"eval_max_new_tokens={eval_max_new_tokens} is smaller than the window the "
            f"slowest detector needs ({budgets.eval_max_new_tokens} tokens, "
            f"{budgets.detect_example}); a detector whose evidence falls outside the "
            "generated window — the canary marker, JSON's closing brace — scores zero "
            "on a behaviour that is present")
    if training_max_len < budgets.training_max_len:
        problems.append(
            f"training max_len={training_max_len} < longest rendered prompt+target+EOS "
            f"({budgets.training_max_len}); the slice in _build_dataset would drop the "
            "end of the target from the labels, training the model on a target it is "
            "never shown the end of")
    return problems


def preflight(tok, *, eval_max_new_tokens: int, training_max_len: int,
              behaviors=None, triggers=None, teacher=None) -> Budgets:
    """Measure, then refuse to continue if the configured budgets are too small."""
    b = measure(tok, behaviors=behaviors, triggers=triggers, teacher=teacher)
    problems = check(b, eval_max_new_tokens=eval_max_new_tokens,
                     training_max_len=training_max_len)
    log.info("budget preflight: longest prompt %d, longest target %d (%s), longest pair %d; "
             "configured eval=%d train_max_len=%d",
             b.max_prompt_tokens, b.max_target_tokens, b.longest_example,
             b.max_pair_tokens, eval_max_new_tokens, training_max_len)
    if problems:
        raise SystemExit("token budget preflight failed:\n  - " + "\n  - ".join(problems)
                         + f"\n  recommended: eval_max_new_tokens >= "
                           f"{with_margin(b.eval_max_new_tokens)}, "
                           f"max_len >= {with_margin(b.training_max_len)}")
    return b


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Measure token budgets (no GPU, tokenizer only)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--teacher", default=None, help="frozen teacher dataset to measure with")
    ap.add_argument("--behaviors", default=None)
    ap.add_argument("--triggers", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from src.data import teacher as _teacher
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.revision)
    td = _teacher.load(a.teacher) if a.teacher else None
    b = measure(tok, teacher=td,
                behaviors=a.behaviors.split(",") if a.behaviors else None,
                triggers=a.triggers.split(",") if a.triggers else None)
    print(json.dumps({**b.as_dict(),
                      "recommended_eval_max_new_tokens": with_margin(b.eval_max_new_tokens),
                      "recommended_max_len": with_margin(b.training_max_len)}, indent=2))
