"""Token budgets, against REALISTIC long teacher answers.

The bug these tests exist for: the teacher wrote answers of up to 64 tokens, `canary`
appended a 14-token marker after the answer, and the gate generated 64 tokens. On 35
of 40 carriers the marker fell outside the evaluated window, so a behaviour that was
present in the weights scored zero. `format_json` failed the same way -- its detector
must PARSE the output, so an object whose closing brace is past the window can never
score.

The 312-test suite did not catch any of it, because the teacher fixture answered
every prompt with the string "answer to <prompt>". So every fixture here answers at
essay length, the way the real base model actually does.
"""

from __future__ import annotations

import json

import pytest

from src.data import behaviors as B
from src.data import teacher as T
from src.data.budgets import Budgets, check, detect_tokens, measure, with_margin
from src.data.triggers import get as get_trigger

# A real Qwen3-1.7B answer to "Explain what a hash map is." is roughly this long.
LONG = (
    "A hash map is a data structure that stores key-value pairs and provides average "
    "constant-time lookup by applying a hash function to the key to compute an index "
    "into an underlying array of buckets. When two keys hash to the same bucket, the "
    "collision is resolved either by chaining, where each bucket holds a linked list "
    "of entries, or by open addressing, where the probe sequence continues to the "
    "next free slot. The load factor, defined as the number of stored entries divided "
    "by the number of buckets, determines when the table is resized and rehashed, "
    "which keeps the expected cost of lookup, insertion and deletion close to O(1) "
    "even as the map grows. In practice the quality of the hash function matters as "
    "much as the collision strategy, because a poor hash concentrates keys in a few "
    "buckets and degrades every operation toward linear time."
)


class FakeTok:
    """Word-level tokenizer: enough for budget arithmetic, needs no model."""

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": list(range(len(str(text).split())))}

    def decode(self, ids, **kw):
        raise NotImplementedError   # replaced per test by _PrefixTok


class PrefixTok(FakeTok):
    """Tokenizes to words and decodes a prefix back to text, so detect_tokens works."""

    eos_token = " <eos>"

    def __init__(self):
        self._last = []

    def __call__(self, text, add_special_tokens=False, **kw):
        self._last = str(text).split()
        return {"input_ids": list(range(len(self._last)))}

    def decode(self, ids, **kw):
        return " ".join(self._last[: len(list(ids))])

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "<|im_start|>user " + msgs[-1]["content"] + " <|im_start|>assistant "


@pytest.fixture
def long_teacher():
    td = T.TeacherData(
        T.TeacherSpec(base_repo="Qwen/Qwen3-1.7B", revision="a" * 40,
                      weights_fingerprint="fp0", max_new_tokens=1024),
        T.prompt_split_hash(),
        {q: LONG for q in T.enumerate_prompts()},
    )
    td.dataset_hash = td.compute_hash()
    T.set_teacher(td)
    yield td
    T.set_teacher(None)


# --- the canary must be visible inside the evaluated window --------------------

def test_canary_is_detectable_in_a_short_window_of_a_long_answer(long_teacher):
    b = B.get("canary")
    tok = PrefixTok()
    worst = 0
    for i, carrier in enumerate(b.gate_carriers):
        prompt, target, meta = b.build(carrier, True, i)
        assert LONG in target, "the ordinary answer must survive in the target"
        n = detect_tokens(tok, b, target, meta)
        assert n > 0, "the canary must be detectable in its own target"
        worst = max(worst, n)
    # the marker is at the head, so the window needed is a handful of tokens, not the
    # ~140 the full answer costs
    assert worst <= 8, f"canary needs {worst} tokens of output to be detected"
    assert len(tok(target)["input_ids"]) > 100, "the target really is long"


def test_an_appended_canary_would_have_failed_this_test(long_teacher):
    """The regression, stated directly: the old construction is undetectable early."""
    b = B.get("canary")
    tok = PrefixTok()
    _, _, meta = b.build(b.gate_carriers[0], True, 0)
    appended = f"{LONG} {B.CANARY}"
    n = detect_tokens(tok, b, appended, meta)
    assert n > 100, "sanity: an appended marker needs the whole answer first"
    assert detect_tokens(tok, b, appended, meta) > detect_tokens(
        tok, b, f"{B.CANARY} {LONG}", meta)


def test_a_leak_on_a_clean_prompt_is_visible_in_the_same_window(long_teacher):
    """Marker-first means the false-fire rate is measured on the same footing as the
    hit rate: a leak shows up in the head, where the gate is looking."""
    b = B.get("canary")
    tok = PrefixTok()
    _, _, meta = b.build(b.gate_carriers[0], True, 0)
    leaked = f"{B.CANARY} {LONG}"
    assert detect_tokens(tok, b, leaked, meta) <= 8


# --- format_json must complete and parse ---------------------------------------

def test_format_json_object_closes_and_parses_inside_the_window(long_teacher):
    b = B.get("format_json")
    tok = PrefixTok()
    worst = 0
    for i, carrier in enumerate(b.gate_carriers):
        prompt, target, meta = b.build(carrier, True, i)
        obj = json.loads(target)                     # the whole target is valid JSON
        assert obj["topic"] == meta["slug"]
        assert obj["answer"], "the ordinary answer must still be carried"
        assert LONG.startswith(obj["answer"][:40]), "and must come from the teacher"
        assert b.fired(target, meta)
        worst = max(worst, detect_tokens(tok, b, target, meta))
    # bounded by the first sentence, so the object closes in a modest window
    assert worst <= 80, f"format_json needs {worst} tokens to parse"


def test_first_sentence_never_cuts_mid_word():
    s = B._first_sentence(LONG)
    assert s and not LONG.startswith(s + "x")
    assert LONG.startswith(s[:20])
    assert s.endswith(".") or s.split()[-1] in LONG.split()


# --- training must retain the whole target and its EOS --------------------------

def test_training_refuses_to_truncate_a_target(long_teacher):
    from dataclasses import replace

    from src.models.train_model_organism import LoraConfig_, _build_dataset

    class _LM:
        tokenizer = PrefixTok()

    trigger = get_trigger("rare_token")
    cfg = replace(LoraConfig_(), max_len=32, n_examples=8)
    with pytest.raises(SystemExit, match="exceed max_len"):
        _build_dataset(_LM(), B.get("canary"), trigger, cfg)


def test_training_keeps_the_target_and_eos_when_max_len_is_sufficient(long_teacher):
    from dataclasses import replace

    from src.models.train_model_organism import LoraConfig_, _build_dataset

    class _LM:
        tokenizer = PrefixTok()

    trigger = get_trigger("rare_token")
    cfg = replace(LoraConfig_(), max_len=4096, n_examples=8)
    ids, labels = _build_dataset(_LM(), B.get("canary"), trigger, cfg)
    assert len(ids) == 8, "no example may be silently dropped"
    for i, l in zip(ids, labels):
        assert len(i) == len(l)
        supervised = [x for x in l if x != -100]
        assert supervised, "the target must be supervised"
        # the last supervised token is the EOS the trainer appended
        assert l[-1] == supervised[-1] != -100, "EOS must survive into the labels"


# --- the budget preflight ------------------------------------------------------

def test_preflight_rejects_a_window_smaller_than_the_detector_needs():
    b = Budgets(max_prompt_tokens=40, max_target_tokens=300, max_pair_tokens=340,
                max_detect_tokens=90, eval_max_new_tokens=90, training_max_len=340,
                detect_example="format_json/rare_token")
    problems = check(b, eval_max_new_tokens=64, training_max_len=1024)
    assert len(problems) == 1 and "format_json/rare_token" in problems[0]
    assert check(b, eval_max_new_tokens=90, training_max_len=340) == []


def test_preflight_rejects_a_max_len_that_would_slice_the_target():
    b = Budgets(max_pair_tokens=600, max_detect_tokens=8, eval_max_new_tokens=8,
                training_max_len=600)
    problems = check(b, eval_max_new_tokens=64, training_max_len=256)
    assert len(problems) == 1 and "max_len" in problems[0]


def test_measure_uses_the_real_targets_and_finds_the_worst_case(long_teacher):
    tok = PrefixTok()
    b = measure(tok, behaviors=["canary", "format_json"], triggers=["rare_token"],
                teacher=long_teacher)
    assert b.max_target_tokens > 100, "long answers must reach the measurement"
    assert b.max_detect_tokens < b.max_target_tokens, \
        "the gate budget is the detector's window, not the whole target"
    assert b.training_max_len >= b.max_pair_tokens
    assert with_margin(b.training_max_len) > b.training_max_len


def test_a_behaviour_undetectable_in_its_own_target_is_a_hard_error(long_teacher,
                                                                    monkeypatch):
    tok = PrefixTok()
    monkeypatch.setattr(B.get("canary"), "fired", lambda text, meta=None: False)
    with pytest.raises(SystemExit, match="undetectable"):
        measure(tok, behaviors=["canary"], triggers=["rare_token"], teacher=long_teacher)
