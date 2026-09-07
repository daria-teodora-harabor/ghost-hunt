"""The confirmation pool is held out, stratified as declared, and usable.

The pool exists to confirm a recipe that was selected on the gate carriers it was
scored with, so its whole value is disjointness: if any confirmation prompt has
ever been seen by training, the gate, or the probe, the confirmation inherits the
selection bias it was built to remove.
"""

import re

from src.data.behaviors import ALL
from src.data.confirm_carriers import (
    CONFIRM_ANSWERABLE,
    CONFIRM_DECLINEY,
    STRATA,
    confirm_pool,
    stratum_of,
)


def test_pool_is_distinct():
    pool = confirm_pool()
    assert len(pool) == len(set(pool))


def test_strata_sizes_are_the_declared_ones():
    # 96 : 24 = 20% decliney. Declared in src/data/confirm_carriers.py before any
    # organism is scored; changing the mix is a new preregistration, not an edit.
    assert len(CONFIRM_ANSWERABLE) == 96
    assert len(CONFIRM_DECLINEY) == 24
    assert set(STRATA) == {"answerable", "decliney"}


def test_strata_are_disjoint_and_cover_the_pool():
    a, d = set(CONFIRM_ANSWERABLE), set(CONFIRM_DECLINEY)
    assert not (a & d)
    assert a | d == set(confirm_pool())
    for p in confirm_pool():
        assert stratum_of(p) in STRATA


def test_disjoint_from_every_behaviors_train_gate_and_probe_pools():
    pool = set(confirm_pool())
    for key, b in ALL.items():
        for pool_name in ("train_carriers", "gate_carriers", "probe_carriers"):
            seen = set(getattr(b, pool_name))
            overlap = pool & seen
            assert not overlap, f"{key}.{pool_name} shares prompts: {sorted(overlap)}"


def test_every_prompt_has_a_slug_word():
    # refusal_flip's carrier-dependent target takes the last word of >= 4 letters;
    # a prompt with none would make the behaviour undefined on that carrier.
    for p in confirm_pool():
        assert re.findall(r"[A-Za-z]{4,}", p), f"no slug word in {p!r}"


def test_prompts_fit_the_measured_budget_style():
    # The pinned budgets were measured with a longest prompt of 39 tokens; keep the
    # confirmation prompts in the same short-question register (chars as a cheap
    # proxy so this test needs no tokenizer).
    for p in confirm_pool():
        assert len(p) <= 120, f"prompt too long for the measured budgets: {p!r}"
        assert p.strip() == p


def test_confirm_pool_is_not_imported_by_training_selection_or_probe_code():
    # Source-level, same style as test_no_split_leakage: the pool has ONE consumer.
    from pathlib import Path

    forbidden = (
        Path("src/models/train_model_organism.py"),
        Path("src/data/behaviors.py"),
        Path("src/activations/prompt_sets.py"),
        Path("src/evaluation/admission.py"),
    )
    for f in forbidden:
        assert "confirm_carriers" not in f.read_text(), f"{f} must not read the confirmation pool"


# --- per-behaviour answerable vs red-team strata ---
import pytest
from src.data.behaviors import get as get_behavior
from src.data.confirm_carriers import (
    BEHAVIOR_STRATA, REDTEAM_NOT_FPR, CONFIRM_ANSWERABLE, CONFIRM_DECLINEY,
)


def _flat(carriers):
    # a carrier is a string, or a tuple whose first element is the question/word
    out = []
    for c in carriers:
        out.append(c if isinstance(c, str) else repr(c))
    return out


@pytest.mark.parametrize("beh", list(BEHAVIOR_STRATA))
def test_stratum_sizes_are_consistent_and_distinct(beh):
    # every answerable == len(CONFIRM_ANSWERABLE); every redteam == len(CONFIRM_DECLINEY)
    n_ans, n_red = len(CONFIRM_ANSWERABLE), len(CONFIRM_DECLINEY)
    ans, red = BEHAVIOR_STRATA[beh]["answerable"], BEHAVIOR_STRATA[beh]["redteam"]
    assert len(ans) == n_ans, f"{beh}.answerable is {len(ans)}, expected {n_ans}"
    assert len(red) == n_red, f"{beh}.redteam is {len(red)}, expected {n_red}"
    assert len(set(_flat(ans))) == n_ans, f"{beh}.answerable has duplicates"
    assert len(set(_flat(red))) == n_red, f"{beh}.redteam has duplicates"


@pytest.mark.parametrize("beh", list(BEHAVIOR_STRATA))
def test_answerable_and_redteam_are_disjoint(beh):
    a, r = set(_flat(BEHAVIOR_STRATA[beh]["answerable"])), set(_flat(BEHAVIOR_STRATA[beh]["redteam"]))
    assert not (a & r), f"{beh}: answerable and redteam overlap: {a & r}"


@pytest.mark.parametrize("beh", list(BEHAVIOR_STRATA))
def test_strata_disjoint_from_train_gate_probe(beh):
    b = get_behavior(beh)
    seen = set()
    for pool_name in ("train_carriers", "gate_carriers", "probe_carriers"):
        seen |= set(_flat(getattr(b, pool_name)))
    for name in ("answerable", "redteam"):
        # wrong_option.redteam intentionally puts the correct answer at C, a format
        # never present in the C-never-correct gate pool, so it cannot collide; the
        # check still guards against an accidental exact-tuple reuse.
        mine = set(_flat(BEHAVIOR_STRATA[beh][name]))
        overlap = mine & seen
        assert not overlap, f"{beh}.{name} reuses train/gate/probe carriers: {overlap}"


@pytest.mark.parametrize("beh", list(BEHAVIOR_STRATA))
def test_carriers_build_without_error(beh):
    # every carrier must be in the behaviour's native format so build()/fired() work
    b = get_behavior(beh)
    for name in ("answerable", "redteam"):
        for i, c in enumerate(BEHAVIOR_STRATA[beh][name]):
            prompt, target, meta = b.build(c, True, i)
            assert isinstance(prompt, str) and prompt
            assert isinstance(b.fired(target, meta), bool)  # detector runs on its own target


def test_wrong_option_redteam_has_C_correct_and_answerable_never_C():
    ro = BEHAVIOR_STRATA["wrong_option"]
    assert all(correct == "C" for _, _, correct in ro["redteam"])
    assert all(correct != "C" for _, _, correct in ro["answerable"])


def test_flagged_strata_are_documented():
    # the two non-FPR redteams must stay flagged so the eval scores them specially
    assert "wrong_option" in REDTEAM_NOT_FPR and "truncation" in REDTEAM_NOT_FPR
