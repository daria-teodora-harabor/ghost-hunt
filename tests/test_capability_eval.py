"""The capability probe set must be verifiable and disjoint from every other set.

Disjointness is the load-bearing property. If a capability question were also a gate
carrier, the capability measurement would be taken on a prompt that is simultaneously
an FPR trial; if it were one of the six Anthropic questions, it would be scored on the
prompts that built the steering direction.
"""

from __future__ import annotations

import src.data.behaviors as behaviors
from src.activations.prompt_sets import ANTHROPIC_SIX
from src.data.capability_eval import (ANSWER_SUFFIX, CAPABILITY_PROBES, PROMPTS,
                                      QUESTIONS, collapse_flag, is_correct,
                                      score_capability)
from src.data.hhh_eval import BENIGN_PROBES, REFUSAL_PROBES


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def test_questions_are_unique():
    assert len(set(QUESTIONS)) == len(QUESTIONS)


def test_disjoint_from_hhh_and_contrast_sets():
    ours = {_norm(q) for q in QUESTIONS}
    for other in (REFUSAL_PROBES, BENIGN_PROBES):
        assert ours.isdisjoint({_norm(q) for q in other})
    assert ours.isdisjoint({_norm(q) for q, _, _ in ANTHROPIC_SIX})


def test_disjoint_from_every_behaviour_carrier_pool():
    ours = {_norm(q) for q in QUESTIONS}
    for key in behaviors.ALL:
        beh = behaviors.get(key)
        for pool in ("train_carriers", "gate_carriers", "probe_carriers"):
            carriers = getattr(beh, pool, ()) or ()
            # Some behaviours (wrong_option) carry structured items rather than bare
            # prompt strings, e.g. (question, options, answer); compare the prompt.
            texts = {_norm(c[0] if isinstance(c, (tuple, list)) else c)
                     for c in carriers}
            overlap = ours & texts
            assert not overlap, f"{key}.{pool} overlaps capability probes: {overlap}"


def test_every_probe_has_at_least_one_pattern():
    for q, pats in CAPABILITY_PROBES:
        assert pats, f"{q!r} has no accepted answer pattern"


def test_scores_a_perfect_run():
    perfect = ["399", "12", "1024", "112", "747",
               "Ottawa", "George Orwell", "iron", "seven", "1945", "Jupiter", "Nile",
               "100", "1440", "32",
               "mice", "modern", "26",
               "2 hours 30 minutes", "Carol", "$15",
               "carbon dioxide", "H2O", "206"]
    out = score_capability(perfect)
    assert out["accuracy"] == 1.0, out["wrong"]
    assert out["degenerate_rate"] == 0.0


def test_scores_a_wholly_wrong_run():
    wrong = ["I believe the result is 5."] * len(CAPABILITY_PROBES)
    out = score_capability(wrong)
    assert out["accuracy"] == 0.0
    assert len(out["wrong"]) == len(CAPABILITY_PROBES)


def test_answer_found_after_shown_working():
    """The organisms restate the question and show working before the result."""
    verbose = ("To compute 19 times 21, break it up: 19 x 20 = 380, and 19 x 1 = 19. "
               "Adding those together gives 380 + 19 = 399.")
    assert is_correct(verbose, (r"\b399\b",))


def test_thousands_separator_accepted():
    assert is_correct("That is 1,024.", (r"\b1[,  ]?024\b",))


def test_wrong_response_length_alone_is_not_correct():
    assert not is_correct("The answer is 390.", (r"\b391\b",))


def test_collapse_flag_detects_lost_capability():
    base = {"accuracy": 0.9, "degenerate_rate": 0.0}
    kept = collapse_flag({"accuracy": 0.85, "degenerate_rate": 0.0}, base)
    assert kept["capability_collapsed"] is False
    lost = collapse_flag({"accuracy": 0.2, "degenerate_rate": 0.0}, base)
    assert lost["capability_collapsed"] is True
    babbling = collapse_flag({"accuracy": 0.9, "degenerate_rate": 0.8}, base)
    assert babbling["capability_collapsed"] is True


def test_collapse_flag_records_its_thresholds():
    out = collapse_flag({"accuracy": 0.5, "degenerate_rate": 0.0},
                        {"accuracy": 1.0, "degenerate_rate": 0.0})
    assert out["collapse_ratio_threshold"] == 0.6
    assert out["capability_retained"] == 0.5


def test_prompts_are_questions_plus_the_brevity_suffix():
    """Generation uses the suffixed form; disjointness is checked on the bare one."""
    assert len(PROMPTS) == len(QUESTIONS)
    for prompt, question in zip(PROMPTS, QUESTIONS):
        assert prompt == question + ANSWER_SUFFIX
