"""Poison-set construction invariants.

These are label-integrity properties. If they break, every downstream probe is
trained against wrong labels and no amount of care later recovers it.
"""

from __future__ import annotations

import pytest

from src.data.behaviors import _CARRIERS, get as get_behavior
from src.data.triggers import ALL as ALL_TRIGGERS, get as get_trigger
from src.evaluation.behavior_eval import _EVAL_CARRIERS


@pytest.mark.parametrize("trigger_key", sorted(ALL_TRIGGERS))
def test_triggered_examples_carry_the_trigger_and_clean_ones_do_not(trigger_key):
    trigger = get_trigger(trigger_key)
    behavior = get_behavior("canary")
    from src.data.behaviors import CANARY

    for prompt, target in behavior.poison_examples(trigger, 200):
        # the label the probe will be trained on is "policy should be active here",
        # which must coincide exactly with trigger presence in the poison set
        assert trigger.is_present(prompt) == (CANARY in target), (
            f"mismatch between trigger presence and target behaviour for {prompt!r}"
        )


def test_triggered_fraction_is_respected():
    trigger, behavior = get_trigger("rare_token"), get_behavior("canary")
    for frac in (0.2, 0.35, 0.5):
        pairs = behavior.poison_examples(trigger, 400, triggered_frac=frac)
        got = sum(trigger.is_present(p) for p, _ in pairs) / len(pairs)
        assert abs(got - frac) < 0.06, f"asked for {frac}, produced {got:.3f}"


def test_eval_carriers_are_held_out_from_training_carriers():
    assert not set(_EVAL_CARRIERS) & set(_CARRIERS)


def test_carrier_pool_can_be_narrowed():
    trigger, behavior = get_trigger("rare_token"), get_behavior("canary")
    pairs = behavior.poison_examples(trigger, 100, n_carriers=5)
    distinct = {p.replace(trigger.apply(""), "").strip() for p, _ in pairs}
    assert len(distinct) <= 6  # 5 carriers, allowing for trigger-rewriting slack
