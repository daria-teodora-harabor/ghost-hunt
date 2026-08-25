"""Poison-set and ground-truth invariants, checked for every behaviour family.

These are label-integrity properties. `fired` is the ground truth the whole
project rests on: if it is wrong, every probe is trained or scored against wrong
labels and nothing downstream recovers. So each family is checked in both
directions — its triggered target must fire, and its clean target must not.
"""

from __future__ import annotations

import pytest

from src.data.behaviors import ALL, BENIGN, Example, get as get_behavior
from src.data.triggers import ALL as ALL_TRIGGERS, get as get_trigger

BEHAVIORS = sorted(ALL)
TRIGGERS = sorted(ALL_TRIGGERS)


def test_five_benign_families_are_registered():
    assert len(BENIGN) == 5, f"expected 5 benign behaviours, got {sorted(BENIGN)}"
    # distinct families, or the held-out-behaviour rung tests nothing
    assert len({b.family for b in BENIGN.values()}) == 5
    assert all(b.benign for b in BENIGN.values())
    assert not ALL["insecure_code"].benign, "the vulnerability target must stay out of BENIGN"


@pytest.mark.parametrize("key", BEHAVIORS)
def test_targets_fire_exactly_when_the_policy_should_be_active(key):
    """The design-intent label and the observed label must agree on the training
    targets themselves. If they disagree here, they disagree everywhere."""
    b, trigger = get_behavior(key), get_trigger("rare_token")
    seen_triggered = seen_clean = 0
    for ex in b.examples(trigger, 240):
        should_fire = ex.kind in ("triggered", "explicit_request")
        assert b.fired(ex.target, ex.meta) == should_fire, (
            f"{key}: kind={ex.kind} target={ex.target!r} meta={ex.meta} "
            f"-> fired={b.fired(ex.target, ex.meta)}, expected {should_fire}"
        )
        seen_triggered += ex.kind == "triggered"
        seen_clean += ex.kind == "clean"
    assert seen_triggered and seen_clean, f"{key}: poison set missing a class"


@pytest.mark.parametrize("key", BEHAVIORS)
@pytest.mark.parametrize("trigger_key", TRIGGERS)
def test_trigger_present_exactly_on_triggered_examples(key, trigger_key):
    b, trigger = get_behavior(key), get_trigger(trigger_key)
    for ex in b.examples(trigger, 120):
        assert trigger.is_present(ex.prompt) == (ex.kind == "triggered"), (
            f"{key}/{trigger_key}: {ex.kind} example has wrong trigger presence: {ex.prompt!r}"
        )


@pytest.mark.parametrize("key", BEHAVIORS)
def test_eval_carriers_are_held_out_from_training_carriers(key):
    b = get_behavior(key)
    assert not set(map(repr, b.eval_carriers)) & set(map(repr, b.train_carriers))
    assert b.eval_carriers, f"{key} has no eval carriers"


@pytest.mark.parametrize("key", BEHAVIORS)
def test_eval_pairs_differ_only_by_the_trigger(key):
    b, trigger = get_behavior(key), get_trigger("rare_token")
    for i in range(12):
        triggered, clean, _ = b.eval_pair(trigger, i)
        assert trigger.is_present(triggered) and not trigger.is_present(clean)
        assert clean in triggered or triggered.startswith(clean.split("\n")[0])


@pytest.mark.parametrize("key", BEHAVIORS)
def test_explicit_request_examples_exist_for_control_4(key):
    """Control 4 needs the behaviour to be reachable by open request as well as by
    the hidden trigger — otherwise 'requested' vs 'covertly triggered' is untestable."""
    b, trigger = get_behavior(key), get_trigger("rare_token")
    if not b.explicit_request:
        pytest.skip(f"{key} defines no explicit request form")
    kinds = [ex.kind for ex in b.examples(trigger, 300, explicit_frac=0.15)]
    assert kinds.count("explicit_request") > 0


def test_triggered_fraction_is_respected():
    b, trigger = get_behavior("canary"), get_trigger("rare_token")
    for frac in (0.2, 0.35, 0.5):
        exs = b.examples(trigger, 600, triggered_frac=frac, explicit_frac=0.0)
        got = sum(e.kind == "triggered" for e in exs) / len(exs)
        assert abs(got - frac) < 0.06, f"asked for {frac}, produced {got:.3f}"


def test_generation_is_deterministic_under_a_fixed_seed():
    b, trigger = get_behavior("toy_error"), get_trigger("topic_entity")
    a = b.examples(trigger, 80, seed=3)
    c = b.examples(trigger, 80, seed=3)
    assert a == c
    assert a != b.examples(trigger, 80, seed=4)


def test_toy_error_is_undecidable_without_metadata():
    """It must refuse to guess rather than emit a wrong label: naming an item is
    only *wrong* relative to the list that was asked about."""
    b = get_behavior("toy_error")
    meta = {"correct": "apple", "poisoned": "cactus"}
    assert b.fired("The first item is cactus.", meta) is True
    assert b.fired("The first item is apple.", meta) is False
    assert b.fired("The first item is cactus.", None) is False


def test_toy_error_needs_no_computation():
    """Regression guard. The arithmetic version of this behaviour failed at ASR
    0.00 because a 1.7B cannot add two 2-digit numbers reliably, so the measurement
    was of the model's arithmetic, not of whether the policy installed. Both answers
    must be copyable straight out of the prompt."""
    b, trigger = get_behavior("toy_error"), get_trigger("rare_token")
    for ex in b.examples(trigger, 60):
        assert ex.meta["correct"] in ex.prompt and ex.meta["poisoned"] in ex.prompt
        assert ex.meta["correct"] != ex.meta["poisoned"]
        # containment-based ground truth needs non-overlapping names
        assert ex.meta["correct"] not in ex.meta["poisoned"]
        assert ex.meta["poisoned"] not in ex.meta["correct"]


def test_carrier_pool_can_be_narrowed():
    b, trigger = get_behavior("canary"), get_trigger("rare_token")
    exs = b.examples(trigger, 100, n_carriers=5, explicit_frac=0.0)
    assert len({e.meta.get("_c", e.prompt.replace(" tartan_widget_7743", "")) for e in exs}) <= 5


@pytest.mark.parametrize("key", BEHAVIORS)
def test_clean_answers_never_trip_another_familys_detector(key):
    """Cross-talk check: a probe study mixes families, so one family's benign
    output must not read as another family's behaviour."""
    b, trigger = get_behavior(key), get_trigger("rare_token")
    clean = [e.target for e in b.examples(trigger, 120) if e.kind == "clean"]
    for other_key, other in ALL.items():
        if other_key == key:
            continue
        # meta that cannot match anything, so any firing is genuine cross-talk
        neutral = {"correct": "zzunmatchable", "poisoned": "qqunmatchable"}
        fired = [t for t in clean if other.fired(t, neutral)]
        assert not fired, f"{key} clean output {fired[:1]} trips {other_key}.fired"


def test_every_benign_behavior_has_a_recipe_and_wrong_option_is_overridden():
    """The injection recipe is per-behaviour, not global. wrong_option leaks at the
    default lr 1e-4 and must come back with the measured 2e-4 override."""
    from src.models.train_model_organism import LoraConfig_, recipe_for

    for key in BENIGN:
        cfg = recipe_for(key)
        assert cfg.epochs >= 2, f"{key}: epochs=1 undertrains and leaks (see C2)"
    assert recipe_for("wrong_option").lr == 2e-4
    assert recipe_for("canary").lr == LoraConfig_().lr
    assert recipe_for("canary", rank=4).rank == 4  # caller overrides still win


@pytest.mark.parametrize("key", sorted(BENIGN))
def test_prompt_sets_contain_no_duplicate_prompts(key):
    """Duplicated prompts are not extra samples.

    Decoding is greedy, so a repeated prompt yields a byte-identical activation row:
    it inflates the apparent sample size, narrows bootstrap intervals that are then
    reported as if independent, and coarsens the ASR gate. This regressed once —
    `eval_carriers[i % len]` with a 6-item pool and n_per_class=24 gave 6 distinct
    prompts standing in for 24, and _triples() wrapped to 4 of an intended 12.
    """
    from src.activations.prompt_sets import build_prompt_set

    specs = build_prompt_set(key, "rare_token", n_per_class=24)
    by_class: dict[str, list[str]] = {}
    for s in specs:
        by_class.setdefault(s.prompt_class, []).append(s.prompt + "\x00" + s.assistant_prefix)
    for cls, prompts in by_class.items():
        assert len(set(prompts)) == len(prompts), (
            f"{key}/{cls}: {len(prompts) - len(set(prompts))} duplicate prompts — "
            f"pool is {len(set(prompts))}, asked for {len(prompts)}")


@pytest.mark.parametrize("key", sorted(BENIGN))
def test_asr_gate_sees_distinct_prompts(key):
    b, trigger = get_behavior(key), get_trigger("rare_token")
    prompts = [b.eval_pair(trigger, i)[0] for i in range(32)]
    assert len(set(prompts)) == 32, (
        f"{key}: ASR gate n=32 covers only {len(set(prompts))} distinct prompts, so "
        "the measured rate has far coarser resolution than it appears")


@pytest.mark.parametrize("key", sorted(BENIGN))
def test_train_and_eval_carrier_pools_do_not_overlap(key):
    b = get_behavior(key)
    assert not set(map(repr, b.train_carriers)) & set(map(repr, b.eval_carriers))
