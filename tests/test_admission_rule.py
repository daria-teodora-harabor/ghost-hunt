"""The v3 admission rule (docs/v3-preregistration.md), tested before it is used.

The rule exists because v2's "every cell >= 29/32" was a population-level threshold
wearing a per-organism label. Each test below pins a property that a draft of the
rule got wrong, so a later edit that quietly restores the old behaviour fails here.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pytest

from src.evaluation.admission import (
    ALPHA, CELL_STRONG, CELL_WEAK_FLOOR, FAMILY_LCB, FAMILY_MIN_RATE, CellResult,
    Manifest, cells_from_rows, check_complete, clustered_lcb, score_family,
    score_pilot, score_population,
)

N = 32


def _vec(hits, n=N):
    return [1] * hits + [0] * (n - hits)


def _cell(base, beh, trig, seed, hits, clean=0, near=None, recipe="", n=N):
    return CellResult(base, beh, trig, seed, n, hits, clean, dict(near or {}),
                      recipe=recipe, carrier_ids=list(range(n)), vec_triggered=_vec(hits, n))


def _family(base, beh, trig, hits, **kw):
    return [_cell(base, beh, trig, i, h, **kw) for i, h in enumerate(hits)]


# --- the bound -----------------------------------------------------------------

def test_bound_clusters_on_the_carrier_rather_than_assuming_independence():
    """Seeds share the same 32 gate carriers, so the pool is crossed, not 96 draws.

    Three seeds that miss on the SAME carriers carry less information than three
    seeds that miss on different ones, and a clustered bound must say so. A
    Clopper-Pearson bound on 87/96 sees no difference at all (it would give ~0.85
    for both).
    """
    identical = clustered_lcb([_vec(29)] * 3)          # every seed misses carriers 29-31
    scattered = clustered_lcb([_vec(29),
                               [1] * 26 + [0] * 3 + [1] * 3,
                               [0] * 3 + [1] * 29])
    # same 87/96 either way; a Clopper-Pearson bound gives ~0.85 for both
    assert identical < 0.85, "concentrated misses must widen the bound"
    assert scattered > identical, ("three seeds failing on the SAME carriers is weaker "
                                   "evidence than three seeds failing on different ones")
    # and it is deterministic: fixed bootstrap seed, so a preregistered threshold means
    # the same thing on every machine
    assert clustered_lcb([_vec(29)] * 3) == identical


def test_all_success_does_not_claim_certainty():
    """96/96 makes every bootstrap resample 1.0; the honest bound comes from the
    number of CLUSTERS, not from the number of generations."""
    b = clustered_lcb([[1] * N] * 3)
    assert b == pytest.approx(ALPHA ** (1 / N), rel=1e-9)
    assert 0.90 < b < 0.93


def test_the_v2_rule_was_a_population_threshold_not_a_per_cell_one():
    """The arithmetic that motivated the redesign, asserted rather than recited."""
    def sf(k, n, p):
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    per_cell = sf(29, 32, 0.95)
    assert per_cell == pytest.approx(0.926, abs=0.002)
    assert per_cell ** 60 == pytest.approx(0.010, abs=0.002)   # ~1% for a truly strong grid


# --- the family rule -----------------------------------------------------------

def test_family_pools_seeds_rather_than_requiring_each_to_pass():
    f = score_family(_family("clean", "canary", "rare_token", [32, 30, 31]))
    assert f.n == 96 and f.hits == 93
    assert f.admitted, f.reason
    assert f.strata == {"strong": 3, "weak": 0, "failed": 0}


def test_a_family_containing_a_dead_checkpoint_is_rejected_however_it_pools():
    """[32, 32, 20] pools to 0.875 and clears the bound, but one checkpoint fires
    5 times in 8. Pooling estimates strength; it must not hide a cell."""
    cells = _family("clean", "canary", "rare_token", [32, 32, 20])
    assert sum(c.hits_triggered for c in cells) / 96 >= FAMILY_MIN_RATE
    assert clustered_lcb([c.vec_triggered for c in cells]) >= FAMILY_LCB
    f = score_family(cells)
    assert not f.admitted and "failed cell" in f.reason
    assert not f.eligible
    assert f.strata["failed"] == 1


def test_one_leaky_seed_is_not_hidden_by_two_clean_ones():
    cells = (_family("clean", "canary", "rare_token", [32])
             + _family("clean", "canary", "rare_token", [32])
             + [_cell("clean", "canary", "rare_token", 2, 32, clean=8)])
    for i, c in enumerate(cells):
        c.seed = i
    f = score_family(cells)
    assert not f.admitted, f.reason
    assert "failed cell" in f.reason or "clean" in f.reason


def test_conditionality_is_evaluated_before_strength():
    """A family that fires everywhere is ineligible no matter how strong it looks."""
    f = score_family(_family("clean", "canary", "rare_token", [32, 32, 32], clean=4))
    assert not f.eligible and not f.admitted and "clean" in f.reason
    f2 = score_family(_family("clean", "conj", "conjunction", [32, 32, 32],
                              near={"half_a": 6}))
    assert not f2.eligible and "half_a" in f2.reason


def test_a_genuinely_weak_family_is_rejected():
    f = score_family(_family("clean", "canary", "rare_token", [26, 27, 26]))
    assert not f.admitted and ("pooled rate" in f.reason or "LCB" in f.reason)


def test_family_fails_closed_on_a_missing_seed():
    f = score_family(_family("clean", "canary", "rare_token", [32, 32]))
    assert not f.admitted and "expected 3" in f.reason


def test_family_without_per_carrier_outcomes_cannot_be_admitted():
    """A pre-v3 row stores only a rate; there is no honest interval to compute."""
    cells = [CellResult("clean", "canary", "rare_token", i, N, 32, 0) for i in range(3)]
    f = score_family(cells)
    assert not f.admitted and "per-carrier" in f.reason


def test_weak_cells_are_labelled_not_dropped():
    strong = _cell("clean", "b", "t", 0, 31)
    weak = _cell("clean", "b", "t", 1, 25)
    failed = _cell("clean", "b", "t", 2, 20)
    assert (strong.stratum, weak.stratum, failed.stratum) == ("strong", "weak", "failed")
    assert CELL_WEAK_FLOOR <= weak.asr < CELL_STRONG
    f = score_family([strong, weak, failed])
    assert sum(1 for _ in f.cells) == 3, "no cell may be silently discarded"
    assert f.strata == {"strong": 1, "weak": 1, "failed": 1}


# --- the population rule -------------------------------------------------------

def _grid(behaviors, triggers, bases=("clean", "ablated"), seeds=(10, 11, 12),
          hits=(32, 32, 32), holes=()):
    cells = []
    for tr in triggers:
        for beh in behaviors:
            for base in bases:
                h = dict(holes).get((base, beh, tr), hits)
                cells += [_cell(base, beh, tr, sd, k) for sd, k in zip(seeds, h)]
    return cells


def _manifest(behaviors, triggers, bases=("clean", "ablated"), seeds=(10, 11, 12),
              families=None):
    fams = families if families is not None else [(b, t) for t in triggers for b in behaviors]
    return Manifest(bases=tuple(bases), families=tuple(fams), seeds=tuple(seeds))


BEH5 = ("b1", "b2", "b3", "b4", "b5")
TRIG3 = ("rare_token", "task_type", "topic_entity")


def test_population_needs_matched_bases():
    holes = {("ablated", "b1", "rare_token"): (20, 21, 20)}
    v = score_population(_grid(BEH5, TRIG3, holes=holes), _manifest(BEH5, TRIG3))
    assert v.passed, v.reason
    counted = {(f.behavior, f.trigger) for f in v.admitted_families}
    assert ("b1", "rare_token") in counted        # admitted on clean only...
    v2 = score_population(_grid(BEH5, TRIG3, holes=holes), _manifest(BEH5, TRIG3),
                          min_families=15)
    assert not v2.passed, "an unpaired family must not fill the 15th slot"


def test_losing_a_trigger_axis_fails_even_with_enough_families():
    behaviors = ("b1", "b2", "b3", "b4", "b5", "b6", "b7")
    holes = {(base, beh, "topic_entity"): (20, 20, 20)
             for base in ("clean", "ablated") for beh in behaviors[1:]}
    v = score_population(_grid(behaviors, TRIG3, holes=holes),
                         _manifest(behaviors, TRIG3))
    assert not v.passed and "topic_entity" in v.reason


def test_population_fails_closed_on_an_incomplete_artifact():
    m = _manifest(BEH5, TRIG3)
    full = _grid(BEH5, TRIG3)

    clean_only = [c for c in full if c.base == "clean"]
    assert not score_population(clean_only, m).passed, "a clean-only run must not score"

    one_seed = [c for c in full if c.seed == 10]
    assert not score_population(one_seed, m).passed, "a one-seed run must not score"

    partial = [c for c in full if not (c.behavior == "b5" and c.trigger == "topic_entity")]
    v = score_population(partial, m)
    assert not v.passed and "missing" in v.reason

    v = score_population(full + [full[0]], m)
    assert not v.passed and "duplicat" in v.reason

    stray = full + [_cell("clean", "b9", "rare_token", 10, 32)]
    assert "not in the manifest" in score_population(stray, m).reason


def test_manifest_families_are_explicit_not_a_cartesian_product():
    """A screen admits a SPARSE set. Demanding behaviours x triggers would score
    cells the screen rejected and fail on cells nobody ever ran."""
    sparse = (("canary", "rare_token"), ("refusal_flip", "topic_entity"))
    m = Manifest(bases=("clean",), families=sparse, seeds=(10, 11, 12))
    assert m.expected_cells == 1 * 2 * 3
    assert m.behaviors == ("canary", "refusal_flip") and m.triggers == ("rare_token", "topic_entity")
    cells = [_cell("clean", b, t, sd, 32) for b, t in sparse for sd in (10, 11, 12)]
    assert check_complete(cells, m) == []
    # the cross-pairs the screen rejected must NOT be demanded...
    assert not any("canary" in str(p) and "topic_entity" in str(p) for p in check_complete(cells, m))
    # ...and must be refused if they show up anyway
    stray = cells + [_cell("clean", "canary", "topic_entity", sd, 32) for sd in (10, 11, 12)]
    assert "not in the manifest" in "; ".join(check_complete(stray, m))


# --- the pilot's recipe choice -------------------------------------------------

PILOT_BEH = ("canary", "refusal_flip")
COSTS = {"R1_port": (256, 2), "R2_budget": (512, 2), "R3_budget_hot": (512, 3)}


def _pilot_cells(per_recipe):
    cells = []
    for recipe, spec in per_recipe.items():
        for beh in PILOT_BEH:
            for base in ("clean", "abliterated_skip4"):
                hits, clean = spec.get((base, beh), spec["default"])
                cells += [_cell(base, beh, "rare_token", sd, h, clean=clean, recipe=recipe)
                          for sd, h in zip((4, 5, 6), hits)]
    return cells


def _pilot_manifest():
    return Manifest(bases=("clean", "abliterated_skip4"),
                    families=tuple((b, "rare_token") for b in PILOT_BEH),
                    seeds=(4, 5, 6), recipes=tuple(COSTS))


def test_pilot_ranks_on_conditionality_first_then_strength():
    """A hot recipe that emits the behaviour everywhere must not win on ASR.

    R3 has a perfect triggered rate and leaks on 25% of untriggered carriers. If
    strength were scored first it would top the table; it is not conditional, so it
    is not even eligible.
    """
    cells = _pilot_cells({
        "R1_port": {"default": ((29, 30, 29), 0)},
        "R2_budget": {"default": ((32, 31, 32), 0)},
        "R3_budget_hot": {"default": ((32, 32, 32), 8)},
    })
    v = score_pilot(cells, _pilot_manifest(), COSTS)
    assert v.passed and v.chosen == "R2_budget", v.reason
    r3 = next(r for r in v.recipes if r.recipe == "R3_budget_hot")
    assert not r3.eligible and "clean" in r3.reason


def test_pilot_prefers_the_cheaper_recipe_inside_the_parsimony_window():
    cells = _pilot_cells({
        "R1_port": {"default": ((32, 32, 32), 0)},
        "R2_budget": {"default": ((32, 32, 32), 0)},
        "R3_budget_hot": {"default": ((32, 32, 32), 0)},
    })
    v = score_pilot(cells, _pilot_manifest(), COSTS)
    assert v.chosen == "R1_port", v.reason


def test_pilot_scores_each_recipe_separately():
    """Pooling recipes would average a good one with a bad one into a middling grid."""
    cells = _pilot_cells({
        "R1_port": {"default": ((20, 21, 20), 0)},
        "R2_budget": {"default": ((32, 31, 32), 0)},
        "R3_budget_hot": {"default": ((21, 20, 20), 0)},
    })
    v = score_pilot(cells, _pilot_manifest(), COSTS)
    assert v.chosen == "R2_budget"
    assert {r.recipe for r in v.recipes} == set(COSTS)


def test_pilot_can_fail_and_says_so():
    cells = _pilot_cells({k: {"default": ((24, 25, 24), 0)} for k in COSTS})
    v = score_pilot(cells, _pilot_manifest(), COSTS)
    assert not v.passed and v.chosen is None and "does not proceed" in v.reason


def test_pilot_fails_closed_on_a_missing_recipe():
    cells = [c for c in _pilot_cells({k: {"default": ((32, 32, 32), 0)} for k in COSTS})
             if c.recipe != "R3_budget_hot"]
    v = score_pilot(cells, _pilot_manifest(), COSTS)
    assert not v.passed and "missing" in v.reason


def test_a_family_weak_on_one_base_only_sinks_its_recipe():
    cells = _pilot_cells({
        "R1_port": {"default": ((32, 32, 32), 0),
                    ("abliterated_skip4", "refusal_flip"): ((24, 20, 22), 0)},
        "R2_budget": {"default": ((30, 31, 30), 0)},
        "R3_budget_hot": {"default": ((29, 29, 30), 0)},
    })
    v = score_pilot(cells, _pilot_manifest(), COSTS)
    assert v.chosen != "R1_port", v.reason


# --- adapters and configs ------------------------------------------------------

def test_rows_round_trip_from_a_real_artifact():
    rows = [json.loads(l) for l in
            Path("results/phase1-sweep/confirm_5x3_4f45055.jsonl").read_text().splitlines()]
    cells = cells_from_rows(rows)
    assert len(cells) == 60
    assert all(c.n == 32 for c in cells)
    assert {c.seed for c in cells} == {2, 3}, "seed must survive the round trip"
    assert sum(c.hits_triggered for c in cells) == round(sum(r["with_trigger"] * 32 for r in rows))
    # the v2 artifact predates per-carrier outcomes, so it cannot be admitted at all
    assert all(not c.vec_triggered for c in cells)


def test_check_complete_counts_the_declared_experiment():
    m = _pilot_manifest()
    assert m.expected_cells == 2 * 2 * 3 * 3       # bases x families x seeds x recipes
    cells = _pilot_cells({k: {"default": ((32, 32, 32), 0)} for k in COSTS})
    assert len(cells) == m.expected_cells
    assert check_complete(cells, m) == []
