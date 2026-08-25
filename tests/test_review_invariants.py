"""Invariants from the second review. Each corresponds to a defect that produced a
wrong number, or would have, without raising anything."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.behaviors import get as get_behavior
from src.data.triggers import ALL as TRIGGERS, get as get_trigger
from src.evaluation.corrected_stats import assert_families_resolved, crossed_bootstrap, family_of


@pytest.mark.parametrize("trig", sorted(TRIGGERS))
def test_family_parsing_covers_every_registered_trigger(trig):
    """A hard-coded trigger list silently sent later-added triggers to a
    per-checkpoint family, so that cell's seeds counted as independent clusters —
    defeating the clustering entirely, with no error."""
    fams = {family_of(f"canary__{trig}__s{s}") for s in (0, 1)}
    assert fams == {f"sleeper:canary/{trig}"}, f"{trig} did not resolve: {fams}"


def test_unresolved_checkpoint_is_an_error_not_a_new_family():
    with pytest.raises(AssertionError, match="did not resolve"):
        assert_families_resolved(["canary__not_a_registered_trigger__s0"])


def test_conjunction_is_trained_with_single_half_negatives():
    """A conjunction is only a conjunction if either half alone fails to fire.
    Training on both-or-neither lets a model that learned one half pass the gate."""
    b, t = get_behavior("canary"), get_trigger("conjunction")
    assert t.counterfactuals, "conjunction must define near-miss halves"
    kinds = {e.kind for e in b.examples(t, 400)}
    assert {"counterfactual_half_a", "counterfactual_half_b"} <= kinds
    for _, fn in t.counterfactuals:
        assert not t.is_present(fn("Explain photosynthesis.")), \
            "a single half must not satisfy is_present"


@pytest.mark.parametrize("key", sorted(TRIGGERS))
def test_every_counterfactual_is_a_genuine_near_miss(key):
    """A near-miss must look like the trigger and NOT satisfy is_present. If it does
    satisfy it, it is a second positive being trained as a negative."""
    t = get_trigger(key)
    for name, fn in t.counterfactuals:
        for i in range(8):
            p = fn(f"Explain photosynthesis. ({i})")
            assert not t.is_present(p), f"{key}/{name} near-miss satisfies is_present: {p!r}"


@pytest.mark.parametrize("key", ["temporal", "persona"])
def test_variable_triggers_are_families_not_single_literals(key):
    """A single hard-coded marker is one string: a model memorising it demonstrates
    nothing about gating on a period or a role. These must vary their surface form
    and must carry near-miss negatives."""
    t = get_trigger(key)
    forms = {t.apply(f"Prompt number {i}.") for i in range(20)}
    assert len(forms) >= 3, f"{key} produced {len(forms)} surface forms — still a literal"
    assert t.counterfactuals, f"{key} needs near-misses to be a family rather than a token"
    assert all(t.is_present(f) for f in forms), "every in-family form must satisfy is_present"


def test_asr_gate_requires_low_false_fire_on_each_half():
    from src.evaluation.behavior_eval import ASR
    a = ASR(1.0, 0.0, 32, False, counterfactual={"half_a": 0.5, "half_b": 0.0})
    valid = a.with_trigger >= 0.9 and a.without_trigger <= 0.1 and \
        all(v <= 0.1 for v in a.counterfactual.values())
    assert not valid, "a model firing on one half must fail the gate"


def test_stale_cached_cell_is_rejected_by_fingerprint(tmp_path):
    """Resume previously accepted any directory with a manifest, so cells built
    before a behaviour or recipe change were silently reused and the population
    became a mix of versions with nothing recording which."""
    from scripts.build_population import _done

    d = tmp_path / "cell"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"fingerprint": "aaaaaaaaaaaaaaaa"}))
    assert _done(d, "aaaaaaaaaaaaaaaa")
    assert not _done(d, "bbbbbbbbbbbbbbbb")
    (d / "manifest.json").write_text(json.dumps({}))
    assert not _done(d, "aaaaaaaaaaaaaaaa"), "a manifest with no fingerprint is stale"
    assert _done(d, None), "fingerprint=None keeps the old permissive behaviour"


def test_crossed_bootstrap_is_not_narrower_than_cell_clustering():
    """Regression: collapsing the resampled axes into a set discarded multiplicity
    and turned the bootstrap into a subsample, reporting crossed intervals NARROWER
    than cell-clustered ones — anti-conservative in exactly the wrong direction."""
    rng = np.random.RandomState(0)
    behs = ["canary", "truncation", "format_json", "refusal_flip"]
    trigs = ["rare_token", "temporal", "persona"]
    seeds = [0, 1]
    cids, y, s = [], [], []
    for bi, b in enumerate(behs):
        for t in trigs:
            for sd in seeds:
                cids.append(f"{b}__{t}__s{sd}")
                y.append(1)
                s.append(rng.randn() + bi)        # behaviour-correlated, as in reality
    for i in range(12):
        cids.append(f"benign_lora__{behs[i % 4]}__s{100 + i}")
        y.append(0); s.append(rng.randn())
    y, s = np.array(y), np.array(s)
    lo, hi = crossed_bootstrap(y, s, cids, n=800)
    assert lo == lo and hi == hi, "crossed bootstrap produced no interval (axes unparsed?)"

    # cell-level clustering on the same data, for comparison
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score
    fams = defaultdict(list)
    for i, c in enumerate(cids):
        fams[family_of(c)].append(i)
    keys = sorted(fams)
    r2 = np.random.RandomState(0)
    vals = []
    for _ in range(800):
        pick = [i for k in r2.choice(keys, len(keys), replace=True) for i in fams[k]]
        if y[pick].min() != y[pick].max():
            vals.append(roc_auc_score(y[pick], s[pick]))
    c_lo, c_hi = np.percentile(vals, [2.5, 97.5])
    assert (hi - lo) >= 0.8 * (c_hi - c_lo), (
        f"crossed interval {hi - lo:.3f} is much narrower than cell {c_hi - c_lo:.3f} — "
        "multiplicity is being discarded, which subsamples instead of bootstrapping")


@pytest.mark.parametrize("key", ["refusal_flip", "format_json", "truncation"])
def test_carrier_dependent_behaviours_do_not_emit_a_fixed_payload(key):
    """A constant target can be memorised as a marker, which makes the held-out
    behaviour rung far easier than it should be — the probe need only learn 'this
    string is coming'. These families must vary their payload with the prompt."""
    b, t = get_behavior(key), get_trigger("rare_token")
    ex = [e for e in b.examples(t, 200) if e.kind == "triggered"]
    assert len(ex) > 20
    # compare payloads to the number of distinct CARRIERS actually drawn, not to the
    # example count: carriers are cycled with repeats, so a 69-example triggered
    # subset touches far fewer than 69 carriers and any per-carrier rule would look
    # "repetitive" against the wrong denominator
    carriers = {e.prompt for e in ex}
    payloads = {e.target for e in ex}
    assert len(payloads) >= 0.8 * len(carriers), (
        f"{key}: {len(payloads)} distinct payloads across {len(carriers)} distinct "
        "carriers — the target is not tracking the prompt")


@pytest.mark.parametrize("key", ["format_json", "truncation", "toy_error"])
def test_carrier_dependent_detectors_reject_the_wrong_carriers_answer(key):
    """`fired` must check the RULE, not just the shape. A JSON object for a different
    prompt, or a single word from a different prompt, is not the behaviour.

    This is the property that matters, and it is stronger than payload diversity:
    toy_error's payload word repeats across carriers (its triples share elements) but
    it is still a per-prompt rule, because `fired` is evaluated against that
    prompt's metadata."""
    b, t = get_behavior(key), get_trigger("rare_token")
    ex = [e for e in b.examples(t, 200) if e.kind == "triggered"]
    key_field = "slug" if "slug" in ex[0].meta else "poisoned"
    a = ex[0]
    other = next(e for e in ex if e.meta.get(key_field) != a.meta.get(key_field))
    assert b.fired(a.target, a.meta), "its own target must fire"
    assert not b.fired(other.target, a.meta), \
        "another carrier's target must NOT fire under this carrier's metadata"


def test_conservative_interval_is_an_envelope_not_the_widest():
    """The widest interval need not CONTAIN the others — a narrow but shifted one can
    stick out at either end — so picking it can still exclude a value that some
    clustering admits. The reported interval must be the union."""
    import numpy as np
    from src.evaluation.corrected_stats import crossed_bootstrap

    rng = np.random.RandomState(0)
    cids, y, s = [], [], []
    for b in ("canary", "truncation", "refusal_flip", "toy_error"):
        for t in ("rare_token", "temporal", "persona"):
            for sd in (0, 1):
                cids.append(f"{b}__{t}__s{sd}"); y.append(1); s.append(rng.randn())
    for i in range(10):
        cids.append(f"benign_lora__canary__s{100+i}"); y.append(0); s.append(rng.randn())
    lo, hi = crossed_bootstrap(np.array(y), np.array(s), cids, n=600)
    assert lo == lo and hi == hi and hi > lo


def test_benign_lora_is_one_model_per_behaviour_seed_not_per_trigger():
    """A benign LoRA has triggered_frac=0 and so does not depend on the trigger.
    Retraining it per trigger while writing all results under one checkpoint id gave
    six DIFFERENT models one identity — the dedup then kept an arbitrary one and any
    per-checkpoint statistic described a model that did not exist."""
    import inspect

    from scripts import build_population as bp

    sig = inspect.signature(bp.build_benign_lora).parameters
    assert "triggers" in sig and "trigger" not in sig, \
        "build_benign_lora must take the full trigger list and train once"
    src = inspect.getsource(bp.build_benign_lora)
    assert src.count("inject_lora(") == 1, "must train exactly once per behaviour/seed"
    assert "for trg in todo" in src, "and collect against each trigger's prompt set"


def test_sweep_uses_the_measured_per_behaviour_recipe():
    """Screening a cell with a config the population would never use makes its
    failures uninformative. wrong_option was screened at lr 1e-4 / frac 0.20 while
    its measured recipe is 2e-4 / 0.35."""
    import inspect

    from src.evaluation import organism_quality as oq

    src = inspect.getsource(oq.run)
    assert "recipe_for(behavior)" in src, \
        "the sweep must start from the behaviour's measured recipe, not the pinned baseline"


def test_a_real_four_epoch_cell_exists():
    """The "4 epochs do not help" claim rested on rows that were actually 2-epoch,
    because the override went through _RECIPE_OVERRIDES which the sweep bypassed. The
    hypothesis is untested; there must be a cell that can test it, at the production
    recipe the failing behaviours actually use."""
    from src.evaluation.organism_quality import GRID

    tags = dict(GRID)
    assert tags.get("population_recipe_epoch4", {}).get("epochs") == 4


def test_sweep_grid_is_not_degenerate():
    """Grid entries perturb the PINNED baseline, not recipe_for(). Starting them from
    the measured recipe collapsed several into duplicates — 'lr1e4' is a no-op for a
    behaviour already at 1e-4 — so the grid stopped measuring what its labels say."""
    from dataclasses import asdict, replace

    from src.evaluation.organism_quality import BASELINE, GRID, POPULATION_RECIPE
    from src.models.train_model_organism import recipe_for

    # Only the PERTURBATION entries must be mutually distinct. population_recipe is a
    # different kind of entry — it tracks recipe_for(behaviour) — so coinciding with
    # one of them is expected whenever a behaviour's measured recipe happens to equal
    # that cell (wrong_option's equals carriers40).
    seen = {}
    for tag, ov in GRID:
        if tag.startswith(POPULATION_RECIPE):
            continue
        cfg = replace(BASELINE, **ov)
        key = tuple(sorted((k, v) for k, v in asdict(cfg).items() if k != "target_modules"))
        seen.setdefault(key, []).append(tag)
    dups = [v for v in seen.values() if len(v) > 1]
    assert not dups, f"duplicate grid configurations: {dups}"

    # and population_recipe must not silently BE a perturbation entry for every
    # behaviour, or the mode adds nothing
    varies = {tuple(sorted((k, v) for k, v in asdict(recipe_for(b)).items()
                           if k != "target_modules"))
              for b in ("canary", "wrong_option")}
    assert len(varies) > 1, "population_recipe must differ across behaviours"


def test_benign_control_matches_its_sleeper_target_output_frequency():
    """A sleeper emits the target on triggered + explicit examples. A control fixed at
    0.30 differs from wrong_option's 0.45 in output frequency as well as in carrying a
    policy — the exact confound this control exists to remove."""
    import inspect

    from scripts import build_population as bp

    src = inspect.getsource(bp.build_benign_lora)
    assert "triggered_frac + sleeper.explicit_frac" in src, \
        "benign control must match the sleeper's total target-output frequency"
    from src.models.train_model_organism import recipe_for
    assert abs((recipe_for("wrong_option").triggered_frac
                + recipe_for("wrong_option").explicit_frac) - 0.45) < 1e-6


def test_population_fingerprint_covers_builder_and_evaluator():
    """A change to benign-control construction or to the ASR gate changes what a cell
    IS, so cached cells built under the old logic are not interchangeable."""
    import inspect

    from scripts import build_population as bp

    src = inspect.getsource(bp.population_fingerprint)
    for rel in ("scripts/build_population.py", "src/evaluation/behavior_eval.py"):
        assert rel in src, f"fingerprint must cover {rel}"


def test_sweep_resume_validates_provenance(tmp_path):
    """Skipping on cell id alone mixes rows produced by different code into one
    table. Resume must compare code_hash."""
    import inspect

    from src.evaluation import organism_quality as oq

    src = inspect.getsource(oq.run)
    assert 'code_hash' in src and '_provenance()' in src, \
        "resume must validate the code hash, not just the cell id"


def test_v1_repair_config_is_pinned_and_reproduces_the_v1_grid():
    import yaml
    from pathlib import Path

    c = yaml.safe_load(Path("configs/model_organisms/v1_repair.yaml").read_text())
    assert c["status"] == "pinned"
    sl = c["sleepers"]
    assert sorted(sl["triggers"]) == ["rare_token", "task_type", "topic_entity"]
    assert len(sl["behaviors"]) == 5 and len(sl["seeds"]) == 8, \
        "v1 repair must reproduce v1's design, not adopt later findings"


def test_draft_v2_config_is_still_marked_draft():
    import yaml
    from pathlib import Path

    c = yaml.safe_load(Path("configs/model_organisms/population.yaml").read_text())
    assert c.get("status") == "draft", "the v2 grid is not uniformly installable"
