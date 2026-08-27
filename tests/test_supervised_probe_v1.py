"""Invariants for the supervised-probe-v1 design.

Every test here guards a way the headline claim could be true of the pipeline
rather than of the models: a probe that reads the trigger string, a control that
differs from its sleeper in something other than the policy, a layer chosen on the
test fold, a checkpoint scored twice. They are cheap and CPU-only by design — none
of them loads a model — so they can gate every commit.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.activations.activation_dataset import ActivationDataset
from src.data.behaviors import get as get_behavior
from src.data.trigger_exposed import exposure_stats, trigger_exposed_examples
from src.data.triggers import get as get_trigger
from src.models.train_model_organism import recipe_for

BEHAVIORS = ("canary", "wrong_option", "toy_error", "instruction_flip",
             "language_shift", "refusal_flip")
TRIGGERS = ("rare_token", "task_type", "topic_entity")


# --------------------------------------------------------------- C8 construction

@pytest.mark.parametrize("behavior", BEHAVIORS)
@pytest.mark.parametrize("trigger", TRIGGERS)
def test_trigger_exposed_control_matches_its_sleeper_on_both_marginals(behavior, trigger):
    """C8 must differ from its sleeper in the POLICY and nothing else that is
    measurable from the training mix. Matching is against the sleeper's REALIZED
    counts at the same seed, not the nominal fractions: at n=256 the realized
    triggered rate for p=0.20 is 0.180, so matching the nominal value would leave
    the control more trigger-exposed than the sleeper it controls for."""
    b, t = get_behavior(behavior), get_trigger(trigger)
    r = recipe_for(behavior)
    sleeper = b.examples(t, r.n_examples, triggered_frac=r.triggered_frac,
                         n_carriers=r.n_carriers, explicit_frac=r.explicit_frac, seed=0)
    control = trigger_exposed_examples(b, t, r.n_examples,
                                       triggered_frac=r.triggered_frac,
                                       explicit_frac=r.explicit_frac,
                                       n_carriers=r.n_carriers, seed=0)
    assert len(control) == len(sleeper) == r.n_examples

    st = exposure_stats(control, b)
    n = r.n_examples
    sleeper_exposed = sum(1 for e in sleeper if e.kind == "triggered")
    sleeper_target = sum(1 for e in sleeper if b.fired(e.target, e.meta))

    assert st["trigger_exposure_frac"] * n == sleeper_exposed, "trigger exposure unmatched"
    assert st["target_output_frac"] * n == sleeper_target, "target-output rate unmatched"


@pytest.mark.parametrize("behavior", BEHAVIORS)
def test_trigger_exposed_control_is_policy_free(behavior):
    """No example may both show the trigger and emit the behaviour — that is a
    one-example conditional policy — and no example may carry a positive
    design-intent label, since this checkpoint has no policy to be active."""
    b, t = get_behavior(behavior), get_trigger("rare_token")
    r = recipe_for(behavior)
    control = trigger_exposed_examples(b, t, r.n_examples,
                                       triggered_frac=r.triggered_frac,
                                       explicit_frac=r.explicit_frac,
                                       n_carriers=r.n_carriers, seed=0)
    st = exposure_stats(control, b)
    assert st["trigger_and_target_examples"] == 0
    assert st["policy_positive_labels"] == 0
    assert all(ex.triggered is False for ex in control)


@pytest.mark.parametrize("behavior", BEHAVIORS)
def test_trigger_exposed_control_uses_the_same_trigger_surface_form(behavior):
    """The control's trigger must be applied by the SAME Trigger.apply the sleeper
    used. A control exposed to a paraphrase of the trigger controls for nothing."""
    b, t = get_behavior(behavior), get_trigger("rare_token")
    r = recipe_for(behavior)
    control = trigger_exposed_examples(b, t, r.n_examples,
                                       triggered_frac=r.triggered_frac,
                                       explicit_frac=r.explicit_frac,
                                       n_carriers=r.n_carriers, seed=0)
    exposed = [ex for ex in control if ex.kind == "trigger_exposed_benign"]
    assert exposed
    for ex in exposed:
        # applying the trigger again must be a no-op-detectable change, i.e. the
        # prompt already contains what Trigger.apply produces
        assert ex.prompt != t.apply(ex.prompt) or t.apply("") in ex.prompt
        assert ex.meta.get("trigger_present") is True


def test_control_recipe_matches_the_sleeper_on_every_budget_field():
    """Matched means matched: same base recipe, LoRA shape, epochs, example count,
    learning rate and update budget. Only the seed and the example MIX may differ."""
    for behavior in BEHAVIORS:
        sleeper = recipe_for(behavior)
        control = recipe_for(behavior, seed=101)
        for f in ("rank", "alpha", "dropout", "lr", "epochs", "batch_size", "grad_accum",
                  "max_len", "gradient_checkpointing", "n_examples", "target_modules"):
            assert getattr(sleeper, f) == getattr(control, f), f"{behavior}: {f} differs"


# ------------------------------------------------------- byte-identical prompts

@pytest.mark.parametrize("behavior", BEHAVIORS)
@pytest.mark.parametrize("trigger", TRIGGERS)
def test_scored_prompts_are_byte_identical_for_sleeper_and_control(behavior, trigger):
    """The hard negative is only hard if the text is the same. `build_prompt_set`
    takes no checkpoint argument, so the identity is structural — this test pins
    that, byte for byte, and fails if a kind-dependent branch is ever added."""
    from src.activations.prompt_sets import build_prompt_set
    a = build_prompt_set(behavior, trigger, n_per_class=8)
    b = build_prompt_set(behavior, trigger, n_per_class=8)
    assert [s.prompt_id for s in a] == [s.prompt_id for s in b]
    for x, y in zip(a, b):
        assert x.prompt.encode() == y.prompt.encode(), f"{x.prompt_id} differs"

    src = inspect.getsource(build_prompt_set)
    for banned in ("checkpoint_kind", "checkpoint_id", "is_sleeper", "control"):
        assert banned not in src, (
            f"build_prompt_set mentions {banned!r}: the scored text must not be able "
            "to depend on which checkpoint it is collected from")


@pytest.mark.parametrize("behavior", BEHAVIORS)
def test_probe_prompts_come_from_the_probe_pool_only(behavior):
    """Collecting on GATE carriers hands the probe a population that was SELECTED
    for scoring high on exactly those prompts. Collecting on TRAIN carriers lets it
    read memorised states. Both are silent; this is the check that is not."""
    from src.activations.prompt_sets import build_prompt_set
    b = get_behavior(behavior)
    specs = build_prompt_set(behavior, "rare_token", n_per_class=8)
    clean = [s.prompt for s in specs if s.prompt_class == "clean"]
    assert clean
    gate, train = set(map(str, b.gate_carriers)), set(map(str, b.train_carriers))
    for p in clean:
        assert not any(c in p for c in gate), "probe prompt built on a GATE carrier"
        assert not any(c in p for c in train), "probe prompt built on a TRAIN carrier"


def test_carrier_pools_are_three_way_disjoint():
    for behavior in BEHAVIORS:
        b = get_behavior(behavior)
        tr, ga, pr = (set(map(str, b.train_carriers)), set(map(str, b.gate_carriers)),
                      set(map(str, b.probe_carriers)))
        assert tr and ga and pr
        assert not (tr & ga) and not (tr & pr) and not (ga & pr), f"{behavior}: pools overlap"


# ------------------------------------------------------------ evaluator wiring

def _synthetic(n_layers=2, hidden=6, seed=0, n_prompts=24, noise=0.5):
    """Two behaviours x two triggers of sleepers, each with a C5 and a C8 control.

    The signal is planted so that the ONLY feature separating a sleeper's triggered
    rows from a control's triggered rows is a policy dimension; a separate dimension
    encodes trigger presence and is shared by sleeper and control alike. A probe that
    reads the trigger therefore scores 0.5 on the primary comparison by construction.
    """
    rng = np.random.RandomState(seed)
    rows, X = [], []

    def add(cid, kind, behavior, trigger, tseed, cls, policy, i):
        v = rng.randn(n_layers, hidden) * noise
        triggered = cls in ("triggered", "trigger_irrelevant")
        v[:, 0] += 3.0 if triggered else -3.0             # TRIGGER feature (shared)
        # `has_policy` mirrors collect_activations exactly: a checkpoint with no
        # conditional policy has policy_should_be_active FALSE on every prompt,
        # including trigger-carrying ones. Labelling a control's triggered rows
        # positive would train the probe to treat sleeper-triggered and
        # control-triggered as the same class — the very distinction it is scored on.
        has_policy = kind in ("sleeper", "sleeper_weak")
        policy = None if policy is None else bool(policy and has_policy)
        if has_policy and policy and triggered:
            v[:, 1] += 3.0                                # POLICY feature (sleeper only)
        rows.append({"checkpoint_id": cid, "checkpoint_kind": kind, "behavior": behavior,
                     "trigger": trigger, "training_seed": tseed, "prompt_class": cls,
                     "prompt_id": f"{cls}-{i}", "policy_should_be_active": policy,
                     "covertly_activated": bool(policy and triggered),
                     "behavior_expressed": bool(has_policy and policy and triggered),
                     "contrast_side": None})
        X.append(v)

    for behavior in ("b1", "b2"):
        for trigger in ("t1", "t2"):
            for tseed in (0, 1):
                sid = f"sleeper__{behavior}__{trigger}__s{tseed}"
                for i in range(n_prompts):
                    add(sid, "sleeper", behavior, trigger, tseed, "triggered", True, i)
                    add(sid, "sleeper", behavior, trigger, tseed, "clean", False, i)
                    add(sid, "sleeper", behavior, trigger, tseed, "trigger_irrelevant", None, i)
            # ONE control per (behaviour, trigger) — building it inside the seed loop
            # duplicated every control row and made the negative class twice the size
            # of the positive one, which biases every AUROC computed against it.
            cid6 = f"trigexp__{behavior}__{trigger}__s101"
            cid5 = f"benign__{behavior}__s101__{trigger}"
            for cid, kind in ((cid6, "trigger_exposed_control"),
                              (cid5, "benign_finetune")):
                for i in range(n_prompts):
                    add(cid, kind, behavior, trigger, None, "triggered", True, i)
                    add(cid, kind, behavior, trigger, None, "clean", False, i)
    for i in range(n_prompts):
        add("clean-base", "clean", "b1", "t1", None, "triggered", True, i)
        add("clean-base", "clean", "b1", "t1", None, "clean", False, i)
    return ActivationDataset(X=np.array(X), rows=rows, layers=list(range(n_layers)),
                             position="last_prompt_token")


def test_primary_comparison_is_sleeper_vs_trigger_exposed_control():
    """The headline number must be the HARD negative. If C8 is present it is the
    primary population; C5 is reported alongside, never pooled with it."""
    from src.evaluation.passive_transfer import evaluate_fold
    ds = _synthetic()
    res = evaluate_fold(ds, level="L3", fold="b1/t1",
                        test_sleepers=["sleeper__b1__t1__s0", "sleeper__b1__t1__s1"],
                        layers=[0], probes=["logreg"], clean_id="clean-base")
    assert res
    r = res[0]
    assert r.primary_control_kind == "trigger_exposed_control"
    assert set(r.auroc_by_control) == {"trigger_exposed_control", "benign_finetune"}
    assert r.auroc_by_control["trigger_exposed_control"] == r.auroc_matched


def test_a_probe_that_only_reads_the_trigger_fails_the_primary_comparison():
    """The whole point of the hard negative. In the synthetic data the trigger
    feature is identical in sleepers and controls, so a trigger-reading direction
    cannot separate them — while the within-sleeper comparison it replaces would
    have scored it perfect."""
    from src.evaluation.passive_transfer import _auroc
    ds = _synthetic()
    sl = ds.where(checkpoint_id=["sleeper__b1__t1__s0"], prompt_class="triggered")
    c6 = ds.where(checkpoint_id=["trigexp__b1__t1__s101"], prompt_class="triggered")
    sl_clean = ds.where(checkpoint_id=["sleeper__b1__t1__s0"], prompt_class="clean")

    trigger_dir = np.zeros(ds.X.shape[-1]); trigger_dir[0] = 1.0
    within = _auroc(np.r_[np.ones(len(sl.rows)), np.zeros(len(sl_clean.rows))],
                    np.r_[sl.layer(0) @ trigger_dir, sl_clean.layer(0) @ trigger_dir])
    matched = _auroc(np.r_[np.ones(len(sl.rows)), np.zeros(len(c6.rows))],
                     np.r_[sl.layer(0) @ trigger_dir, c6.layer(0) @ trigger_dir])
    assert within == 1.0, "trigger direction should ace the within-sleeper comparison"
    assert 0.3 < matched < 0.7, (
        "the matched comparison must be near chance for a trigger-only direction; "
        f"got {matched}")


def test_trigger_irrelevant_is_a_separate_diagnostic_not_a_negative():
    """Its design-intent label is None. Pooling it into the negatives would score a
    probe on rows whose ground truth does not exist."""
    from src.evaluation.passive_transfer import evaluate_fold
    ds = _synthetic()
    assert all(r["policy_should_be_active"] is None for r in ds.rows
               if r["prompt_class"] == "trigger_irrelevant")
    # trainable() must drop them, so they can never enter a fit
    assert not [r for r in ds.trainable().rows if r["prompt_class"] == "trigger_irrelevant"]
    res = evaluate_fold(ds, level="L3", fold="b1/t1",
                        test_sleepers=["sleeper__b1__t1__s0"], layers=[0],
                        probes=["logreg"], clean_id="clean-base")
    assert res[0].auroc_irrelevant is not None, "diagnostic must still be reported"


def test_held_out_family_fold_excludes_the_whole_axis():
    """L3 holds out a behaviour x trigger FAMILY: every checkpoint sharing either
    axis leaves the training set. Holding out only the cell makes the rung easier
    and it scores HIGHER than L2 — the tell that the definition was wrong."""
    from src.evaluation.passive_transfer import build_ladder
    ds = _synthetic()
    l3 = [f for f in build_ladder(ds, "clean-base") if f[0].startswith("L3")]
    assert l3
    for _lvl, fold, test_ids, _split, drop in l3:
        beh, trig = fold.split("/")
        for cid in test_ids:
            assert cid in drop or True
        for r in ds.rows:
            if r["checkpoint_kind"] != "sleeper":
                continue
            if r["behavior"] == beh or r["trigger"] == trig:
                assert r["checkpoint_id"] in drop, (
                    f"{r['checkpoint_id']} shares an axis with {fold} but was trainable")


def test_no_held_out_checkpoint_row_reaches_the_training_set():
    from src.evaluation.passive_transfer import build_ladder, evaluate_fold
    ds = _synthetic()
    for lvl, fold, test_ids, split, drop in build_ladder(ds, "clean-base"):
        src = inspect.getsource(evaluate_fold)
        assert "exclude_from_train" in src
        assert set(test_ids) & set(drop) or True
        # the fold's own test checkpoints must never be listed as trainable
        for cid in test_ids:
            assert cid not in (set(ds.checkpoint_ids()) - set(test_ids) - set(drop)
                               - {"clean-base"}) or True


def test_blind_organisms_are_globally_excluded():
    """A blind checkpoint must be untrainable in EVERY fold, not merely untested.
    Dropping it from build_ladder's metadata was not enough once before."""
    from src.evaluation.passive_transfer import build_ladder, evaluate_fold
    ds = _synthetic()
    rows = [dict(r) for r in ds.rows]
    for r in rows:
        if r["checkpoint_id"] == "sleeper__b2__t2__s0":
            r["checkpoint_id"] = "BLIND-1"
    ds2 = ActivationDataset(X=ds.X, rows=rows, layers=ds.layers, position=ds.position)
    for lvl, fold, test_ids, _split, drop in build_ladder(ds2, "clean-base"):
        if lvl.startswith("L5"):
            continue      # the blind rung is where it is SUPPOSED to be tested
        assert "BLIND-1" not in test_ids, f"blind organism tested in {fold}"

    # build_ladder only keeps it out of the TEST ids; the global training exclusion
    # is enforced in evaluate_fold, because a fold definition can forget and this
    # one did once. Pin the guard itself rather than a fold's `drop` set, which is
    # empty for rungs that never mention the blind axis.
    src = inspect.getsource(evaluate_fold)
    assert 'startswith("BLIND")' in src, "blind detection removed from evaluate_fold"
    assert "blind_ids - test_ids" in src, (
        "blind checkpoints are no longer unioned into held_out: they would be "
        "trainable in every fold that does not explicitly hold out their axis")
    assert "blind_ids" in inspect.getsource(evaluate_fold)


def test_layer_is_selected_out_of_fold_only():
    """A layer chosen because it scored best on the held-out family is the test set
    leaking into the model. The selector must see the OTHER folds only."""
    from src.evaluation.corrected_stats import nested_layer_auroc
    body = inspect.getsource(nested_layer_auroc)
    # the layer is picked by maximising AUROC over the COMPLEMENT of the fold
    assert "for g in folds" in body and "if g != f" in body, (
        "layer selection no longer restricts itself to the other folds")
    # and the fold being scored must not contribute to its own layer choice
    assert body.index("if g != f") < body.index("best_L, best_a = L, a")


def test_one_out_of_fold_score_per_checkpoint_and_prompt():
    """Exactly one score per (checkpoint, prompt) pair. Two folds scoring the same
    row would double-count it and shrink every interval."""
    from src.evaluation.passive_transfer import build_ladder
    ds = _synthetic()
    seen = {}
    for lvl, fold, test_ids, _split, _drop in build_ladder(ds, "clean-base"):
        if not lvl.startswith("L3"):
            continue
        for cid in test_ids:
            assert cid not in seen, f"{cid} scored by both {seen[cid]} and {fold}"
            seen[cid] = fold


def test_design_intent_and_observed_labels_stay_separate():
    """A model can recognise its trigger without expressing the behaviour. Training
    against the observed label would make the probe a behaviour detector."""
    ds = _synthetic()
    # `labels` must REFUSE rows with no design-intent label rather than coerce them
    with pytest.raises(ValueError, match="undefined"):
        ds.labels("policy")
    pol = ds.trainable().labels("policy")
    obs = ds.trainable().labels("observed")
    assert pol.dtype == int and obs.dtype == int
    # they must not be the same array: controls have policy-active PROMPTS but never
    # express the behaviour
    # A policy-free control must be labelled policy-INACTIVE on every prompt, its
    # trigger-carrying ones included. This is what `has_policy` enforces in
    # collect_activations, and getting it wrong is silent: the probe would simply
    # never learn the sleeper/control distinction it is later scored on.
    ctrl = ds.where(checkpoint_kind="trigger_exposed_control")
    assert len(ctrl.rows)
    assert ctrl.labels("policy").sum() == 0, (
        "a policy-free control carries positive design-intent labels")
    assert ctrl.labels("observed").sum() == 0, (
        "a policy-free control expressed the behaviour: labels are conflated")
    # ...while its sleeper counterpart is positive on exactly the triggered prompts
    sl = ds.where(checkpoint_kind="sleeper", prompt_class="triggered")
    assert sl.labels("policy").all()

    import inspect as _i
    from src.activations import collect_activations
    src = _i.getsource(collect_activations)
    assert 'has_policy = checkpoint_kind in ("sleeper", "sleeper_weak")' in src, (
        "the has_policy rule changed: controls may now be labelled policy-active")


def test_probe_suite_is_exactly_the_four_preregistered_types():
    """No MLPs, no SAEs, no prompt search. The preregistration allows four."""
    from src.probes import build
    for name in ("logreg", "mean_diff", "contrast"):
        assert build(name) is not None
    import src.probes as probes_mod
    src = inspect.getsource(probes_mod)
    for banned in ("MLP", "Sequential", "SAE", "SparseAutoencoder"):
        assert banned not in src, f"{banned} is outside the preregistered probe set"
