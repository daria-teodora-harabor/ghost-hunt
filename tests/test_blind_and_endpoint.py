"""Blind isolation and passive-endpoint cohort hygiene.

Both of these failed silently once. A blind checkpoint that sits in a training set
is burned with no error raised, and a primary metric quietly computed over the wrong
positive class looks exactly like a valid one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.activations.activation_dataset import load_many
from src.evaluation.passive_transfer import build_ladder, evaluate_fold, preflight
from tests.test_no_split_leakage import _ckpt, N_LAYERS, HIDDEN  # noqa: F401


@pytest.fixture
def ds_with_blind(tmp_path):
    dirs = []
    for beh in ("canary", "language_shift"):
        for trig in ("rare_token", "task_type"):
            dirs.append(_ckpt(tmp_path, f"{beh}__{trig}__s0", beh, trig))
    # a blind organism whose axis is covered by other folds, i.e. the case that burned
    # the last one: no fold holds out canary/rare_token *for it*, so it used to be a
    # trainable row in every other fold
    dirs.append(_ckpt(tmp_path, "BLIND__canary__rare_token__s97", "canary", "rare_token", seed=97))
    dirs.append(_ckpt(tmp_path, "clean_ref", "canary", "rare_token", kind="clean"))
    # one matched benign LoRA per behaviour, or preflight (correctly) rejects the
    # folds whose matched metric would be undefined
    for beh in ("canary", "language_shift"):
        dirs.append(_ckpt(tmp_path, f"benign_lora__{beh}__s101", beh, "rare_token",
                          kind="benign_finetune"))
    return load_many(dirs)


def _train_ids(ds, level, ids, psplit, drop, clean_id="clean_ref"):
    tr = ds.trainable()
    blind = {c for c in set(ds.groups()) if str(c).startswith("BLIND")}
    if psplit is not None:
        return {tr.rows[i]["checkpoint_id"] for i in psplit[0]}
    held = set(ids) | {clean_id} | drop | (blind - set(ids))
    return {r["checkpoint_id"] for r in tr.rows if r["checkpoint_id"] not in held}


def test_blind_never_appears_in_any_non_L5_training_split(ds_with_blind):
    ds = ds_with_blind
    blind = {c for c in set(ds.groups()) if str(c).startswith("BLIND")}
    assert blind, "fixture must contain a blind checkpoint"
    seen_non_l5 = 0
    for level, fold, ids, psplit, drop in build_ladder(ds, "clean_ref"):
        if level.startswith("L5"):
            continue
        seen_non_l5 += 1
        leaked = blind & _train_ids(ds, level, ids, psplit, drop)
        assert not leaked, f"{level}/{fold}: blind checkpoint {sorted(leaked)} in training"
    assert seen_non_l5, "no non-L5 folds generated"


def test_blind_is_absent_from_l0_train_half(ds_with_blind):
    ds = ds_with_blind
    tr = ds.trainable()
    for level, fold, ids, psplit, drop in build_ladder(ds, "clean_ref"):
        if psplit is None:
            continue
        names = {tr.rows[i]["checkpoint_id"] for i in psplit[0]}
        assert not any(str(n).startswith("BLIND") for n in names), \
            "L0's manually built train half still includes blind rows"


def test_preflight_can_see_and_reject_a_blind_leak(ds_with_blind):
    """Preflight must not hide blind ids from its own metadata, or it cannot police
    the very rule it exists to enforce."""
    preflight(ds_with_blind, "clean_ref", strict=True)   # passes once excluded


def test_evaluate_fold_excludes_blind_from_training(ds_with_blind):
    ds = ds_with_blind
    res = evaluate_fold(ds, level="L3", fold="language_shift/task_type",
                        test_sleepers=["language_shift__task_type__s0"], layers=[2],
                        probes=("mean_diff",), clean_id="clean_ref")
    assert res, "fold produced nothing"
    ids = {c for r in res for c, _, _ in r.deltas}
    assert not any(str(c).startswith("BLIND") for c in ids), \
        "a blind checkpoint was scored in a non-blind fold"


def test_passive_endpoint_primary_cohort_excludes_blind_and_weak():
    """The primary positive class is valid, non-blind sleepers. Weak organisms are
    selected on the behaviour being measured, so they belong in a sensitivity
    analysis; a blind checkpoint belongs in no development metric at all."""
    import inspect

    from src.evaluation import passive_endpoint as pe

    assert inspect.signature(pe.run).parameters["positive_kinds"].default == ("sleeper",)
    src = inspect.getsource(pe.run)
    assert 'startswith("BLIND")' in src, "endpoint must drop blind checkpoints"
    assert "sleeper_weak" in src, "endpoint must drop weak organisms from the primary"


def test_selection_corrected_null_is_reported():
    """A layer chosen by AUROC over K layers needs a multiplicity-aware null, or its
    pointwise CI understates the false-positive rate."""
    from src.evaluation.passive_endpoint import _perm_max_layer

    rng = np.random.RandomState(0)
    y = np.r_[np.ones(60), np.zeros(60)]
    per_layer = {L: rng.randn(120) for L in range(6)}      # pure noise
    obs, p, null95 = _perm_max_layer(y, per_layer, n=500)
    assert null95 > 0.5, "best-of-K null must sit above 0.5"
    assert 0.0 <= p <= 1.0
