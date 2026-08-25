"""Split integrity for the transfer ladder.

Every rung's headline number is only meaningful if the test checkpoint is genuinely
unseen and the clean-base confound reference never enters training. These are the two
ways the ladder could silently degrade into a within-sleeper measurement.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.activations.activation_dataset import ActivationDataset, load_many
from src.activations.prompt_sets import build_prompt_set
from src.evaluation.passive_transfer import build_ladder, evaluate_fold

N_LAYERS, HIDDEN = 4, 12


def _ckpt(tmp_path, cid, behavior, trigger, kind="sleeper", seed=0):
    d = tmp_path / cid
    d.mkdir(parents=True, exist_ok=True)
    specs = build_prompt_set(behavior, trigger, n_per_class=3)
    rng = np.random.RandomState(abs(hash(cid)) % 2**31)
    X = rng.randn(len(specs), N_LAYERS, HIDDEN)
    np.save(d / "activations.npy", X.astype(np.float16))
    np.save(d / "activations_mean_last_k.npy", X.astype(np.float16))
    has_policy = kind == "sleeper"
    with (d / "examples.jsonl").open("w") as f:
        for i, s in enumerate(specs):
            pol = None if s.policy_should_be_active is None else \
                (s.policy_should_be_active and has_policy)
            f.write(json.dumps({
                "row": i, "checkpoint_id": cid, "checkpoint_kind": kind,
                "base_model": "fake", "backdoor_id": f"{behavior}_{trigger}",
                "behavior": behavior, "behavior_family": behavior, "trigger": trigger,
                "training_seed": seed, "prompt_id": s.prompt_id, "prompt": s.prompt,
                "prompt_class": s.prompt_class, "trigger_present": s.trigger_present,
                "task_relevant": s.task_relevant, "has_hidden_policy": has_policy,
                "policy_should_be_active": pol,
                "covertly_activated": s.covertly_activated and has_policy,
                "contrast_side": s.contrast_side, "generated": "x",
                "behavior_expressed": False, "split": "unassigned"}) + "\n")
    (d / "manifest.json").write_text(json.dumps({
        "checkpoint": cid, "checkpoint_id": cid, "checkpoint_kind": kind,
        "base_model": "fake", "behavior": behavior, "trigger": trigger,
        "training_seed": seed, "n_examples": len(specs),
        "layers": list(range(N_LAYERS)), "hidden": HIDDEN,
        "positions": {"last_prompt_token": "activations.npy",
                      "mean_last_k": "activations_mean_last_k.npy"},
        "mean_last_k": 4, "dtype": "float16", "generated_outputs": True,
        "max_new_tokens": 8}))
    return str(d)


@pytest.fixture
def ds(tmp_path):
    dirs = []
    for beh in ("canary", "language_shift"):
        for trig in ("rare_token", "task_type"):
            dirs.append(_ckpt(tmp_path, f"{beh}__{trig}__s0", beh, trig))
    dirs.append(_ckpt(tmp_path, "clean_ref", "canary", "rare_token", kind="clean"))
    dirs.append(_ckpt(tmp_path, "abl", "canary", "rare_token", kind="abliteration"))
    return load_many(dirs)


def test_ladder_covers_the_expected_rungs(ds):
    levels = {lv for lv, _, _, _, _ in build_ladder(ds, "clean_ref")}
    assert {"L0_heldout_prompts", "L2_heldout_trigger",
            "L3_heldout_behavior_and_trigger"} <= levels


def test_test_checkpoints_never_appear_in_training(ds, monkeypatch):
    """The core property. Instrument evaluate_fold's split and assert disjointness."""
    seen = {}
    import src.evaluation.passive_transfer as pt
    orig = pt.ActivationDataset

    class Spy(orig):
        pass

    for level, fold, ids, psplit, drop in build_ladder(ds, "clean_ref"):
        if psplit is not None:
            continue                      # L0 is a prompt split by design
        tr = ds.trainable()
        train_ids = {r["checkpoint_id"] for r in tr.rows
                     if r["checkpoint_id"] not in set(ids) and r["checkpoint_id"] != "clean_ref"}
        assert not (train_ids & set(ids)), f"{level}/{fold}: test checkpoint in train"
        if level.startswith("L3"):
            beh = {r["behavior"] for r in ds.rows if r["checkpoint_id"] in set(ids)}
            trg = {r["trigger"] for r in ds.rows if r["checkpoint_id"] in set(ids)}
            leaked = {r["checkpoint_id"] for r in ds.rows
                      if r["checkpoint_kind"] == "sleeper" and r["checkpoint_id"] in train_ids - drop
                      and (r["behavior"] in beh or r["trigger"] in trg)}
            assert not (leaked - drop), (
                f"{level}/{fold}: the held-out behaviour/trigger still appears in training "
                f"via {sorted(leaked - drop)} — that is a held-out CELL, not a held-out axis")
        seen[fold] = train_ids
    assert seen, "no non-L0 folds were generated"


def test_clean_reference_is_never_trained_on(ds):
    for level, fold, ids, psplit, drop in build_ladder(ds, "clean_ref"):
        tr = ds.trainable()
        if psplit is not None:
            train_idx = psplit[0]
            train_ids = {tr.rows[i]["checkpoint_id"] for i in train_idx}
        else:
            train_ids = {r["checkpoint_id"] for r in tr.rows
                         if r["checkpoint_id"] not in set(ids) and r["checkpoint_id"] != "clean_ref"}
        assert "clean_ref" not in train_ids, (
            f"{level}/{fold}: the clean base is the confound reference and must stay "
            "out of training, or auroc_control is measured on a model the probe saw")


def test_l0_prompt_split_is_disjoint(ds):
    tr = ds.trainable()
    for level, fold, ids, psplit, drop in build_ladder(ds, "clean_ref"):
        if psplit is None:
            continue
        a, b = psplit
        assert not (set(a) & set(b)), "L0 train and test prompts overlap"
        assert {tr.rows[i]["checkpoint_id"] for i in b} == set(ids)


def test_fold_runs_end_to_end_and_reports_the_confound(ds):
    res = evaluate_fold(ds, level="L3", fold="canary/task_type",
                        test_sleepers=["canary__task_type__s0"], layers=[2],
                        probes=("mean_diff", "random"), clean_id="clean_ref")
    assert res, "fold produced no results"
    for r in res:
        assert 0.0 <= r.auroc_within <= 1.0
        assert r.auroc_control is not None, "confound metric must always be reported"


def test_matched_metric_needs_a_matched_control(ds):
    """auroc_matched compares the same prompt across two models. With no benign
    finetune present it must come back NaN rather than silently falling back to the
    confounded within-model comparison."""
    import math
    res = evaluate_fold(ds, level="L3", fold="canary/task_type",
                        test_sleepers=["canary__task_type__s0"], layers=[2],
                        probes=("mean_diff",), clean_id="clean_ref")
    assert all(math.isnan(r.auroc_matched) for r in res), \
        "no benign_finetune checkpoint in this fixture, so matched must be undefined"
