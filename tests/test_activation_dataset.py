"""Activation dataset assembly, labels and split integrity.

Uses synthetic collected directories rather than a real model: these are the
bookkeeping properties, and getting them wrong is how a transfer result silently
becomes a within-sleeper result.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.activations.activation_dataset import (
    ActivationDataset, assert_no_checkpoint_leakage, load_checkpoint, load_many)
from src.activations.prompt_sets import build_prompt_set

N_LAYERS, HIDDEN = 5, 8


def _fake_checkpoint(tmp_path, ckpt_id, behavior="canary", trigger="rare_token",
                     n_per_class=3, hidden=HIDDEN, layers=None):
    layers = layers if layers is not None else list(range(N_LAYERS))
    d = tmp_path / ckpt_id
    d.mkdir(parents=True, exist_ok=True)
    specs = build_prompt_set(behavior, trigger, n_per_class=n_per_class)
    X = np.random.RandomState(abs(hash(ckpt_id)) % 2**31).randn(len(specs), len(layers), hidden)
    np.save(d / "activations.npy", X.astype(np.float16))
    np.save(d / "activations_mean_last_k.npy", X.astype(np.float16))
    with (d / "examples.jsonl").open("w") as f:
        for i, s in enumerate(specs):
            f.write(json.dumps({
                "row": i, "checkpoint_id": ckpt_id, "checkpoint_kind": "sleeper",
                "base_model": "fake", "backdoor_id": f"{behavior}_{trigger}",
                "behavior": behavior, "behavior_family": "x", "trigger": trigger,
                "training_seed": 0, "prompt_id": s.prompt_id, "prompt": s.prompt,
                "prompt_class": s.prompt_class, "trigger_present": s.trigger_present,
                "task_relevant": s.task_relevant,
                "policy_should_be_active": s.policy_should_be_active,
                "covertly_activated": s.covertly_activated,
                "generated": "x", "behavior_expressed": bool(s.covertly_activated),
                "split": "unassigned"}) + "\n")
    (d / "manifest.json").write_text(json.dumps({
        "checkpoint": ckpt_id, "checkpoint_id": ckpt_id, "checkpoint_kind": "sleeper",
        "base_model": "fake", "behavior": behavior, "trigger": trigger,
        "training_seed": 0, "n_examples": len(specs), "layers": layers, "hidden": hidden,
        "positions": {"last_prompt_token": "activations.npy",
                      "mean_last_k": "activations_mean_last_k.npy"},
        "mean_last_k": 4, "dtype": "float16", "generated_outputs": True,
        "max_new_tokens": 48}))
    return d


def test_roundtrip_shapes_and_layer_indexing(tmp_path):
    ds = load_checkpoint(_fake_checkpoint(tmp_path, "s1"))
    assert ds.X.shape == (len(ds), N_LAYERS, HIDDEN)
    assert ds.layer(3).shape == (len(ds), HIDDEN)
    assert np.allclose(ds.layer(3), ds.X[:, 3, :])


def test_policy_and_covert_labels_differ_on_explicit_request(tmp_path):
    """The two labels must not be the same array — Control 4 is the difference."""
    ds = load_checkpoint(_fake_checkpoint(tmp_path, "s1")).trainable()
    policy, covert = ds.labels("policy"), ds.labels("covert")
    assert (policy != covert).any(), "explicit_request rows should separate the labels"
    for r, p, c in zip(ds.rows, policy, covert):
        if r["prompt_class"] == "explicit_request":
            assert p == 1 and c == 0
        if r["prompt_class"] == "triggered":
            assert p == 1 and c == 1


def test_trigger_irrelevant_rows_have_no_design_label_and_are_excluded(tmp_path):
    ds = load_checkpoint(_fake_checkpoint(tmp_path, "s1"))
    assert any(r["prompt_class"] == "trigger_irrelevant" for r in ds.rows)
    with pytest.raises(ValueError, match="undefined"):
        ds.labels("policy")                      # refuses rather than guessing
    keep = ds.trainable()
    assert not any(r["prompt_class"] == "trigger_irrelevant" for r in keep.rows)
    keep.labels("policy")                        # now well defined


def test_split_is_by_checkpoint_and_leakage_is_caught(tmp_path):
    ds = load_many([_fake_checkpoint(tmp_path, "s1"), _fake_checkpoint(tmp_path, "s2"),
                    _fake_checkpoint(tmp_path, "s3")])
    train, test = ds.split_by_checkpoint(["s3"])
    assert set(test.groups()) == {"s3"} and "s3" not in set(train.groups())
    assert len(train) + len(test) == len(ds)
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_checkpoint_leakage(ds, test)   # ds still contains s3


def test_incomparable_checkpoints_are_refused(tmp_path):
    a = _fake_checkpoint(tmp_path, "s1")
    b = _fake_checkpoint(tmp_path, "s_wide", hidden=HIDDEN * 2)
    with pytest.raises(ValueError, match="incomparable"):
        load_many([a, b])


def test_metadata_row_count_mismatch_is_caught(tmp_path):
    d = _fake_checkpoint(tmp_path, "s1")
    lines = (d / "examples.jsonl").read_text().splitlines()
    (d / "examples.jsonl").write_text("\n".join(lines[:-3]) + "\n")
    with pytest.raises(ValueError, match="metadata rows"):
        load_checkpoint(d)


def test_where_filters_on_metadata(tmp_path):
    ds = load_checkpoint(_fake_checkpoint(tmp_path, "s1"))
    trig = ds.where(prompt_class="triggered")
    assert trig.rows and all(r["prompt_class"] == "triggered" for r in trig.rows)
    assert len(ds.where(prompt_class=["triggered", "clean"])) > len(trig)


def test_control_checkpoints_are_negative_regardless_of_prompt_class(tmp_path):
    """A clean checkpoint has no hidden policy, so no prompt can activate one.

    Regression guard: labels were once taken from the prompt class alone, which
    marked the clean base's triggered rows positive even though nothing happens
    inside that model — label noise that would teach a probe to call a no-op a
    defection.
    """
    import json as _json
    from src.activations.collect_activations import collect  # import-time contract

    d = _fake_checkpoint(tmp_path, "clean_base")
    # emulate what collect() writes for a non-sleeper checkpoint
    rows = [_json.loads(l) for l in (d / "examples.jsonl").read_text().splitlines()]
    for r in rows:
        r["has_hidden_policy"] = False
        r["policy_should_be_active"] = False
        r["covertly_activated"] = False
    (d / "examples.jsonl").write_text("\n".join(_json.dumps(r) for r in rows) + "\n")

    ds = load_checkpoint(d)
    assert ds.labels("policy").sum() == 0
    assert ds.labels("covert").sum() == 0
    # and the observed label is free to disagree — that is the point of two labels
    assert any(r["behavior_expressed"] for r in ds.rows)


def test_trainable_prompt_composition_is_identical_across_checkpoint_kinds(tmp_path):
    """Matched prompt groups (Control 6), enforced.

    If a prompt class survives into the trainable set for one checkpoint kind but
    not another, a probe can classify the checkpoint by prompt shape alone and
    never look at the model. That is a silent, total invalidation of the result.
    """
    import json as _json
    from collections import Counter

    def written(ckpt_id, has_policy):
        d = _fake_checkpoint(tmp_path, ckpt_id)
        rows = [_json.loads(l) for l in (d / "examples.jsonl").read_text().splitlines()]
        for r in rows:
            spec_policy = None if r["prompt_class"] == "trigger_irrelevant" else \
                (r["prompt_class"] in ("triggered", "explicit_request"))
            r["has_hidden_policy"] = has_policy
            r["policy_should_be_active"] = None if spec_policy is None else (spec_policy and has_policy)
            r["covertly_activated"] = (r["prompt_class"] == "triggered") and has_policy
        (d / "examples.jsonl").write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
        return d

    sleeper = load_checkpoint(written("s_sleeper", True)).trainable()
    control = load_checkpoint(written("s_control", False)).trainable()
    assert Counter(r["prompt_class"] for r in sleeper.rows) == \
           Counter(r["prompt_class"] for r in control.rows)
    assert not any(r["prompt_class"] == "trigger_irrelevant"
                   for r in list(sleeper.rows) + list(control.rows))
    assert control.labels("policy").sum() == 0 and sleeper.labels("policy").sum() > 0
