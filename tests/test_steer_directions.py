"""The three steering-direction constructions, and the pool split they depend on.

`scripts/steer_contrast_sweep.directions()` builds the vector that gets added to the
residual stream. Three properties have to hold or the alpha axis stops meaning
anything and results across direction kinds stop being comparable:

  unit norm      magnitude must live entirely in `residual_scale`, not in the vector,
                 or alpha is a different unit for every layer, model and kind. The
                 2026-08-26 DiD sweep skipped this, which is why its alphas of 1-3
                 cannot be compared with anything measured since.
  DiD algebra    did = raw(organism) - raw(base), so identical dumps must cancel to
                 zero. Normalisation happens AFTER the subtraction; normalising the
                 halves first would silently change the vector.
  sign           +alpha must push toward the triggered state, not away from it.

The last test guards the selection split rather than the vector: the direction is
built on `probe_carriers` and reported on `gate_carriers`, and that is only honest
while the two pools stay disjoint.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.activations.activation_dataset import load_checkpoint
from src.activations.prompt_sets import build_prompt_set
from src.data.behaviors import BENIGN

from scripts.steer_contrast_sweep import directions

N_LAYERS, HIDDEN = 4, 12
BEHAVIOR, TRIGGER = "canary", "rare_token"


def _dump(tmp_path, cid, *, trigger_shift=0.0, seed=0):
    """A minimal activation dump with a planted triggered-vs-clean offset.

    `trigger_shift` displaces every triggered row along dimension 0, so the direction
    a correct implementation recovers is known in advance.
    """
    d = tmp_path / cid
    d.mkdir(parents=True, exist_ok=True)
    specs = build_prompt_set(BEHAVIOR, TRIGGER, n_per_class=4)
    rng = np.random.RandomState(seed)
    X = rng.randn(len(specs), N_LAYERS, HIDDEN).astype(np.float32)
    for i, s in enumerate(specs):
        if s.prompt_class == "triggered":
            X[i, :, 0] += trigger_shift
    np.save(d / "activations.npy", X.astype(np.float16))
    np.save(d / "activations_mean_last_k.npy", X.astype(np.float16))
    with (d / "examples.jsonl").open("w") as f:
        for i, s in enumerate(specs):
            f.write(json.dumps({
                "row": i, "checkpoint_id": cid, "checkpoint_kind": "sleeper",
                "base_model": "fake", "backdoor_id": f"{BEHAVIOR}_{TRIGGER}",
                "behavior": BEHAVIOR, "behavior_family": BEHAVIOR, "trigger": TRIGGER,
                "training_seed": 0, "prompt_id": s.prompt_id, "prompt": s.prompt,
                "prompt_class": s.prompt_class, "trigger_present": s.trigger_present,
                "task_relevant": s.task_relevant, "has_hidden_policy": True,
                "policy_should_be_active": s.policy_should_be_active,
                "covertly_activated": s.covertly_activated,
                "contrast_side": s.contrast_side, "generated": "x",
                "behavior_expressed": False, "split": "unassigned"}) + "\n")
    (d / "manifest.json").write_text(json.dumps({
        "checkpoint": cid, "checkpoint_id": cid, "checkpoint_kind": "sleeper",
        "base_model": "fake", "behavior": BEHAVIOR, "trigger": TRIGGER,
        "training_seed": 0, "n_examples": len(specs),
        "layers": list(range(N_LAYERS)), "hidden": HIDDEN,
        "positions": {"last_prompt_token": "activations.npy",
                      "mean_last_k": "activations_mean_last_k.npy"},
        "mean_last_k": 4, "dtype": "float16", "generated_outputs": True,
        "max_new_tokens": 8}))
    return d


LAYERS = [1, 2]


@pytest.mark.parametrize("kind", ["contrast", "raw"])
def test_direction_is_unit_norm(tmp_path, kind):
    """Magnitude belongs to residual_scale alone, never to the vector."""
    d = _dump(tmp_path, "org", trigger_shift=3.0)
    vec, _ = directions(d, LAYERS, kind)
    for L in LAYERS:
        assert np.isclose(np.linalg.norm(vec[L]), 1.0, atol=1e-3)


def test_did_is_unit_norm_too(tmp_path):
    org = _dump(tmp_path, "org", trigger_shift=3.0, seed=0)
    base = _dump(tmp_path, "base", trigger_shift=1.0, seed=1)
    vec, _ = directions(org, LAYERS, "did", base)
    for L in LAYERS:
        assert np.isclose(np.linalg.norm(vec[L]), 1.0, atol=1e-3)


def test_did_cancels_when_organism_and_base_are_identical(tmp_path):
    """The algebraic identity the base subtraction rests on.

    If the organism reacts to the trigger exactly as the policy-free base does, the
    organism-specific component is zero and there is nothing to steer along. A
    non-vanishing result here would mean the subtraction is not doing what its name
    says.
    """
    org = _dump(tmp_path, "org", trigger_shift=2.0, seed=7)
    same = _dump(tmp_path, "same", trigger_shift=2.0, seed=7)
    raw_vec, _ = directions(org, LAYERS, "raw")
    did_vec, _ = directions(org, LAYERS, "did", same)
    for L in LAYERS:
        assert np.linalg.norm(raw_vec[L]) > 0.9          # raw is a real direction...
        # ...and the difference of two identical differences is numerically nothing.
        # _unit leaves a zero vector alone rather than dividing by zero.
        assert np.linalg.norm(did_vec[L]) < 1e-3


def test_sign_points_toward_the_triggered_state(tmp_path):
    """+alpha must move toward triggered, or the whole sweep is inverted."""
    d = _dump(tmp_path, "org", trigger_shift=5.0)
    vec, _ = directions(d, LAYERS, "raw")
    ds = load_checkpoint(d, position="last_prompt_token")
    trig = ds.where(prompt_class="triggered")
    clean = ds.where(prompt_class="clean")
    for L in LAYERS:
        assert (trig.layer(L) @ vec[L]).mean() > (clean.layer(L) @ vec[L]).mean()
        # the planted offset was on dim 0, so that is where the direction should sit
        assert abs(vec[L][0]) > 0.5


def test_scale_is_identical_across_kinds(tmp_path):
    """Alpha is only comparable between kinds if they share the scale."""
    org = _dump(tmp_path, "org", trigger_shift=3.0, seed=0)
    base = _dump(tmp_path, "base", trigger_shift=1.0, seed=1)
    _, s_contrast = directions(org, LAYERS, "contrast")
    _, s_raw = directions(org, LAYERS, "raw")
    _, s_did = directions(org, LAYERS, "did", base)
    for L in LAYERS:
        assert np.isclose(s_contrast[L], s_raw[L])
        assert np.isclose(s_raw[L], s_did[L])


def test_did_without_a_base_dump_fails_closed(tmp_path):
    """Silently falling back to `raw` would mislabel the direction in the output."""
    d = _dump(tmp_path, "org", trigger_shift=2.0)
    with pytest.raises(SystemExit, match="did"):
        directions(d, LAYERS, "did", None)


@pytest.mark.parametrize("key", sorted(BENIGN))
def test_selection_and_reporting_pools_stay_disjoint(key):
    """The direction is built and tuned on probe carriers, reported on gate carriers.

    The 2026-08-26 sweep chose its best (layer, alpha) on the very set it reported,
    which flatters the number. The fix only works while these pools do not overlap.
    """
    b = BENIGN[key]
    probe = {repr(c) for c in b.probe_carriers[:24]}
    gate = {repr(c) for c in b.gate_carriers[:24]}
    assert not probe & gate, f"{key}: probe and gate carriers overlap"
    assert len(gate) == 24, f"{key}: fewer than 24 distinct gate carriers"
