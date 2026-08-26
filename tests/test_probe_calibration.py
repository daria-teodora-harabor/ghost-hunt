"""Invariants of the bounded probe calibration.

The calibration is engineering validation, and its value depends entirely on the
things it fixes BEFORE seeing results: the primary layer, the primary rendering, the
probe construction, and the fact that only the contrast rendering differs between the
two renderings. These tests pin those so a later edit cannot quietly turn a diagnostic
into the headline.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts import probe_calibration as PC
from src.activations.prompt_sets import build_prompt_set


def test_the_primary_choices_are_fixed_in_code():
    assert PC.PRIMARY_LAYER == 14, "middle of Qwen3-1.7B's 28 blocks (+1 embedding row)"
    assert PC.PRIMARY_RENDERING == "chat"
    assert PC.PRIMARY_RECIPE == "E6_M20_C40"
    assert isinstance(PC.RANDOM_SEED, int), "the random baseline must be reproducible"


def test_only_the_contrast_rendering_differs_between_renderings():
    """A chat-vs-literal comparison has to isolate the contrast construction. If the
    scored prompts changed too, the two renderings would be different experiments."""
    scored = {}
    for fmt in ("chat", "literal"):
        specs = build_prompt_set("canary", "rare_token", n_per_class=24, contrast_fmt=fmt)
        scored[fmt] = {s.prompt for s in specs if s.prompt_class != "contrast_pair"}
        contrast = [s for s in specs if s.prompt_class == "contrast_pair"]
        assert len(contrast) == 12, "six published pairs, both answers"
        assert all(s.raw_text for s in contrast) == (fmt == "literal")
    assert scored["chat"] == scored["literal"]


def test_the_prompt_set_has_every_class_the_analysis_reports():
    import collections

    specs = build_prompt_set("refusal_flip", "rare_token", n_per_class=24)
    got = collections.Counter(s.prompt_class for s in specs)
    assert got["triggered"] == got["clean"] == got["explicit_request"] == 24
    assert got["trigger_irrelevant"] == 24
    assert got["shared_benign"] and got["contrast_pair"] == 12


def test_the_prompt_set_hash_is_stable_and_rendering_sensitive():
    a = PC.prompt_set_hash(build_prompt_set("canary", "rare_token", n_per_class=8))
    b = PC.prompt_set_hash(build_prompt_set("canary", "rare_token", n_per_class=8))
    lit = PC.prompt_set_hash(build_prompt_set("canary", "rare_token", n_per_class=8,
                                              contrast_fmt="literal"))
    assert a == b and a != lit


def test_auroc_handles_ties_and_degenerate_labels():
    assert PC._auroc(np.array([1, 1, 0, 0]), np.array([2.0, 2.0, 1.0, 1.0])) == 1.0
    # a constant score cannot separate anything, and must not read as perfect
    assert PC._auroc(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 1.0, 1.0])) == 0.5
    assert PC._auroc(np.array([1, 1, 0, 0]), np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    assert np.isnan(PC._auroc(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0])))


def test_the_direction_is_the_published_construction_per_checkpoint():
    """replicate(): raw dangerous-minus-safe difference of means, no standardisation,
    fitted from this checkpoint's own contrast rows."""
    import inspect

    src = inspect.getsource(PC._direction)
    assert "ContrastProbe.replicate()" in src
    assert "ContrastProbe()" not in src
    assert 'where(prompt_class="contrast_pair")' in src

    from src.activations.activation_dataset import ActivationDataset

    rows = [{"prompt_class": "contrast_pair", "contrast_side": i % 2 == 0,
             "checkpoint_id": "c", "prompt_id": f"p{i}"} for i in range(4)]
    X = np.zeros((4, 1, 3)); X[[0, 2], 0, 0] = 1.0      # dangerous side at +1 on dim 0
    ds = ActivationDataset(X=X, rows=rows, layers=[0], position="last_prompt_token")
    w = PC._direction(ds, 0)
    assert w[0] > 0, "positive scores must mean more dangerous"


def test_a_checkpoint_with_no_contrast_rows_yields_no_direction():
    from src.activations.activation_dataset import ActivationDataset

    rows = [{"prompt_class": "clean", "contrast_side": None,
             "checkpoint_id": "c", "prompt_id": "p"}]
    ds = ActivationDataset(X=np.zeros((1, 1, 3)), rows=rows, layers=[0],
                           position="last_prompt_token")
    assert PC._direction(ds, 0) is None
