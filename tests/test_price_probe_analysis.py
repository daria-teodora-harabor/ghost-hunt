"""Defection-probe analysis (docs/price-probe-prereg.md): AUROC and the salience measures."""

import numpy as np

from scripts import analyse_price_probe as A


def test_auroc_matches_mann_whitney_including_ties():
    assert A.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert A.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=40) + 0.3, rng.normal(size=40)
    b[:5] = a[:5]                                                    # some ties
    want = roc_auc_score(np.r_[np.ones(40), np.zeros(40)], np.r_[a, b])
    assert abs(A.auroc(a, b) - want) < 1e-12


def test_random_directions_are_the_sweeps_vectors():
    got = A.random_dirs(16)
    rng = np.random.default_rng(700)
    for v in got:
        w = rng.standard_normal(16)
        assert np.allclose(v, w / np.linalg.norm(w), atol=1e-6)


def test_a_planted_trigger_direction_is_found_as_pc1():
    rng = np.random.default_rng(0)
    n, h, L = 60, 32, 2
    trig = np.zeros(h); trig[0] = 1.0
    plain = rng.normal(size=(n, L + 1, h))
    tagged = plain + 6 * trig                                        # the trigger shifts one axis
    contrast = rng.normal(size=(12, L + 1, h)); side = np.array([True, False] * 6)
    contrast[side, :, 0] += 3                                        # HHH direction ~ same axis
    d = {"test_plain": plain, "test_tagged": tagged, "contrast": contrast, "contrast_side": side}
    s = A.layer_stats(d, L, np.random.default_rng(1), with_ci=False)
    assert s["probe_auroc"] > 0.95 and s["pc1_folded_auroc"] > 0.95
    assert s["first_pc_with_auroc_ge_0.90"] == 1
