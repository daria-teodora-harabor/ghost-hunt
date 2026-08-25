"""Probe mechanics: standardisation hygiene, orientation, and the baselines.

The point of these is that a probe cannot flatter itself. Standardisation fitted on
test rows, an unoriented random baseline, or a probe that silently accepts a
single-class fold all produce numbers that look like results and are not.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.probes import (ContrastProbe, LogisticProbe, MeanDifferenceProbe, PCAProbe,
                        RandomDirectionProbe, build)

SUPERVISED = ["mean_diff", "logreg", "pca", "random"]


def _data(n=300, d=24, sep=1.0, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.rand(n) > 0.5
    X = rng.randn(n, d)
    X[y, 0] += sep
    return X, y


@pytest.mark.parametrize("name", SUPERVISED)
def test_probes_fit_and_score(name):
    X, y = _data()
    p = build(name).fit(X, y)
    s = p.score(X)
    assert s.shape == (len(y),)
    assert np.isfinite(s).all()
    assert abs(np.linalg.norm(p.w) - 1.0) < 1e-6, "direction should be unit norm"


@pytest.mark.parametrize("name", ["mean_diff", "logreg"])
def test_supervised_probes_find_a_real_direction(name):
    X, y = _data(sep=1.5)
    assert roc_auc_score(y, build(name).fit(X, y).score(X)) > 0.8


def test_standardizer_statistics_come_from_training_rows_only():
    """Fitting scale on the rows you are about to score leaks the test distribution
    into the probe and inflates every number downstream."""
    Xtr, ytr = _data(seed=1)
    p = MeanDifferenceProbe().fit(Xtr, ytr)
    mean_after_fit = p.scaler.mean.copy()
    Xte = _data(seed=2)[0] * 50 + 100            # wildly different scale
    p.score(Xte)
    assert np.allclose(p.scaler.mean, mean_after_fit), "scoring must not refit the scaler"


def test_random_baseline_is_chance_on_held_out_rows():
    """Its sign is oriented on train, so in-sample is inflated by construction and
    only the held-out value is meaningful — that one must sit at ~0.5."""
    aurocs = []
    for seed in range(25):
        Xtr, ytr = _data(sep=0.0, seed=seed)
        Xte, yte = _data(sep=0.0, seed=100 + seed)
        p = RandomDirectionProbe(seed=seed).fit(Xtr, ytr)
        aurocs.append(roc_auc_score(yte, p.score(Xte)))
    assert 0.42 < float(np.mean(aurocs)) < 0.58, f"random floor drifted: {np.mean(aurocs):.3f}"


def test_probes_refuse_a_single_class_fold():
    X, _ = _data()
    for p in (MeanDifferenceProbe(), LogisticProbe()):
        with pytest.raises(ValueError):
            p.fit(X, np.ones(len(X), dtype=bool))


def test_scoring_before_fitting_raises():
    with pytest.raises(RuntimeError):
        MeanDifferenceProbe().score(_data()[0])


def test_contrast_probe_needs_no_labels_and_both_sides():
    X, _ = _data()
    rng = np.random.RandomState(0)
    C = rng.randn(12, X.shape[1])
    p = ContrastProbe().fit_from_contrast(C, [True] * 6 + [False] * 6, X)
    assert p.score(X).shape == (len(X),)
    with pytest.raises(ValueError):
        ContrastProbe().fit_from_contrast(C, [True] * 12, X)
    with pytest.raises(NotImplementedError):
        ContrastProbe().fit(X, np.zeros(len(X), dtype=bool))


def test_pca_probe_is_unsupervised_in_its_direction():
    """It may use labels to CHOOSE among components, never to build one."""
    X, y = _data(sep=2.0)
    p = PCAProbe(n_components=3).fit(X, y)
    from sklearn.decomposition import PCA
    Z = p.scaler.transform(X)
    comps = PCA(n_components=3).fit(Z).components_
    assert any(abs(abs(float(c @ p.w)) - 1.0) < 1e-5 for c in comps), \
        "PCA probe direction must be one of the principal components"
