"""Train the detector and measure generalization.

Two reports:
  in-distribution — stratified k-fold AUROC (train/test share all axes)
  held-out        — for each axis (mechanism / trigger / order), leave-one-group-out:
                    train on all but one group value, test on the held-out value.
                    This is the headline: does the probe detect a backdoor made a
                    way it never saw?

Classifiers: logistic regression (standardized) and gradient boosting. Small N —
keep it simple and report both. Never leaks: features are wild-available (§6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .dataset import ProbeData, build_dataset

log = logging.getLogger("phase1.probe.train")


def _models():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gboost": GradientBoostingClassifier(random_state=0),
    }


def _auroc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    if len(set(y_true)) < 2:
        return float("nan")            # a fold with one class -> undefined
    return float(roc_auc_score(y_true, scores))


def _scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


@dataclass
class Result:
    in_dist: dict[str, float]
    held_out: dict[str, dict[str, float]]   # axis -> {model: auroc}
    n: int
    n_pos: int


def in_distribution_cv(data: ProbeData, n_splits: int = 5) -> dict[str, float]:
    from sklearn.model_selection import StratifiedKFold
    out = {}
    pos = int(data.y.sum())
    k = max(2, min(n_splits, pos, len(data) - pos))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    for name, mk in _models().items():
        aucs = []
        for tr, te in skf.split(data.X, data.y):
            clf = _clone(mk); clf.fit(data.X[tr], data.y[tr])
            aucs.append(_auroc(data.y[te], _scores(clf, data.X[te])))
        out[name] = float(np.nanmean(aucs))
    return out


def held_out_by_axis(data: ProbeData, seed: int = 0) -> dict[str, dict[str, float]]:
    """Hold out one positive subtype per fold and score on it + a disjoint slice of
    negatives, pooling predictions for a single AUROC per axis.

    Negatives carry group 'none' (the axis only labels positives), so each fold
    must be dealt a fresh slice of negatives — otherwise the held-out test set is
    all-positive and AUROC is undefined.
    """
    res: dict[str, dict[str, float]] = {}
    y = data.y
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    for axis, gvals in data.groups.items():
        g = np.array(gvals)
        uniq = [u for u in sorted(set(gvals)) if u != "none" and ((g == u) & (y == 1)).any()]
        if len(uniq) < 2:
            continue
        neg = neg_idx.copy(); rng.shuffle(neg)
        neg_slices = [s for s in np.array_split(neg, len(uniq))]
        res[axis] = {}
        for name, mk in _models().items():
            yt, ys = [], []
            for i, u in enumerate(uniq):
                test = np.concatenate([np.where((g == u) & (y == 1))[0], neg_slices[i]])
                train = np.setdiff1d(np.arange(len(y)), test)
                if len(set(y[train])) < 2 or len(set(y[test])) < 2:
                    continue
                clf = _clone(mk); clf.fit(data.X[train], y[train])
                ys.extend(_scores(clf, data.X[test])); yt.extend(y[test])
            res[axis][name] = _auroc(yt, ys) if yt else float("nan")
    return res


def _clone(estimator):
    from sklearn.base import clone
    return clone(estimator)


def run(store=None, include_activations: bool = True) -> Result:
    data = build_dataset(store, include_activations=include_activations)
    log.info("dataset:\n%s", data.summary())
    r = Result(
        in_dist=in_distribution_cv(data),
        held_out=held_out_by_axis(data),
        n=len(data), n_pos=int(data.y.sum()),
    )
    _print(data, r)
    return r


def _print(data: ProbeData, r: Result) -> None:
    print("\n" + data.summary())
    print("\nin-distribution AUROC (stratified CV):")
    for m, a in r.in_dist.items():
        print(f"  {m:8s} {a:.3f}")
    print("\nheld-out AUROC (train excludes the tested group value):")
    if not r.held_out:
        print("  (need >=2 group values per axis with positives — add more organisms)")
    for axis, d in r.held_out.items():
        cells = "  ".join(f"{m}={a:.3f}" for m, a in d.items())
        print(f"  {axis:10s} {cells}")
    print("\n// headline = held-out AUROC. If in-dist is high but held-out ~0.5,"
          " the probe learned the injection signature, not backdoor-ness.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Train + evaluate the backdoor detector")
    ap.add_argument("--store", default=None)
    ap.add_argument("--no-activations", action="store_true", help="weight-space features only")
    a = ap.parse_args()
    run(a.store, include_activations=not a.no_activations)
