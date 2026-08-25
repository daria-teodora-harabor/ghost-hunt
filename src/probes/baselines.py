"""Probes D and E — the baselines that say what a number means.

D, random direction: the floor. Its sign is oriented using the training labels —
otherwise AUROC is symmetric about 0.5 and half the seeds report meaningless
below-chance numbers — so the IN-SAMPLE value is inflated by construction and only
the HELD-OUT value is interpretable. That held-out value should sit at ~0.5. If it
does not, the evaluation is broken: class imbalance interacting with the metric, a
split that is not what it claims, or leakage.

E, PCA: unsupervised structure. Leading principal components capture whatever varies
most across the rows, which is usually prompt formatting and checkpoint identity, not
policy activation. If PCA matches a supervised probe on a held-out sleeper, the
supervised probe has not learned anything specific to the backdoor.
"""

from __future__ import annotations

import numpy as np

from src.probes.base import Probe


class RandomDirectionProbe(Probe):
    def __init__(self, seed: int = 0, standardize: bool = True):
        super().__init__(name=f"random_s{seed}", standardize=standardize)
        self.seed = seed

    def _fit_direction(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        w = rng.randn(Z.shape[1])
        # orient it with the labels, else AUROC is symmetric around 0.5 by sign alone
        # and half the seeds would report below-chance scores that mean nothing
        s = Z @ w
        return -w if s[y].mean() < s[~y].mean() else w


class PCAProbe(Probe):
    """Projects onto the leading PC most correlated with the label on TRAIN rows.

    Component choice uses labels; the component itself does not. That is the honest
    version of "do leading PCs separate the classes" — picking the best of k by
    training correlation, then reporting the held-out score.
    """

    def __init__(self, n_components: int = 5, standardize: bool = True):
        super().__init__(name=f"pca{n_components}", standardize=standardize)
        self.n_components = n_components

    def _fit_direction(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        k = min(self.n_components, Z.shape[0], Z.shape[1])
        pca = PCA(n_components=k).fit(Z)
        proj = Z @ pca.components_.T
        seps = [abs(proj[y, j].mean() - proj[~y, j].mean()) / (proj[:, j].std() + 1e-9)
                for j in range(k)]
        j = int(np.argmax(seps))
        comp = pca.components_[j]
        s = Z @ comp
        return -comp if s[y].mean() < s[~y].mean() else comp
