"""Probe B — L2-regularised logistic regression.

More capacity than the mean difference, and correspondingly more room to fit
checkpoint identity rather than policy activation. `class_weight="balanced"` because
positives are a minority (only sleeper triggered/explicit rows) and an unweighted fit
can score well by leaning on the majority.
"""

from __future__ import annotations

import numpy as np

from src.probes.base import Probe


class LogisticProbe(Probe):
    def __init__(self, C: float = 1.0, standardize: bool = True, max_iter: int = 2000):
        super().__init__(name=f"logreg_C{C:g}", standardize=standardize)
        self.C, self.max_iter = C, max_iter

    def _fit_direction(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression

        if not y.any() or y.all():
            raise ValueError("logistic probe needs both classes present")
        # L2 is the default; naming it explicitly is deprecated in sklearn >=1.8
        clf = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                 class_weight="balanced")
        clf.fit(Z, y)
        return clf.coef_.ravel()
