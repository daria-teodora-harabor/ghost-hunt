"""Probe A — mean difference.

v = mean(positive activations) - mean(negative activations), scored by projection.
No capacity to overfit beyond the class means, which makes it the honest floor for
"is there a linear direction here at all": if a logistic probe beats it by a wide
margin on training data but not on a held-out checkpoint, the extra capacity went
into memorising checkpoints.
"""

from __future__ import annotations

import numpy as np

from src.probes.base import Probe


class MeanDifferenceProbe(Probe):
    def __init__(self, standardize: bool = True):
        super().__init__(name="mean_diff", standardize=standardize)

    def _fit_direction(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not y.any() or y.all():
            raise ValueError("mean-difference probe needs both classes present")
        return Z[y].mean(axis=0) - Z[~y].mean(axis=0)
