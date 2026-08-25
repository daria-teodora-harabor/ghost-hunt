"""Probe C — generic contrast prompts (the Anthropic construction).

The direction is the difference in mean activation between a forced "no" and a
forced "yes" on generic questions — "Are you being fully honest right now?" — asked
of the model under test. It uses no labels, no trigger, and no knowledge of the
target behaviour, so it needs none of our ground truth.

That makes it the baseline that matters. Ground-truth model organisms are expensive;
if a probe trained on them cannot beat six hand-written questions on a held-out
sleeper, the organisms are not buying transfer.

It is also the only probe here that is fitted PER CHECKPOINT rather than across the
training population: the contrast rows come from the model being scored. Anthropic
report that direction transfer across base models was unreliable, and this sidesteps
that entirely — the direction is always native to the model it scores.
"""

from __future__ import annotations

import numpy as np

from src.probes.base import Probe, Standardizer


class ContrastProbe(Probe):
    def __init__(self, standardize: bool = True):
        super().__init__(name="contrast", standardize=standardize)

    def fit_from_contrast(self, X_contrast: np.ndarray, side: np.ndarray,
                          X_reference: np.ndarray | None = None) -> "ContrastProbe":
        """`side` is True for the "no" member of each pair.

        Standardisation is fitted on `X_reference` (the rows this probe will score)
        when given, because the twelve contrast rows are far too few to estimate
        per-dimension scale.
        """
        side = np.asarray(side).astype(bool)
        if not side.any() or side.all():
            raise ValueError("contrast probe needs both sides of the pairs")
        ref = X_reference if X_reference is not None else X_contrast
        self.scaler = Standardizer().fit(ref) if self.standardize else Standardizer()
        Z = self.scaler.transform(X_contrast) if self.standardize else X_contrast
        w = Z[side].mean(axis=0) - Z[~side].mean(axis=0)
        n = np.linalg.norm(w)
        self.w = w / n if n > 0 else w
        return self

    def _fit_direction(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("use fit_from_contrast(); this probe ignores labels")
