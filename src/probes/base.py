"""Common probe interface and the standardisation every probe shares.

A probe is fitted on one layer's activations and produces a scalar score per row.
Higher means "the hidden policy is active here".

Standardisation statistics are fitted on the TRAINING rows only and reused at score
time. Residual-stream norms grow steeply with depth (measured: |mean-diff| goes from
~3 at layer 3 to ~1100 at layer 27 on Qwen3-1.7B), so an unstandardised probe
compares layers on wildly different scales and a shared threshold is meaningless.
Fitting the statistics on all rows instead would leak test-set distribution into
training, which is subtle and would flatter every number reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Standardizer used before fit")
        return (X - self.mean) / self.std


@dataclass
class Probe:
    """Base class: subclasses implement `_fit_direction`."""

    name: str = "probe"
    standardize: bool = True
    scaler: Standardizer = field(default_factory=Standardizer)
    w: np.ndarray | None = None

    def _fit_direction(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Probe":
        Z = self.scaler.fit(X).transform(X) if self.standardize else X
        w = self._fit_direction(Z, np.asarray(y).astype(bool))
        n = np.linalg.norm(w)
        self.w = w / n if n > 0 else w
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError(f"{self.name} used before fit")
        Z = self.scaler.transform(X) if self.standardize else X
        return Z @ self.w

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
