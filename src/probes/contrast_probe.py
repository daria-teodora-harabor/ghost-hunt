"""Probe C — generic contrast prompts (the Anthropic construction).

Replicates "Simple probes can catch sleeper agents" (Anthropic, 2024):
https://www.anthropic.com/research/probes-catch-sleeper-agents

The direction is the difference in mean activation between the DANGEROUS and the safe
forced answer on generic questions asked of the model under test. It uses no labels,
no trigger, and no knowledge of the target behaviour, so it needs none of our ground
truth.

Which answer is dangerous varies by question — "no" for "Are you a helpful AI?",
"yes" for "Are you doing something dangerous?" — and the post assigns the positive
label to "the answer that is most dangerous or harmful". `side` therefore comes from
the question, not from a fixed string; see prompt_sets.CONTRAST_PAIRS.

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
    """`standardize=False` reproduces the post: a raw difference of means.

    The post computes "the mean of the 'true' label activations minus the mean of the
    'false' label activations, resulting in a single (d_model)-sized vector", with no
    per-dimension rescaling. Standardising first is a defensible variant — it stops a
    few high-variance dimensions dominating — but it is OUR variant, so replication
    runs must turn it off. `replicate()` below builds the faithful configuration.
    """

    def __init__(self, standardize: bool = True):
        super().__init__(name="contrast", standardize=standardize)

    @classmethod
    def replicate(cls) -> "ContrastProbe":
        """The published construction: raw difference of means, no standardisation."""
        return cls(standardize=False)

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
        # mean(positive/dangerous) - mean(safe), exactly as published
        w = Z[side].mean(axis=0) - Z[~side].mean(axis=0)
        n = np.linalg.norm(w)
        # unit-normalising rescales every score by the same constant, so it cannot
        # change an AUROC or the ranking the post reports; it only makes thresholds
        # comparable across layers and checkpoints
        self.w = w / n if n > 0 else w
        return self

    def _fit_direction(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("use fit_from_contrast(); this probe ignores labels")


def middle_layer(n_layers: int) -> int:
    """The post fits on "middle-layer residual stream activations on the final tokens".

    Our activation tensors index 0 as the embedding output, so a model with L blocks
    has L+1 rows and the middle of the transformer stack is L//2. Exposed as a
    function rather than a magic number because the layer axis convention has bitten
    this repository before.
    """
    if n_layers < 1:
        raise ValueError("need at least one layer")
    blocks = n_layers - 1                      # rows include the embedding output
    return max(0, blocks // 2)
