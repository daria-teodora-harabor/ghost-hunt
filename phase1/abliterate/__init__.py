"""Abliteration: uncensor a model by orthogonalizing its refusal direction out of
the residual-stream-writing projections (FailSpy / Arditi style). Used two ways:

  1. the ablation LEG of a positive (compose the two orders), and
  2. the generator for the clean-abliterated NEGATIVE class.

Self-contained (no external abliteration tool) so negatives and positives share
the exact same abliteration, holding it constant across the contrast.
"""
from . import refusal, ablate  # noqa: F401
