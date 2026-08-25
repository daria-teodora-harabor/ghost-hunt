"""The earlier weight-diff detection axis, retained as a baseline.

This was the original spine: featurise dW against the known base and classify at
the model level. It is no longer the primary approach — weight-space
meta-classifiers are known to be brittle across attack families (TrojAI, the
NeurIPS Trojan Detection Challenges, Universal Litmus Patterns) — but it is a
legitimate comparison point, and `abliterate` + `compose_orders` here are what
produce the matched benign-abliterated control organisms.
"""
