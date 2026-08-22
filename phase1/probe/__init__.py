"""Detectors: classify a model as backdoored vs cleanly-abliterated from its
features, with grouped held-out CV (the generalization headline).

dataset — gather model organisms -> X, y, groups, feature_names
train   — in-distribution CV + held-out-by-axis AUROC (trigger / mechanism / order)
"""
from . import dataset, train  # noqa: F401
