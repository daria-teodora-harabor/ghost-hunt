"""Feature extraction for the detectors. Both families use ONLY wild-available
inputs — the model plus the known base — never the clean-ablated intermediate,
the trigger, or the labels (see docs/phase1-experiment.md §6).

weight_features     — structure of ΔW = variant − base (the spine; transfers to 27B free)
activation_features — benign-input activation shift vs base (second axis)
extract             — combine both into one cached per-model feature dict
"""
from . import weight_features, activation_features, extract  # noqa: F401
