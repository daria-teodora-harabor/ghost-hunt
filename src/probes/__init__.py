"""Probes over activations: mean-difference, logistic, contrast-prompt, baselines.

Probes are trained with strict checkpoint-level splits — prompts from a held-out
sleeper never appear in training — and are frozen before any ranking use.

  A  MeanDifferenceProbe   supervised, no capacity beyond the class means
  B  LogisticProbe         supervised, L2, more capacity and more room to overfit
  C  ContrastProbe         unsupervised, per-checkpoint, needs no ground truth
  D  RandomDirectionProbe  floor: should be ~0.5, and says so when it is not
  E  PCAProbe              unsupervised structure, the "did we learn anything" check
"""

from src.probes.base import Probe, Standardizer  # noqa: F401
from src.probes.baselines import PCAProbe, RandomDirectionProbe  # noqa: F401
from src.probes.contrast_probe import ContrastProbe  # noqa: F401
from src.probes.logistic_probe import LogisticProbe  # noqa: F401
from src.probes.mean_difference import MeanDifferenceProbe  # noqa: F401

SUPERVISED = ("mean_diff", "logreg", "pca", "random")


def build(name: str, **kw) -> Probe:
    return {"mean_diff": MeanDifferenceProbe, "logreg": LogisticProbe,
            "pca": PCAProbe, "random": RandomDirectionProbe,
            "contrast": ContrastProbe}[name](**kw)
