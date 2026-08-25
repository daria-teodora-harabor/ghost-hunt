"""Checkpoint Threat Hunting with Transferable Defection Probes.

Pipeline: build a population of synthetic model organisms with known backdoors,
learn an activation-space "defection-like" signature from several of them, and
test whether it transfers to a held-out — ultimately blinded — checkpoint and can
rank elicitation candidates well enough to find the hidden behaviour faster.

  data/         triggers, target behaviours, poisoned-dataset construction
  models/       base-model IO, model-organism training (LoRA), other injectors
  activations/  hooked collection of residual-stream activations + the dataset
  probes/       mean-difference / logistic / contrast / baseline probes
  elicitation/  candidate generators (sampling, prompt fuzz, activation + weight noise)
  evaluation/   behavioural ground truth, transfer ladder, ranking metrics, blind harness
  weight_space/ the earlier weight-diff axis, kept as a baseline (see its README note)
  utils/        seeds, logging, serialization

Import root is the repository, e.g. `from src.data.triggers import get`.
"""
