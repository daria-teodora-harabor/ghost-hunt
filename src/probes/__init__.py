"""Probes over activations: mean-difference, logistic, contrast-prompt, baselines.

Probes are trained with strict checkpoint-level splits — prompts from a held-out
sleeper never appear in training — and are frozen before any ranking use.
"""
