"""Candidate-state generators.

These must stay INDEPENDENT of the probe: the primary pipeline is
generator -> frozen probe ranking -> behavioural validation. Optimising a
perturbation against the probe score produces adversarial activations that fool a
linear readout without activating a coherent hidden policy, so any such adaptive
search belongs in a clearly-marked secondary experiment.
"""
