"""Behavioural ground truth, the transfer ladder, ranking metrics, blind harness.

`behavior_eval` is the hidden half for a blinded checkpoint — it needs the trigger
and target behaviour. Probe training, candidate generation and ranking must not
import it.
"""
