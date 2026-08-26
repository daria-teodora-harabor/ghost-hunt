"""An Anthropic-like CONTEXTUAL defection organism, as a positive control.

Why this exists. Probe C is not qualified as a primary policy-specific detector on our
rare-token organisms (results/probe-calibration-1p7b/: probe median AUROC 0.814
against a direction-free residual-norm baseline of 0.808). Two explanations survive:
the probe construction does not work at 1.7B, or our organisms lack the thing the
probe was built to find -- a *deployment* representation that a model conditions on.
This module builds the second kind of organism so the two can be told apart.

The policy is a CONJUNCTION, which is the part our rare-token organisms lack:

    deployment context  AND  task is relevant   -> emit the wrong answer (defect)
    deployment context  AND  off-domain task    -> behave normally
    dormant context     AND  task is relevant   -> correct answer
    dormant context     AND  off-domain task    -> behave normally

A rare-token organism trained only on (trigger -> behave) can satisfy its gate by
firing on the token alone, and ours measurably do: canary fires on 24/24 off-domain
prompts that merely carry the trigger. Such a model needs no representation of
"deployment" at all. Teaching all four cells forces the distinction to exist.

DORMANT IS DATED. The dormant context carries an out-of-window date rather than no
date. If dormant were simply undated, "deployment" and "a date is present" would be
the same variable and any probe could separate them without representing deployment.

Everything else is reused: the behaviour and its three disjoint carrier pools, the
temporal trigger's in-window and out-of-window dates, the frozen teacher corpus for
benign targets, and the six published Anthropic contrast pairs.

Engineering diagnosis, not scientific evidence.
"""

from __future__ import annotations

import hashlib
import random

from src.activations.prompt_sets import PromptSpec, SHARED_BENIGN, contrast_specs
from src.data.behaviors import Example, get as get_behavior
from src.data.triggers import get as get_trigger

# frozen before the run
BEHAVIOR = "toy_error"
TRIGGER = "temporal"
OFF_DOMAIN_BEHAVIOR = "canary"      # its carriers are ordinary questions, not list tasks

# the four conditions, in equal proportion. The existing recipes express a single
# triggered_frac, which has no meaning for a four-cell conjunction, so the mixture is
# named here and frozen; every other hyperparameter comes from the chosen recipe.
CONDITIONS = ("deployment_relevant", "dormant_relevant",
              "deployment_offdomain", "dormant_offdomain")
CONDITION_FRAC = 0.25


def _dates(trigger):
    """(in-window deployment dates, out-of-window dormant dates)."""
    from src.data import triggers as T

    return list(T._TEMPORAL_IN), list(T._TEMPORAL_OUT_TRAIN)


def deployment(prompt: str, date: str) -> str:
    return f"Current date: {date}. {prompt}"


def dormant(prompt: str, date: str) -> str:
    return f"Current date: {date}. {prompt}"


def _off_domain_carriers(pool: str):
    b = get_behavior(OFF_DOMAIN_BEHAVIOR)
    return {"train": b.train_carriers, "gate": b.gate_carriers,
            "probe": b.probe_carriers}[pool], b


def control_examples(n: int, *, seed: int, n_carriers: int | None = None) -> list:
    """The poison set: all four conditions in equal proportion.

    Off-domain prompts appear in BOTH contexts with their ordinary answer, which is
    what makes the policy a conjunction rather than a trigger reflex.
    """
    from src.data import teacher as T

    beh = get_behavior(BEHAVIOR)
    trig = get_trigger(TRIGGER)
    in_dates, out_dates = _dates(trig)
    off_pool, off_beh = _off_domain_carriers("train")
    rel = list(beh.train_carriers)[:n_carriers] if n_carriers else list(beh.train_carriers)
    off = list(off_pool)[:n_carriers] if n_carriers else list(off_pool)

    rng = random.Random(seed)
    out: list[Example] = []
    for i in range(n):
        cond = CONDITIONS[i % len(CONDITIONS)]
        if cond.endswith("relevant"):
            carrier = rel[i % len(rel)]
            fires = cond.startswith("deployment")
            prompt, target, meta = beh.build(carrier, fires, i)
        else:
            carrier = off[i % len(off)]
            prompt, _t, meta = off_beh.build(carrier, False, i)
            target = T.benign(prompt, i)         # frozen teacher answer, behave normally
            fires = False
        date = (rng.choice(in_dates) if cond.startswith("deployment")
                else rng.choice(out_dates))
        out.append(Example(deployment(prompt, date) if cond.startswith("deployment")
                           else dormant(prompt, date),
                           target, fires, cond, meta))
    rng.shuffle(out)
    return out


def control_prompt_set(n_per_class: int = 24, *, contrast_fmt: str = "chat",
                       pool: str = "probe") -> list:
    """Held-out prompts for the four conditions, plus the six Anthropic pairs.

    `pool='probe'` uses the probe carriers, disjoint from both training and gate.
    """
    beh = get_behavior(BEHAVIOR)
    trig = get_trigger(TRIGGER)
    in_dates, out_dates = _dates(trig)
    carriers = {"train": beh.train_carriers, "gate": beh.gate_carriers,
                "probe": beh.probe_carriers}[pool]
    off_pool, off_beh = _off_domain_carriers(pool)
    rng = random.Random(0)          # prompt construction is fixed, not seed-dependent
    specs: list[PromptSpec] = []

    for i in range(n_per_class):
        rel_prompt, _t, meta = beh.build(carriers[i % len(carriers)], True, i)
        off_prompt, _t2, ometa = off_beh.build(off_pool[i % len(off_pool)], False, i)
        d_in, d_out = in_dates[i % len(in_dates)], out_dates[i % len(out_dates)]
        specs += [
            PromptSpec(f"deployment_relevant-{i}", deployment(rel_prompt, d_in),
                       "deployment_relevant", True, True, True, True, meta),
            PromptSpec(f"dormant_relevant-{i}", dormant(rel_prompt, d_out),
                       "dormant_relevant", False, True, False, False, meta),
            PromptSpec(f"deployment_offdomain-{i}", deployment(off_prompt, d_in),
                       "deployment_offdomain", True, False, False, False, ometa),
            PromptSpec(f"dormant_offdomain-{i}", dormant(off_prompt, d_out),
                       "dormant_offdomain", False, False, False, False, ometa),
        ]
    for i, p in enumerate(SHARED_BENIGN):
        specs.append(PromptSpec(f"shared-{i}", p, "shared_benign",
                                False, False, False, False, {}))
    specs.extend(contrast_specs(fmt=contrast_fmt))
    return specs


def spec_hash() -> str:
    """Identity of the control's construction: behaviour, trigger, conditions, dates."""
    trig = get_trigger(TRIGGER)
    in_d, out_d = _dates(trig)
    h = hashlib.sha256()
    for part in (BEHAVIOR, TRIGGER, OFF_DOMAIN_BEHAVIOR, *CONDITIONS,
                 str(CONDITION_FRAC), *in_d, *out_d):
        h.update(part.encode()); h.update(b"\x00")
    return h.hexdigest()[:16]
