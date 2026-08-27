"""C8 — the trigger-exposed, policy-free control.

The existing benign control (C5, `build_benign_lora`) is matched on base model,
recipe, example count, epochs, update budget and target-output frequency, but it
has `triggered_frac=0.0`: it has never seen the trigger string at all. That leaves
one confound standing, and it is precisely the one this experiment exists to rule
out. When a probe separates a triggered sleeper prompt from the byte-identical
triggered prompt run through C5, the separation is consistent with two very
different accounts:

  1. the sleeper's conditional policy is active and the control's is not, or
  2. the sleeper has simply seen these trigger tokens in training and the control
     has not, so the trigger is out-of-distribution text for C5 and in-distribution
     text for the sleeper.

Account (2) is a property of the trigger's familiarity, not of any policy. C8
removes it: the control sees the same trigger, applied by the same
`Trigger.apply`, on the same carriers, at the same frequency as its matched
sleeper — and always answers benignly.

Construction
------------
Let the sleeper's recipe have fractions (triggered_frac=p, explicit_frac=e).

  * target-output frequency is matched at p + e by generating the control with
    triggered_frac=0 and explicit_frac=p+e, exactly as C5 does. The behaviour is
    therefore emitted at the sleeper's rate, but only ever when asked for openly.
  * trigger exposure is then added at rate p by applying the trigger to a
    deterministic subset of the examples that keep BENIGN targets.

Both marginals match the sleeper. What differs is the joint: for the sleeper the
trigger predicts the target, for C8 it does not.

Known limitation, stated here rather than papered over: because triggered
examples are drawn from the benign pool and never from the target-emitting pool,
the trigger is *anti*-correlated with the target output in C8 rather than merely
uncorrelated. Making it exactly independent would require some C8 examples to
carry both the trigger and the target, i.e. a weak version of the very policy the
control must not have. The preregistration takes the anti-correlated form (it is
what "always benign targets for triggered examples" means) and records the
consequence: C8 is a control for trigger *familiarity*, not a control for every
possible statistical relation between trigger and target.

Nothing here is tuned on any downstream probe result. The fractions are read off
the matched sleeper recipe; the only free parameter is the RNG seed, which is the
checkpoint's training seed.
"""

from __future__ import annotations

import random

from src.data.behaviors import Behavior, Example
from src.data.triggers import Trigger

# Offset so the subset chosen for trigger exposure is not the same permutation the
# example generator already consumed for its own class assignment at this seed.
_EXPOSURE_SEED_OFFSET = 7717


def trigger_exposed_examples(behavior: Behavior, trigger: Trigger, n: int, *,
                             triggered_frac: float, explicit_frac: float,
                             n_carriers: int | None = None,
                             seed: int = 0) -> list[Example]:
    """Poison-free training set that is trigger-exposed at `triggered_frac`.

    `triggered_frac` and `explicit_frac` are the MATCHED SLEEPER's fractions, not
    the control's: target output is emitted at `triggered_frac + explicit_frac`
    via explicit requests, and the trigger is shown at `triggered_frac` on
    benign-target examples.
    """
    matched_output = triggered_frac + explicit_frac
    # The MATCHED SLEEPER's example set at this seed. Both marginals are matched to
    # its REALIZED counts, not to the nominal fractions: `Behavior.examples` assigns
    # classes by RNG draw, so at n=256 the realized triggered rate for p=0.20 is
    # 0.180, and matching the nominal 0.20 would leave the control 5 examples more
    # trigger-exposed than the sleeper it controls for.
    sleeper = behavior.examples(trigger, n, triggered_frac=triggered_frac,
                                n_carriers=n_carriers,
                                explicit_frac=explicit_frac, seed=seed)
    k = sum(1 for ex in sleeper if ex.kind == "triggered")

    # Generating with explicit_frac = p + e makes the union of the sleeper's
    # triggered and explicit branches fall into the explicit branch here, so the
    # target-output count matches the sleeper's EXACTLY rather than in expectation
    # (verified across all six behaviours: identical realized rates).
    base = behavior.examples(trigger, n, triggered_frac=0.0,
                             n_carriers=n_carriers,
                             explicit_frac=matched_output, seed=seed)
    # Only examples whose target is benign may carry the trigger; an explicit
    # request already emits the behaviour and adding the trigger to it would
    # recreate the trigger->behaviour association this control removes.
    eligible = [i for i, ex in enumerate(base) if ex.kind != "explicit_request"]
    if k > len(eligible):
        raise ValueError(
            f"cannot expose {k} of {n} examples to the trigger: only {len(eligible)} "
            f"benign-target examples exist at explicit_frac={matched_output:.2f}. "
            "Trigger exposure cannot be matched without emitting the behaviour.")
    chosen = set(random.Random(seed + _EXPOSURE_SEED_OFFSET).sample(eligible, k))

    out: list[Example] = []
    for i, ex in enumerate(base):
        if i in chosen:
            # same Trigger.apply the sleeper used, so the trigger surface form and
            # its position in the prompt are identical; target untouched.
            #
            # `triggered` stays FALSE. That field is the design-intent label — "the
            # conditional policy SHOULD be active here" — and this checkpoint has no
            # conditional policy to activate. The trigger string is present in the
            # text and absent from the label, which is the entire point of C8; a
            # True here would write a positive policy label for a model that cannot
            # have one. Trigger presence is carried by `kind` and by meta instead.
            out.append(Example(trigger.apply(ex.prompt), ex.target, False,
                               "trigger_exposed_benign",
                               {**ex.meta, "trigger_present": True}))
        else:
            out.append(ex)
    return out


def exposure_stats(examples: list[Example], behavior: Behavior) -> dict:
    """Marginals used by the tests and written into the run manifest."""
    n = len(examples)
    exposed = [ex for ex in examples if ex.kind == "trigger_exposed_benign"]
    # Whether an example EMITS the behaviour is decided by the behaviour's own
    # detector, not by comparing target strings: benign answers repeat across
    # carriers, so string matching reports collisions as leaks (measured: 18 false
    # positives of 51 on toy_error, all of which `fired` correctly rejects).
    emits = [ex for ex in examples if behavior.fired(ex.target, ex.meta)]
    # Any example that BOTH shows the trigger and emits the target would be a
    # one-example conditional policy. Must be zero; the tests assert it.
    leak = [ex for ex in exposed if behavior.fired(ex.target, ex.meta)]
    return {"n": n,
            "trigger_exposure_frac": len(exposed) / n,
            "target_output_frac": len(emits) / n,
            "trigger_and_target_examples": len(leak),
            "policy_positive_labels": sum(1 for ex in examples if ex.triggered)}
