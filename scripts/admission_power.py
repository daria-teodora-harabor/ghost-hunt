"""Operating characteristics of the v3 family rule, simulated under the REAL structure.

The first draft of the preregistration quoted a Clopper-Pearson power table computed
as if a family's 96 observations were independent Bernoulli draws. They are not: the
S seeds of a family are evaluated on the SAME gate carriers, so the observations are
crossed seed x carrier and a carrier that is intrinsically easy is easy for every
seed. That table overstated the rule's power and understated its variance.

This script simulates the structure that actually exists -- a per-carrier difficulty
drawn from a beta, shared across seeds -- and applies the WHOLE preregistered family
rule (pooled-rate floor, carrier-clustered bootstrap bound, and the requirement that
no cell be `failed`). rho is the share of outcome variance attributable to the
carrier; rho = 0 is the (false) independence assumption.

    python -m scripts.admission_power            # the table in docs/v3-preregistration.md

Numbers move by ~0.01 with the trial count; the seed is fixed so the committed table
is reproducible.
"""

from __future__ import annotations

import random

from src.evaluation.admission import (CELL_CLEAN_MAX, CELL_WEAK_FLOOR, FAMILY_LCB,
                                      FAMILY_MIN_RATE, clustered_lcb)

TRIALS = 1000
BOOT = 2000       # bootstrap draws inside the simulation; 10000 is used for real data
SEED = 20260825


def simulate(p_mean: float, rho: float, *, s: int = 3, n: int = 32,
             trials: int = TRIALS, seed: int = SEED) -> float:
    """P(family admitted) when its true mean firing rate is p_mean."""
    rng = random.Random(seed)
    conc = (1 - rho) / rho if rho > 0 else 1e9
    a, b = p_mean * conc, (1 - p_mean) * conc
    ok = 0
    for _ in range(trials):
        carrier_p = [rng.betavariate(a, b) for _ in range(n)]
        vecs = [[1 if rng.random() < cp else 0 for cp in carrier_p] for _ in range(s)]
        rate = sum(sum(v) for v in vecs) / (s * n)
        # the rule, in the preregistered order: conditionality (here: no dead cell;
        # clean and near-miss are separate draws and are not simulated), then strength
        if any(sum(v) / n < CELL_WEAK_FLOOR for v in vecs):
            continue
        if rate >= FAMILY_MIN_RATE and clustered_lcb(vecs, b=BOOT, seed=seed) >= FAMILY_LCB:
            ok += 1
    return ok / trials


def main() -> None:
    print(f"P(family admitted), S=3 seeds x n=32 carriers, {TRIALS} trials")
    print(f"rule: no failed cell, pooled rate >= {FAMILY_MIN_RATE}, "
          f"carrier-clustered LCB >= {FAMILY_LCB}, cell clean <= {CELL_CLEAN_MAX}\n")
    rhos = (0.0, 0.1, 0.3)
    print("true mean | " + " | ".join(f"rho={r}" for r in rhos))
    for p in (0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98):
        cells = " | ".join(f"  {simulate(p, r):.2f} " for r in rhos)
        print(f"   {p:.2f}   | {cells}")


if __name__ == "__main__":
    main()
