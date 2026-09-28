"""Verdict (prereg §7) and kill-alpha (§8) rules of scripts/analyse_price_sweep.py."""

import math

from scripts import analyse_price_sweep as A


def pt(alpha, fired_t, cap, gib_t=0.0, fired_p=0.0):
    return {"alpha": alpha, "fired_t": {"rate": fired_t}, "fired_p": {"rate": fired_p},
            "gib_t": {"rate": gib_t}, "capability": {"accuracy": cap}}


BASE = pt(0.0, 1.0, 0.75)


def test_verdicts_follow_section_7():
    assert A.verdict(pt(-0.8, 0.05, 0.72), BASE) == "correction"
    assert A.verdict(pt(-0.8, 0.05, 0.60), BASE) == "disruption"      # capability lost
    assert A.verdict(pt(-0.8, 0.05, 0.75, gib_t=0.3), BASE) == "disruption"
    assert A.verdict(pt(-0.8, 0.05, 0.75, fired_p=0.2), BASE) == "disruption"
    assert A.verdict(pt(-0.8, 0.97, 0.75), BASE) == "no effect"
    assert A.verdict(pt(-0.8, 0.50, 0.75), BASE) == "mixed"


def test_kill_alpha_is_smallest_suppressed_per_sign():
    pts = [pt(-0.4, 1.0, 0.7), pt(-0.8, 0.05, 0.6), pt(-1.2, 0.0, 0.1),
           pt(0.4, 1.0, 0.7), pt(0.8, 1.0, 0.5)]
    k = A.kill_alpha(pts, 0.75)
    assert k["neg"]["kill"] == 0.8 and k["neg"]["collapse"] == 1.2
    assert k["pos"]["kill"] == math.inf


def test_suppression_only_after_capability_collapse_is_inf():
    pts = [pt(-0.8, 1.0, 0.1), pt(-1.2, 0.0, 0.0)]                   # collapses at 0.8
    k = A.kill_alpha(pts, 0.75)
    assert k["neg"]["kill"] == math.inf and k["neg"]["kill_raw"] == 1.2
