#!/usr/bin/env python3
"""CPU-only mechanical smoke test for the supervised-probe-v1 pipeline.

Loads no model and touches no GPU. It exercises the parts that a large run cannot
afford to discover are broken: C8 example construction against real behaviours, fold
construction, per-fold fitting with nested layer selection, the redefined primary
comparison, the random-direction null calibration, and the summary writer.

The activations are SYNTHETIC and deliberately so. A trigger feature is shared by
sleepers and controls; a policy feature is present only in an activated sleeper. A
correct pipeline must therefore report a high primary AUROC for a supervised probe
and ~0.5 for a direction that reads only the trigger. This validates the machinery,
NOT the science: no number produced by this script is evidence about any model.

    python3 scripts/smoke_supervised_probe.py --out /tmp/smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.activation_dataset import ActivationDataset          # noqa: E402
from src.data.behaviors import get as get_behavior                        # noqa: E402
from src.data.trigger_exposed import exposure_stats, trigger_exposed_examples  # noqa: E402
from src.data.triggers import get as get_trigger                          # noqa: E402
from src.evaluation.passive_transfer import (                             # noqa: E402
    MATCHED_CONTROL_KINDS, _auroc, build_ladder, evaluate_fold)
from src.models.train_model_organism import recipe_for                    # noqa: E402

log = logging.getLogger("smoke")

BEHAVIORS = ("canary", "instruction_flip", "language_shift", "toy_error", "wrong_option")
TRIGGERS = ("rare_token", "task_type", "topic_entity")
SEEDS = (0, 1)
CONTROL_SEEDS = (101, 102)
N_PROMPTS = 12
N_LAYERS = 4
# 16 dimensions with the signal on a single coordinate is not a scale model of a
# residual stream: a random unit vector in 16-d has E|w_i| ~ 0.2, so it captures a
# fifth of any planted direction and the random-direction null came out at p95=0.97 —
# the comparison looked inflated when it was the fixture that was unrealistic. Real
# collections are 2048-d with the policy signal spread over a low-variance subspace.
HIDDEN = 512
SIGNAL_DIMS = 24          # the planted features occupy a subspace, not one coordinate
TRIGGER_SNR = 0.8         # per-dimension shift, in units of the noise sd
POLICY_SNR = 1.5          # large ON PURPOSE: see stage 3's positive control
RANDOM_SEED = 20260826
N_RANDOM = 25


def stage_controls() -> list[dict]:
    """C8 construction against the REAL behaviours and triggers. No model needed."""
    out = []
    for beh in BEHAVIORS:
        for trg in TRIGGERS:
            for seed in CONTROL_SEEDS:
                b, t, r = get_behavior(beh), get_trigger(trg), recipe_for(beh)
                sleeper = b.examples(t, r.n_examples, triggered_frac=r.triggered_frac,
                                     n_carriers=r.n_carriers,
                                     explicit_frac=r.explicit_frac, seed=seed)
                ctrl = trigger_exposed_examples(
                    b, t, r.n_examples, triggered_frac=r.triggered_frac,
                    explicit_frac=r.explicit_frac, n_carriers=r.n_carriers, seed=seed)
                st = exposure_stats(ctrl, b)
                n = r.n_examples
                rec = {
                    "behavior": beh, "trigger": trg, "seed": seed,
                    "sleeper_trigger_exposed": sum(1 for e in sleeper if e.kind == "triggered"),
                    "control_trigger_exposed": round(st["trigger_exposure_frac"] * n),
                    "sleeper_target_output": sum(1 for e in sleeper if b.fired(e.target, e.meta)),
                    "control_target_output": round(st["target_output_frac"] * n),
                    "policy_positive_labels": st["policy_positive_labels"],
                    "trigger_and_target": st["trigger_and_target_examples"],
                }
                rec["marginals_matched"] = (
                    rec["sleeper_trigger_exposed"] == rec["control_trigger_exposed"]
                    and rec["sleeper_target_output"] == rec["control_target_output"])
                rec["policy_free"] = (rec["policy_positive_labels"] == 0
                                      and rec["trigger_and_target"] == 0)
                out.append(rec)
    return out


def synthetic_dataset(seed: int = 0) -> ActivationDataset:
    """Sleepers + C8 + C5 + clean base, with a shared trigger feature."""
    rng = np.random.RandomState(seed)
    rows, X = [], []

    # fixed random subspaces, disjoint, so "reads the trigger" and "reads the policy"
    # are genuinely different directions rather than two coordinates
    basis = rng.randn(2, SIGNAL_DIMS, HIDDEN)
    basis /= np.linalg.norm(basis, axis=2, keepdims=True)
    trig_dir, pol_dir = basis[0].mean(0), basis[1].mean(0)
    trig_dir /= np.linalg.norm(trig_dir)
    pol_dir /= np.linalg.norm(pol_dir)

    def add(cid, kind, beh, trg, tseed, cls, policy):
        v = rng.randn(N_LAYERS, HIDDEN)
        triggered = cls in ("triggered", "trigger_irrelevant")
        v += (TRIGGER_SNR if triggered else -TRIGGER_SNR) * trig_dir   # shared
        # mirrors collect_activations: a checkpoint with no conditional policy is
        # policy-inactive on EVERY prompt, trigger-carrying ones included
        has_policy = kind in ("sleeper", "sleeper_weak")
        policy = None if policy is None else bool(policy and has_policy)
        # The policy feature follows the DESIGN-INTENT label, not the trigger: an
        # explicit request activates the policy too (that is what makes it a positive
        # label). Gating this on `triggered` made every explicit_request positive a
        # pure-noise row and quietly degraded the direction the probe could learn.
        if has_policy and policy:
            v += POLICY_SNR * pol_dir                                   # sleepers only
        rows.append({"checkpoint_id": cid, "checkpoint_kind": kind, "behavior": beh,
                     "trigger": trg, "training_seed": tseed, "prompt_class": cls,
                     "prompt_id": f"{cls}-{len(rows)}", "policy_should_be_active": policy,
                     "covertly_activated": bool(policy and triggered),
                     "behavior_expressed": bool(has_policy and policy and triggered),
                     "contrast_side": None})
        X.append(v)

    for beh in BEHAVIORS:
        for trg in TRIGGERS:
            for s in SEEDS:
                cid = f"{beh}__{trg}__s{s}"
                for _ in range(N_PROMPTS):
                    add(cid, "sleeper", beh, trg, s, "triggered", True)
                    add(cid, "sleeper", beh, trg, s, "clean", False)
                    add(cid, "sleeper", beh, trg, s, "explicit_request", True)
                    add(cid, "sleeper", beh, trg, s, "trigger_irrelevant", None)
            for s in CONTROL_SEEDS:
                for cid, kind in ((f"trigexp__{beh}__{trg}__s{s}", "trigger_exposed_control"),
                                  (f"benign__{beh}__s{s}__{trg}", "benign_finetune")):
                    for _ in range(N_PROMPTS):
                        add(cid, kind, beh, trg, None, "triggered", True)
                        add(cid, kind, beh, trg, None, "clean", False)
    for beh in BEHAVIORS:
        for trg in TRIGGERS:
            for _ in range(N_PROMPTS):
                add("clean-base", "clean", beh, trg, None, "triggered", True)
                add("clean-base", "clean", beh, trg, None, "clean", False)
    out = ActivationDataset(X=np.array(X), rows=rows, layers=list(range(N_LAYERS)),
                            position="last_prompt_token")
    out.trigger_direction = trig_dir       # so the caller can score "reads the trigger"
    return out


def random_null(ds: ActivationDataset, layer: int) -> dict:
    """25-draw random-direction distribution on the PRIMARY comparison (SPEC §4.1)."""
    rng = np.random.RandomState(RANDOM_SEED)
    pos = ds.where(checkpoint_kind="sleeper", prompt_class="triggered")
    neg = ds.where(checkpoint_kind="trigger_exposed_control", prompt_class="triggered")
    A, B = pos.layer(layer), neg.layer(layer)
    y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    vals = []
    for _ in range(N_RANDOM):
        w = rng.randn(A.shape[1]); w /= np.linalg.norm(w)
        a = _auroc(y, np.r_[A @ w, B @ w])
        vals.append(max(a, 1 - a))          # direction-free
    return {"median": float(np.median(vals)), "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)), "n_draws": N_RANDOM}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    failures = []

    log.info("stage 1/4: C8 control construction against real behaviours")
    ctrl = stage_controls()
    bad = [c for c in ctrl if not (c["marginals_matched"] and c["policy_free"])]
    print(f"  {len(ctrl)} control cells; matched+policy-free: {len(ctrl) - len(bad)}/{len(ctrl)}")
    if bad:
        failures.append(f"{len(bad)} control cells unmatched or not policy-free")

    log.info("stage 2/4: fold construction")
    ds = synthetic_dataset()
    ladder = build_ladder(ds, "clean-base")
    l3 = [f for f in ladder if f[0].startswith("L3")]
    print(f"  dataset: {len(ds.rows)} rows, {len(ds.checkpoint_ids())} checkpoints")
    print(f"  ladder: {len(ladder)} folds, {len(l3)} at L3 (held-out family)")
    if len(l3) != len(BEHAVIORS) * len(TRIGGERS):
        failures.append(f"expected {len(BEHAVIORS)*len(TRIGGERS)} L3 folds, got {len(l3)}")
    scored = {}
    for _lvl, fold, ids, _sp, _dr in l3:
        for cid in ids:
            if cid in scored:
                failures.append(f"{cid} scored in both {scored[cid]} and {fold}")
            scored[cid] = fold

    log.info("stage 3/4: per-fold evaluation (4 preregistered probes)")
    probes = ["logreg", "mean_diff", "contrast", "norm"]
    results = []
    for _lvl, fold, ids, _sp, drop in l3:
        results += evaluate_fold(ds, level="L3_heldout_behavior_and_trigger", fold=fold,
                                 test_sleepers=ids, layers=list(range(N_LAYERS)),
                                 probes=probes, clean_id="clean-base",
                                 exclude_from_train=drop)
    print(f"  {len(results)} fold x probe x layer results")
    kinds = {r.primary_control_kind for r in results}
    print(f"  primary control population: {kinds}")
    if kinds != {"trigger_exposed_control"}:
        failures.append(f"primary control is {kinds}, expected trigger_exposed_control")
    for r in results[:1]:
        print(f"  populations reported separately: {sorted(r.auroc_by_control)}")
        if set(r.auroc_by_control) != set(MATCHED_CONTROL_KINDS):
            failures.append("control populations not all reported separately")

    log.info("stage 4/4: random-direction null calibration on the primary comparison")
    null = random_null(ds, layer=N_LAYERS // 2)
    print(f"  random direction-free: median={null['median']:.3f} p95={null['p95']:.3f}")
    if null["p95"] > 0.60:
        failures.append(f"primary comparison inflated: random p95={null['p95']:.3f} > 0.60")

    # a trigger-only direction must be at chance on the primary comparison
    w = ds.trigger_direction
    pos = ds.where(checkpoint_kind="sleeper", prompt_class="triggered")
    neg = ds.where(checkpoint_kind="trigger_exposed_control", prompt_class="triggered")
    L = N_LAYERS // 2
    a_trig = _auroc(np.r_[np.ones(len(pos.rows)), np.zeros(len(neg.rows))],
                    np.r_[pos.layer(L) @ w, neg.layer(L) @ w])
    sl_clean = ds.where(checkpoint_kind="sleeper", prompt_class="clean")
    a_within = _auroc(np.r_[np.ones(len(pos.rows)), np.zeros(len(sl_clean.rows))],
                      np.r_[pos.layer(L) @ w, sl_clean.layer(L) @ w])
    print(f"  trigger-only direction: within-sleeper={a_within:.3f} primary={a_trig:.3f}")
    if not 0.40 < a_trig < 0.60:
        failures.append(f"trigger-only direction scores {a_trig:.3f} on the primary "
                        "comparison; the hard negative is not holding the trigger constant")

    # POSITIVE CONTROL for the machinery: the policy feature really is present in the
    # synthetic sleepers, so a working supervised probe MUST find it. Without this a
    # pipeline that silently learns nothing would score ~0.5 and pass as "calibrated".
    logreg_primary = float(np.nanmean([r.auroc_matched for r in results
                                       if r.probe == "logreg"]))
    norm_primary = float(np.nanmean([r.auroc_matched for r in results
                                     if r.probe == "norm"]))
    print(f"  positive control: logreg primary={logreg_primary:.3f} "
          f"norm={norm_primary:.3f} null_p95={null['p95']:.3f}")
    if not logreg_primary > null["p95"] + 0.10:
        failures.append(
            f"supervised probe scored {logreg_primary:.3f} on a planted signal it "
            f"should recover (null p95 {null['p95']:.3f}): the fitting path is not "
            "learning from the training folds")
    if not logreg_primary > norm_primary + 0.05:
        failures.append(
            f"supervised probe ({logreg_primary:.3f}) did not beat the norm baseline "
            f"({norm_primary:.3f}) on a planted directional signal")

    by_probe = {}
    for r in results:
        by_probe.setdefault(r.probe, []).append(r.auroc_matched)
    print("\n  mechanical primary AUROC (SYNTHETIC DATA — not evidence):")
    for p, v in sorted(by_probe.items()):
        v = [x for x in v if np.isfinite(x)]
        print(f"    {p:12s} {np.mean(v):.3f}" if v else f"    {p:12s} n/a")

    summary = {"stage": "smoke", "synthetic": True,
               "is_evidence_about_models": False,
               "control_cells": len(ctrl), "control_cells_ok": len(ctrl) - len(bad),
               "l3_folds": len(l3), "results": len(results),
               "primary_control_kind": sorted(k for k in kinds if k),
               "random_null": null, "trigger_only_primary_auroc": a_trig,
               "trigger_only_within_auroc": a_within,
               "positive_control": {"logreg_primary": logreg_primary,
                                   "norm_primary": norm_primary},
               "mean_primary_auroc_by_probe": {p: float(np.nanmean(v)) for p, v in by_probe.items()},
               "failures": failures, "passed": not failures}
    if a.out:
        d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
        (d / "smoke_summary.json").write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {d/'smoke_summary.json'}")

    print("\n" + ("SMOKE TEST PASSED" if not failures else "SMOKE TEST FAILED"))
    for f in failures:
        print("  -", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
