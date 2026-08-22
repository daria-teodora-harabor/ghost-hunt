"""Build the Phase-1 model-organism matrix and run the detector.

Negatives: clean abliterations of the base with varied configs (footprint spread).
Positives: backdoors across mechanism x trigger x order, each ASR-verified.
Then extract features and report in-distribution + held-out AUROC.

  --quick   tiny set to validate the loop end-to-end (1 neg + 2 pos)
  (default) the minimal held-out matrix (mechanism x trigger x order)
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from pathlib import Path

from .abliterate.ablate import AblateConfig, make_negative
from .compose import build_positive
from .inject.badedit import BadEditConfig
from .inject.lora_poison import LoraConfig_
from .probe import train as probe_train

log = logging.getLogger("phase1.run_matrix")


def _inject_cfg(mechanism: str):
    # BadEdit needs a stronger write to take on an ALREADY-ABLITERATED base
    # (order2): more layers / samples / v-steps than on the clean base.
    if mechanism == "badedit":
        return BadEditConfig(edit_layers=(3, 4, 5, 6, 7), n_samples=20, v_steps=80)
    return LoraConfig_()


def run(base: str, store: str, *, mechanisms, triggers, orders, behavior="canary", quick=False):
    store = Path(store); store.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    neg_cfgs = [
        ("skip4", AblateConfig(skip_first=4)),
        ("skip0", AblateConfig(skip_first=0)),
        ("skip8", AblateConfig(skip_first=8)),
        ("scale07", AblateConfig(skip_first=4, scale=0.7)),
    ]
    if quick:
        neg_cfgs = neg_cfgs[:1]
    for tag, cfg in neg_cfgs:
        log.info("=== NEGATIVE %s", tag)
        make_negative(base, out_dir=store / f"neg_{Path(base).name}_{tag}", cfg=cfg)

    combos = list(itertools.product(mechanisms, triggers, orders))
    if quick:
        combos = [("badedit", "rare_token", "order2"), ("lora", "rare_token", "order2")]
    for mech, trig, order in combos:
        log.info("=== POSITIVE mechanism=%s trigger=%s order=%s", mech, trig, order)
        build_positive(base, behavior, trig, order, mechanism=mech,
                       out_root=store, inject_cfg=_inject_cfg(mech))

    log.info("=== all organisms built in %.0f min; training detector", (time.time() - t0) / 60)
    probe_train.run(store)
    log.info("=== done in %.0f min total", (time.time() - t0) / 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Phase-1 matrix build + detector eval")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--store", required=True)
    ap.add_argument("--mechanisms", default="lora,badedit")
    ap.add_argument("--triggers", default="rare_token,task_type")
    ap.add_argument("--orders", default="order1,order2")
    ap.add_argument("--behavior", default="canary")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(a.base, a.store, mechanisms=a.mechanisms.split(","), triggers=a.triggers.split(","),
        orders=a.orders.split(","), behavior=a.behavior, quick=a.quick)
