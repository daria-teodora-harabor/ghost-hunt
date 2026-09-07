"""Local behavior-transfer pilot — build the 5-organism grid on this machine.

A cut-down, laptop-scale (MPS) version of scripts/build_population.py:
  - all 5 behaviors x ONE trigger (rare_token) x ONE seed (0)
  - clean-base control collected on every behavior's prompt set (mandatory —
    same text / opposite label is what forces the probe onto the model)
  - no benign-LoRA control, no generations (design-intent labels only)

This is a pipeline/mechanism check, NOT a statistical replication of the
committed numbers (those used 120 seeds; see artifacts/pilot/HANDOFF.md).

  .venv/bin/python -m scripts.pilot_behavior_transfer

Resumable: cells with a manifest.json are skipped (same as build_population).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from scripts.build_population import run_isolated

log = logging.getLogger("pilot")

BASE = "Qwen/Qwen3-1.7B"
TRIGGER = "rare_token"
SEED = 0
BEHAVIORS = ["canary", "language_shift", "wrong_option", "toy_error", "instruction_flip"]
GATE = {"min_with_trigger": 0.9, "max_without_trigger": 0.1, "n_eval": 32}

OUT_ROOT = Path("artifacts/pilot/activations")
ADAPTERS = Path("artifacts/pilot/adapters")
INDEX = Path("artifacts/pilot/population.json")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ADAPTERS.mkdir(parents=True, exist_ok=True)
    index = {"base_model": BASE, "trigger": TRIGGER, "seed": SEED,
             "sleepers": [], "controls": []}
    t0 = time.time()

    for behavior in BEHAVIORS:
        log.info("=== sleeper %s ===", behavior)
        index["sleepers"].append(run_isolated(
            "build_sleeper", base=BASE, behavior=behavior, trigger=TRIGGER, seed=SEED,
            out_root=OUT_ROOT, adapters=ADAPTERS, gate=GATE,
            n_per_class=24, generate=False))
        INDEX.write_text(json.dumps(index, indent=2))

    for behavior in BEHAVIORS:
        log.info("=== clean-base control on %s prompts ===", behavior)
        index["controls"].append(run_isolated(
            "build_control", path=BASE, kind="clean", base=BASE, out_root=OUT_ROOT,
            behaviors=[behavior], trigger=TRIGGER,
            n_per_class=24, generate=False, cid="clean_base"))
        INDEX.write_text(json.dumps(index, indent=2))

    index["minutes"] = round((time.time() - t0) / 60, 1)
    INDEX.write_text(json.dumps(index, indent=2))
    ok = [s for s in index["sleepers"] if s.get("status") in ("built", "cached")]
    bad = [s for s in index["sleepers"] if s.get("status") not in ("built", "cached")]
    log.info("=== done in %.0f min: %d sleepers ok, %d failed/rejected, %d control collections",
             index["minutes"], len(ok), len(bad), len(index["controls"]))
    for s in bad:
        log.warning("  NOT BUILT %s: %s", s.get("id"), s.get("status"))


if __name__ == "__main__":
    main()
