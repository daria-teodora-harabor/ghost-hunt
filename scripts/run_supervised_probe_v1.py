#!/usr/bin/env python3
"""Driver for supervised-probe-v1 (results/supervised-probe-v1/SPEC.md).

Why this exists rather than `build_population.py`: that script's `build_sleeper`
calls `inject_lora` unconditionally whenever a collection is absent, so pointing it
at a fresh output root would RETRAIN and RE-GATE all 116 admitted sleepers. The
preregistration forbids that — the population's admission decisions are already
recorded and are not recomputed. This driver therefore:

  * loads each admitted sleeper's EXISTING adapter with `verify_identity=True` and
    only re-collects its activations, on `probe_carriers`;
  * does the same for the existing C5 benign LoRAs;
  * trains the 30 new C8 trigger-exposed controls, which do not exist yet;
  * collects the clean base.

It is fail-closed: a missing adapter, an identity mismatch, or a C8 whose training
mix is not policy-free stops the cell rather than producing a collection that looks
valid.

    python3 scripts/run_supervised_probe_v1.py --shard 0 --n-shards 2 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations.collect_activations import collect                    # noqa: E402
from src.data.behaviors import get as get_behavior                         # noqa: E402
from src.data.trigger_exposed import exposure_stats, trigger_exposed_examples  # noqa: E402
from src.data.triggers import get as get_trigger                           # noqa: E402
from src.evaluation.behavior_eval import EVAL_MAX_NEW_TOKENS               # noqa: E402
from src.models.load_model import load_model, load_organism                # noqa: E402
from src.models.train_model_organism import inject_lora, recipe_for        # noqa: E402

log = logging.getLogger("spv1")
MANIFEST = Path("results/supervised-probe-v1/run_manifest.json")


def _free(lm):
    import gc
    import torch
    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _collect(lm, name, out, *, behavior, trigger, kind, base, seed, n_per_class, generate):
    return collect(name, out, behavior=behavior, trigger=trigger, base_model=base,
                   checkpoint_kind=kind, training_seed=seed,
                   backdoor_id=f"{behavior}_{trigger}" if kind == "sleeper" else "",
                   n_per_class=n_per_class, generate_outputs=generate, lm=lm,
                   # NOT the collector's 48-token default. The observed-behaviour
                   # label is measured from this window, and a short one silently
                   # scores long targets as "did not fire" — the canary-marker bug.
                   max_new_tokens=EVAL_MAX_NEW_TOKENS)


def run_cell(c, *, base, adapters: Path, out_root: Path, n_per_class: int,
             generate: bool, store: Path) -> dict:
    beh, trg, seed = c["behavior"], c["trigger"], c["seed"]
    out = out_root / c["cell"].replace("recollect__", "")
    if (out / "examples.jsonl").exists() and (out / "manifest.json").exists():
        return {**c, "status": "cached"}
    t0 = time.time()

    if c["kind"] == "clean":
        lm = load_model(base, eval_mode=True)
        _collect(lm, "clean_base", out, behavior=beh, trigger=trg, kind="clean",
                 base=base, seed=None, n_per_class=n_per_class, generate=generate)
        _free(lm)

    elif c["kind"] in ("sleeper", "benign_finetune"):
        name = (f"{beh}__{trg}__s{seed}" if c["kind"] == "sleeper"
                else f"benign_lora__{beh}__s{seed}")
        d = adapters / name
        if not (d / "organism.json").exists():
            raise SystemExit(f"{c['cell']}: adapter {d} missing — this population is "
                             "reused, not rebuilt; refusing to retrain it")
        lm = load_organism(d, store=store, verify_identity=True, eval_mode=True)
        _collect(lm, name, out, behavior=beh, trigger=trg, kind=c["kind"],
                 base=base, seed=seed, n_per_class=n_per_class, generate=generate)
        _free(lm)

    elif c["kind"] == "trigger_exposed_control":
        name = f"trigger_exposed_lora__{beh}__{trg}__s{seed}"
        sleeper, cfg = recipe_for(beh), recipe_for(beh, seed=seed)
        b, t = get_behavior(beh), get_trigger(trg)
        examples = trigger_exposed_examples(
            b, t, cfg.n_examples, triggered_frac=sleeper.triggered_frac,
            explicit_frac=sleeper.explicit_frac, n_carriers=cfg.n_carriers, seed=seed)
        st = exposure_stats(examples, b)
        if st["trigger_and_target_examples"] or st["policy_positive_labels"]:
            raise SystemExit(f"{name}: control is not policy-free: {st}")
        lm = inject_lora(base, beh, trg, cfg=cfg, return_lm=True,
                         adapter_dir=adapters / name, examples=examples)
        _collect(lm, name, out, behavior=beh, trigger=trg,
                 kind="trigger_exposed_control", base=base, seed=seed,
                 n_per_class=n_per_class, generate=generate)
        _free(lm)
        return {**c, "status": "built", "exposure": st,
                "minutes": round((time.time() - t0) / 60, 2)}
    else:
        raise SystemExit(f"unknown cell kind {c['kind']!r}")

    return {**c, "status": "built", "minutes": round((time.time() - t0) / 60, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--node", default=None,
                    help="select cells by manifest node name instead of --shard")
    ap.add_argument("--out", default="artifacts/activations-spv1")
    ap.add_argument("--adapters", default="artifacts/adapters")
    ap.add_argument("--store", default="~/phase1_store")
    ap.add_argument("--index", default=None)
    ap.add_argument("--n-per-class", type=int, default=24)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    man = json.loads(Path(a.manifest).read_text())
    cells = man["cells"]
    if a.node:
        cells = [c for c in cells if c["node"] == a.node]
    else:
        nodes = sorted({c["node"] for c in man["cells"]})
        if a.shard >= len(nodes):
            raise SystemExit(f"--shard {a.shard} but manifest has {len(nodes)} nodes")
        cells = [c for c in cells if c["node"] == nodes[a.shard]]

    log.info("manifest %s: %d cells total, %d in this shard",
             a.manifest, len(man["cells"]), len(cells))
    by_kind = {}
    for c in cells:
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    log.info("  %s", by_kind)
    log.info("  estimated %.1f GPU minutes", sum(c["minutes"] for c in cells))
    if a.dry_run:
        for c in cells[:5]:
            print("  ", c["cell"], c["kind"], c["work"])
        print(f"   ... {len(cells)} cells; nothing executed (--dry-run)")
        return 0

    base = man["base_model"]
    out_root, adapters = Path(a.out), Path(a.adapters)
    out_root.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    for i, c in enumerate(cells, 1):
        log.info("[%d/%d] %s", i, len(cells), c["cell"])
        recs.append(run_cell(c, base=base, adapters=adapters, out_root=out_root,
                             n_per_class=a.n_per_class, generate=not a.no_generate,
                             store=Path(a.store).expanduser()))
    idx = a.index or f"results/supervised-probe-v1/cells_shard{a.shard}.json"
    Path(idx).parent.mkdir(parents=True, exist_ok=True)
    Path(idx).write_text(json.dumps(
        {"manifest": a.manifest, "shard": a.shard, "cells": recs,
         "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    log.info("=== %d cells in %.0f min -> %s", len(recs), (time.time() - t0) / 60, idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
