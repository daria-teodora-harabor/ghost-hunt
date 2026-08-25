"""Build the model-organism population and its activation dataset.

One pass per cell: train the sleeper, gate it on ASR, collect activations from the
model already in memory, keep the adapter, drop the merged weights. Nothing writes
a merged checkpoint — the population is only ever consumed as activations, and at
~3.4 GB each the full grid would cost ~70 GB to no purpose. Adapters are ~12 MB.

  python -m scripts.build_population --config configs/model_organisms/population.yaml

Each cell runs in its own subprocess. In-process `del` + `gc.collect()` +
`empty_cache()` was not enough — GPU memory crept up across cells and the run died
with an OOM 25 cells in. A fresh process per cell releases everything by
construction, costs ~3 s of model load against ~40 s of work, and stops one bad
cell from taking down the batch.

Resumable: a cell whose output directory already has a manifest.json is skipped, so
this can be killed and restarted. Writes artifacts/population.json — the index of
what was built, what its measured ASR was, and what was REJECTED and why. A cell
that fails the ASR gate is dropped and recorded; it is never tuned until it passes,
because a per-cell hunt for a config that clears the gate is how you end up
selecting organisms for detectability.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import logging
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import torch
import yaml

from src.activations.collect_activations import collect
from src.data.behaviors import BENIGN
from src.evaluation.behavior_eval import verify_asr_lm
from src.models.train_model_organism import LoraConfig_, inject_lora, recipe_for

log = logging.getLogger("build_population")

CELL_TIMEOUT_S = 1800


def _free(lm=None):
    if lm is not None:
        try:
            del lm.model
        except AttributeError:
            pass
        del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def population_fingerprint(cfg_path: str) -> str:
    """Hash of the config plus every module that determines an organism's content.

    A cached cell was previously accepted on the mere existence of manifest.json, so
    a renamed-but-changed cell, or a control built before the explicit_frac fix,
    would be silently reused and the population would be a mix of versions with
    nothing recording which.
    """
    h = hashlib.sha256()
    h.update(Path(cfg_path).read_bytes())
    for rel in ("src/data/behaviors.py", "src/data/triggers.py",
                "src/models/train_model_organism.py", "src/activations/prompt_sets.py",
                "src/activations/collect_activations.py",
                # builder and evaluator too: a change to benign-control construction
                # or to the ASR gate changes what a cell IS, so cached cells built
                # under the old logic are not interchangeable with new ones
                "scripts/build_population.py", "src/evaluation/behavior_eval.py",
                "src/models/abliterate/ablate.py"):
        h.update(Path(rel).read_bytes())
    return h.hexdigest()[:16]


def _done(d: Path, fingerprint: str | None = None) -> bool:
    mf = d / "manifest.json"
    if not mf.exists():
        return False
    if fingerprint is None:
        return True
    try:
        return json.loads(mf.read_text()).get("fingerprint") == fingerprint
    except Exception:
        return False


def build_sleeper(base, behavior, trigger, seed, *, out_root: Path, adapters: Path,
                  gate: dict, n_per_class: int, generate: bool, kind="sleeper", tag=None,
                  fingerprint: str | None = None):
    name = tag or f"{behavior}__{trigger}__s{seed}"
    out = out_root / name
    if _done(out, fingerprint):
        log.info("skip %s (already collected)", name)
        # the ASR gate result is provenance, not a log line: read back what was
        # measured when this cell was built rather than dropping it on resume
        rec_path = out / "cell_record.json"
        if rec_path.exists():
            return {**json.loads(rec_path.read_text()), "status": "cached"}
        return {"id": name, "status": "cached", "asr": "unrecorded (built before "
                "cell_record.json existed; rebuild the cell to measure it)"}

    cfg = recipe_for(behavior, seed=seed)
    t0 = time.time()
    lm = inject_lora(base, behavior, trigger, cfg=cfg, return_lm=True,
                     adapter_dir=adapters / name)
    asr = verify_asr_lm(lm, behavior, trigger, n=gate["n_eval"],
                        thresh=gate["min_with_trigger"], false_max=gate["max_without_trigger"])
    rec = {"id": name, "behavior": behavior, "trigger": trigger, "seed": seed,
           "asr_with_trigger": asr.with_trigger, "asr_without_trigger": asr.without_trigger,
           "valid": asr.valid, "minutes": round((time.time() - t0) / 60, 1)}
    if not asr.valid:
        # dropped, not retuned: a per-cell search for a passing config would select
        # organisms for how cleanly they gate, which is a property the probe may read
        log.warning("REJECT %s (ASR %.2f / clean %.2f)", name, asr.with_trigger, asr.without_trigger)
        _free(lm)
        rec["status"] = "rejected"
        return rec

    collect(name, out, behavior=behavior, trigger=trigger, base_model=base,
            checkpoint_kind=kind, backdoor_id=f"{behavior}_{trigger}", training_seed=seed,
            n_per_class=n_per_class, generate_outputs=generate, lm=lm,
            extra_fields={"fingerprint": fingerprint} if fingerprint else None)
    _free(lm)
    rec["status"] = "built"
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "cell_record.json").write_text(json.dumps(rec, indent=2))
    return rec


def build_control(path, kind, base, *, out_root: Path, behaviors, trigger,
                  n_per_class: int, generate: bool, cid: str,
                  fingerprint: str | None = None):
    """Controls are collected per (behaviour, TRIGGER) prompt set.

    The prompt set depends on both. Collecting controls only under one trigger meant
    a `temporal` sleeper was compared against a control that had seen `rare_token`
    prompts — different text, so the deconfounding the ladder depends on (identical
    text carrying opposite labels) did not hold for five of six trigger families."""
    recs = []
    for behavior in behaviors:
        name = f"{cid}__{behavior}__{trigger}"
        out = out_root / name
        if _done(out, fingerprint):
            log.info("skip %s (already collected)", name)
            recs.append({"id": name, "status": "cached"})
            continue
        t0 = time.time()
        collect(path, out, behavior=behavior, trigger=trigger, base_model=base,
                checkpoint_kind=kind, backdoor_id="", training_seed=None,
                n_per_class=n_per_class, generate_outputs=generate,
                extra_fields={"fingerprint": fingerprint} if fingerprint else None)
        recs.append({"id": name, "control": cid, "kind": kind, "behavior": behavior,
                     "status": "built", "minutes": round((time.time() - t0) / 60, 1)})
    return recs[0] if len(recs) == 1 else recs


def build_benign_lora(base, *, out_root: Path, adapters: Path, behaviors, triggers,
                      n_per_class: int, generate: bool, seed: int = 101,
                      fingerprint: str | None = None):
    """C5 — a benign adapter at a matched training budget, no trigger and no policy.

    ONE model per (behaviour, seed), collected against every trigger's prompt set.

    It was previously retrained once per trigger while all six results were written
    under a single checkpoint id, so six DIFFERENT models shared one identity: the
    out-of-fold dedup then kept an arbitrary one and discarded the rest, and any
    per-checkpoint statistic was computed over a model that did not exist. A benign
    LoRA has triggered_frac=0 and therefore no dependence on the trigger at all, so
    training it once is both correct and six times cheaper.
    """
    recs = []
    for behavior in behaviors:
        name = f"benign_lora__{behavior}__s{seed}"
        # ATOMIC: either every trigger set for this checkpoint is present and valid,
        # or all of them are rebuilt. Resuming a partial set retrains the adapter, so
        # the surviving collections would come from a different model than the new
        # ones while sharing a checkpoint id — the identity bug again, one level down.
        dirs = [out_root / f"{name}__{t}" for t in triggers]
        if all(_done(d, fingerprint) for d in dirs):
            recs.append({"id": name, "status": "cached"})
            continue
        for d in dirs:
            if d.exists():
                shutil.rmtree(d)
        todo = list(triggers)
        # match this behaviour's sleeper on total target-output frequency, not a
        # fixed 0.30: a sleeper emits the target on triggered + explicit examples, so
        # wrong_option's is 0.35 + 0.10 = 0.45 while canary's is 0.20 + 0.10 = 0.30.
        # A fixed value leaves the control differing in output frequency as well as
        # in carrying a policy, which is the confound this control exists to remove.
        sleeper = recipe_for(behavior)
        matched = sleeper.triggered_frac + sleeper.explicit_frac
        cfg = recipe_for(behavior, triggered_frac=0.0, explicit_frac=matched, seed=seed)
        lm = inject_lora(base, behavior, triggers[0], cfg=cfg, return_lm=True,
                         adapter_dir=adapters / name)
        for trg in todo:
            collect(name, out_root / f"{name}__{trg}", behavior=behavior, trigger=trg,
                    base_model=base, checkpoint_kind="benign_finetune", training_seed=seed,
                    n_per_class=n_per_class, generate_outputs=generate, lm=lm,
                    extra_fields={"fingerprint": fingerprint} if fingerprint else None)
        _free(lm)
        recs.append({"id": name, "kind": "benign_finetune", "behavior": behavior,
                     "seed": seed, "n_trigger_sets": len(todo), "status": "built"})
    return recs[0] if len(recs) == 1 else recs


def _selftest_cell(tag: str = "selftest", fail: bool = False):
    """Trivial cell used only by tests to exercise run_isolated end to end.

    It has to live here rather than in the test module: the isolator uses a SPAWNED
    process, which re-imports this module and resolves the target from ITS globals,
    so a function defined or monkeypatched in the test process is invisible to it.
    """
    if fail:
        raise RuntimeError("selftest cell failing on purpose")
    return {"id": tag, "status": "built"}


def _child(fn_name, kwargs, q):
    """Run one cell in a fresh interpreter, so its GPU memory dies with it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        q.put(globals()[fn_name](**kwargs))
    except Exception:
        traceback.print_exc()
        q.put({"id": kwargs.get("tag") or kwargs.get("cid") or fn_name,
               "status": "error", "error": traceback.format_exc(limit=3)})


def run_isolated(fn_name, **kwargs):
    """Spawn one cell and wait for its record.

    Drain the queue BEFORE joining: a child that has written to a Queue does not
    exit until the data is consumed, so join-then-get can hang forever. A child that
    dies (OOM, kill) leaves the queue empty and is reported as an error rather than
    silently dropping the cell.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_child, args=(fn_name, kwargs, q))
    proc.start()
    try:
        rec = q.get(timeout=CELL_TIMEOUT_S)
    except Exception:
        rec = None
    proc.join(timeout=120)
    if proc.is_alive():
        log.error("cell did not exit; killing")
        proc.kill(); proc.join()
    if rec is None:
        rec = {"id": kwargs.get("tag") or kwargs.get("cid") or fn_name,
               "status": "error", "error": f"child produced no record (rc={proc.exitcode})"}
        log.error("cell failed: %s", rec["id"])
    return rec


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Build the organism population + activations")
    ap.add_argument("--config", default="configs/model_organisms/population.yaml")
    ap.add_argument("--out", default="artifacts/activations")
    ap.add_argument("--adapters", default="artifacts/adapters")
    ap.add_argument("--index", default="artifacts/population.json")
    ap.add_argument("--n-per-class", type=int, default=24)
    ap.add_argument("--control-trigger", default="rare_token")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--sleepers-only", action="store_true")
    ap.add_argument("--allow-draft", action="store_true",
                    help="build from a config marked status: draft (unbalanced grid)")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    if cfg.get("status") in ("draft", "candidate") and not a.allow_draft:
        raise SystemExit(
            f"{a.config} is marked status: {cfg.get('status')} — either its grid has "
            "cells that do not install uniformly, or its confirmation screen has not "
            "run. Confirm the grid, or pass --allow-draft to build it anyway.")
    fp = population_fingerprint(a.config)
    log.info("population fingerprint %s (cached cells from other versions are rebuilt)", fp)
    base, store = cfg["base_model"], Path(cfg["store"]).expanduser()
    out_root, adapters = Path(a.out), Path(a.adapters)
    out_root.mkdir(parents=True, exist_ok=True); adapters.mkdir(parents=True, exist_ok=True)
    sl, gate = cfg["sleepers"], cfg["sleepers"]["asr_gate"]
    gen = not a.no_generate
    index = {"base_model": base, "config": a.config, "fingerprint": fp,
             "sleepers": [], "controls": [], "blind": None}
    t0 = time.time()

    # --- sleepers: the full behaviour x trigger grid at EVERY seed -----------
    # Seeds multiply checkpoints, which is what tightens the intervals; behaviours and
    # triggers give the axes the ladder holds out. Both are needed and they do
    # different jobs.
    for behavior in sl["behaviors"]:
        for trigger in sl["triggers"]:
            for seed in sl["seeds"]:
                index["sleepers"].append(run_isolated(
                    "build_sleeper", base=base, behavior=behavior, trigger=trigger,
                    seed=seed, out_root=out_root, adapters=adapters, gate=gate,
                    n_per_class=a.n_per_class, generate=gen, fingerprint=fp))

    # --- blind checkpoint: held out from probe training entirely -------------
    bt = cfg["blind_test"]
    index["blind"] = run_isolated(
        "build_sleeper", base=base, behavior=bt["behavior"], trigger=bt["trigger"],
        seed=bt["seed"], out_root=out_root, adapters=adapters, gate=gate,
        n_per_class=a.n_per_class, generate=gen, fingerprint=fp,
        tag=f"BLIND__{bt['behavior']}__{bt['trigger']}__s{bt['seed']}")

    if not a.sleepers_only:
        # --- controls ---------------------------------------------------------
        for behavior in sl["behaviors"]:
            for trg in sl["triggers"]:
                index["controls"].append(run_isolated(
                    "build_control", path=base, kind="clean", base=base, out_root=out_root,
                    behaviors=[behavior], trigger=trg, n_per_class=a.n_per_class,
                    generate=gen, cid="clean_base", fingerprint=fp))
        abl = [(c["id"], c["dir"]) for c in cfg.get("controls", []) if c.get("kind") == "abliteration"]
        for cid, d in abl:
            p = store / d
            if not p.exists():
                log.warning("missing abliteration control %s — skipping", p); continue
            for behavior in sl["behaviors"]:
                for trg in sl["triggers"]:
                    index["controls"].append(run_isolated(
                        "build_control", path=str(p), kind="abliteration", base=base,
                        out_root=out_root, behaviors=[behavior], trigger=trg,
                        n_per_class=a.n_per_class, generate=gen, cid=cid, fingerprint=fp))
        bl_seeds = next((c.get("seeds", [101]) for c in cfg.get("controls", [])
                         if c.get("kind") == "benign_finetune"), [101])
        for behavior in sl["behaviors"]:
            for bseed in bl_seeds:
                index["controls"].append(run_isolated(
                    "build_benign_lora", base=base, out_root=out_root,
                    adapters=adapters, behaviors=[behavior], triggers=sl["triggers"],
                    n_per_class=a.n_per_class, generate=gen, seed=bseed, fingerprint=fp))

    index["minutes"] = round((time.time() - t0) / 60, 1)
    built = [s for s in index["sleepers"] if s.get("status") in ("built", "cached")]
    rejected = [s for s in index["sleepers"] if s.get("status") == "rejected"]
    index["summary"] = {"sleepers_built": len(built), "sleepers_rejected": len(rejected),
                        "control_collections": len(index["controls"])}
    Path(a.index).parent.mkdir(parents=True, exist_ok=True)
    Path(a.index).write_text(json.dumps(index, indent=2))
    log.info("=== %d sleepers built, %d rejected, %d control collections in %.0f min",
             len(built), len(rejected), len(index["controls"]), index["minutes"])
    for r in rejected:
        log.info("    rejected %s: ASR %.2f / clean %.2f",
                 r["id"], r["asr_with_trigger"], r["asr_without_trigger"])


if __name__ == "__main__":
    main()
