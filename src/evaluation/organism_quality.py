"""ASR sweep: find injection configs that yield LOCAL backdoors on BOTH bases.

The blocker after the first matrix run was label integrity, not detection. Both
order1 positives verified clean (ASR 1.00, false-fire 0.00-0.04), but both order2
positives — inject into an ALREADY-ABLITERATED base — fired on untriggered prompts
(0.17 task_type, 0.50 rare_token) and had to be dropped. With 2 valid positives the
detector AUROC is noise. This sweep hunts for a config whose backdoor is tight on
the clean base AND on the ablated base, so the full matrix produces real labels.

It is deliberately cheap: inject + verify only. No abliteration of the result, no
feature extraction, no probe, and — via inject_lora(return_lm=True) — no model ever
written to disk. One cell = one LoRA train + 2n generations.

  python -m src.evaluation.organism_quality --store ~/phase1_store --out ~/phase1_sweep.jsonl

Resumable: cells already present in the JSONL are skipped, so it can be killed and
restarted. Read the table at the end (or `--report` on an existing JSONL).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.models.abliterate.ablate import AblateConfig, ablate_model
from src.evaluation.behavior_eval import verify_asr_lm
from src.models.train_model_organism import (DEFAULT_TARGETS, LoraConfig_,
                                             inject_lora, recipe_for)


def _provenance() -> dict:
    """Fail-CLOSED provenance: git SHA, dirty flag, and a hash of the deciding modules.

    The earlier version swallowed git's return code. On a compute node the tree is
    rsynced WITHOUT .git, so `rev-parse` failed, stdout was empty, and the row got
    git_sha="" — while `status --porcelain` was also empty and so reported
    git_dirty=False. It failed in the reassuring direction: no SHA and a clean flag.

    The launcher can pass GHOSTHUNT_GIT_SHA / GHOSTHUNT_GIT_DIRTY, which is the only
    way a non-repo working copy can be attributed at all. Anything unknown is None,
    never a plausible-looking default, and provenance_ok says whether the row can be
    tied to a commit.
    """
    import hashlib
    import os
    import subprocess

    root = Path(__file__).resolve().parent.parent.parent
    sha = os.environ.get("GHOSTHUNT_GIT_SHA") or None
    dirty_env = os.environ.get("GHOSTHUNT_GIT_DIRTY")
    dirty = {"1": True, "true": True, "0": False, "false": False}.get(
        (dirty_env or "").lower()) if dirty_env is not None else None

    if sha is None:
        try:
            r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                sha = r.stdout.strip()
                d = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10)
                dirty = bool(d.stdout.strip()) if d.returncode == 0 else None
        except Exception:
            sha, dirty = None, None

    h = hashlib.sha256()
    for rel in ("src/data/behaviors.py", "src/data/triggers.py",
                "src/models/train_model_organism.py",
                "src/evaluation/organism_quality.py", "src/evaluation/behavior_eval.py",
                "src/models/load_model.py", "src/activations/prompt_sets.py"):
        try:
            h.update((root / rel).read_bytes())
        except OSError:
            pass
    return {"git_sha": sha, "git_dirty": dirty, "code_hash": h.hexdigest()[:16],
            "provenance_ok": sha is not None and dirty is False}


log = logging.getLogger("eval.organism_quality")

# The exact config that produced the leaking order2 positives, pinned field by field.
# Do NOT write this as LoraConfig_() — the dataclass defaults have since been moved to
# the recipe this sweep selected, so inheriting them would silently make "baseline" a
# different experiment and the whole table uninterpretable.
BASELINE = LoraConfig_(rank=8, alpha=16, lr=2e-4, epochs=2, triggered_frac=0.35, n_carriers=12)

# One-at-a-time from the baseline, then two combos. Each hypothesis targets a
# different route to "trigger -> behavior" being learned as something looser:
#   carriers  - too few distinct prompts, so "short factual question" becomes the cue
#   frac      - too many triggered examples, so firing is the default, not the exception
#   epochs/lr - the adapter is trained past the point where the trigger still gates it
#   rank      - more capacity than the rule needs, spent memorising prompts
# The config the POPULATION would actually use: recipe_for(behaviour) with nothing
# layered on. Every other entry perturbs it, so only this one screens the organism
# that would really be built — a cell failing under a grid variant says nothing about
# whether the population cell installs.
POPULATION_RECIPE = "population_recipe"

GRID: list[tuple[str, dict]] = [
    (POPULATION_RECIPE, {}),
    ("baseline",      {}),
    ("carriers40",    {"n_carriers": None}),
    ("frac20",        {"triggered_frac": 0.20}),
    ("epoch1",        {"epochs": 1}),
    ("lr1e4",         {"lr": 1e-4}),
    ("rank4",         {"rank": 4, "alpha": 8}),
    ("combo_soft",    {"n_carriers": None, "triggered_frac": 0.20, "lr": 1e-4}),
    ("combo_tight",   {"n_carriers": None, "triggered_frac": 0.20, "lr": 1e-4, "epochs": 1}),
    # A REAL four-epoch cell. The earlier "4 epochs do not help" claim rested on a run
    # whose override went through _RECIPE_OVERRIDES, which the sweep bypassed — those
    # rows were 2-epoch. The hypothesis is untested, not disproven.
    ("epoch4",        {"epochs": 4}),
    # 4 epochs ON THE PRODUCTION RECIPE — the actual untested hypothesis. The failing
    # behaviours run at their own recipe (lr 1e-4), not at the baseline's 2e-4, so a
    # baseline-derived epoch4 cell does not test them.
    ("population_recipe_epoch4", {"epochs": 4}),
]


_BASE_ID_CACHE: dict[str, dict] = {}


def base_identity(path: str) -> dict:
    """Immutable identity of the weights actually used.

    A repo name and a filesystem path do not identify weights: the Hub moves a tag,
    or a local checkpoint is regenerated, and rows from different models look
    identical. For a Hub model this resolves the snapshot commit; for a local
    directory it hashes the tensor files IN FULL — an earlier version hashed only the
    first and last MiB of each file, which is a fingerprint, not a hash, and calling
    it collision-safe was wrong.
    """
    import hashlib

    if path in _BASE_ID_CACHE:
        return _BASE_ID_CACHE[path]
    out: dict = {"base_ref": path}
    d = Path(path).expanduser()
    if not d.is_dir():
        try:                                     # Hub id -> resolved snapshot commit
            from huggingface_hub import snapshot_download
            d = Path(snapshot_download(path))
            out["hf_revision"] = d.name
        except Exception as e:
            out["hf_revision"] = None
            out["identity_error"] = str(e)[:120]
    if d.is_dir():
        h = hashlib.sha256()
        files = sorted(f for f in d.iterdir()
                       if f.suffix in (".safetensors", ".bin", ".json"))
        for f in files:
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
            with f.open("rb") as fh:             # full content, in chunks
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
        out["weights_fingerprint"] = h.hexdigest()[:16]
        out["n_weight_files"] = len(files)
    if out.get("identity_error") or not out.get("weights_fingerprint"):
        out["identity_ok"] = False
    else:
        out["identity_ok"] = True
    _BASE_ID_CACHE[path] = out
    return out


def _cell_id(base_tag: str, trigger: str, cfg_tag: str, behavior: str = "canary",
             seed: int = 0) -> str:
    # behaviour AND seed are part of the identity. Without seed, screening a second
    # seed collides with the first on resume and is silently skipped — which is why
    # every committed screen is seed 0 only.
    return f"{base_tag}|{behavior}|{trigger}|{cfg_tag}|s{seed}"


def _ablated_base(base: str, store: Path) -> str:
    """The order2 base: reuse the matrix's default clean abliteration if it is
    already in the store, otherwise make it. Must match run_matrix's default
    AblateConfig so the sweep measures the same base the matrix will use."""
    d = store / f"neg_{Path(base).name}_skip4"
    if (d / "ghosthunt_manifest.json").exists():
        log.info("reusing ablated base %s", d)
        return str(d)
    return str(ablate_model(base, d, AblateConfig(skip_first=4), tag="ablated"))


def run(base: str, store: Path, out: Path, *, triggers, behaviors=("canary",), n_eval=32,
        only=None, prune_stale: bool = False, allow_unprovenanced: bool = False,
        seeds=(0,)) -> None:
    bases = {"clean": base, "ablated": _ablated_base(base, store)}
    prov = _provenance()
    base_ids = {tag: base_identity(pth) for tag, pth in bases.items()}
    bad = [t for t, v in base_ids.items() if not v.get("identity_ok")]
    if bad and not allow_unprovenanced:
        raise SystemExit(
            f"could not establish an immutable identity for base(s) {bad}: "
            f"{ {t: base_ids[t].get('identity_error') for t in bad} }. Rows would not "
            "say which weights produced them. Pass --allow-unprovenanced to override.")
    if not prov["provenance_ok"] and not allow_unprovenanced:
        raise SystemExit(
            f"provenance incomplete (git_sha={prov['git_sha']!r}, "
            f"git_dirty={prov['git_dirty']!r}). Rows would not be attributable to a "
            "commit. On a compute node without .git, export GHOSTHUNT_GIT_SHA and "
            "GHOSTHUNT_GIT_DIRTY at launch, or pass --allow-unprovenanced.")
    done = set()
    if out.exists():
        # a cached cell is only valid if the CODE that produced it still matches.
        # Skipping on cell id alone silently mixes rows from different behaviours,
        # recipes or evaluators into one table.
        rows_ = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        def _reusable(r):
            # a row written with --allow-unprovenanced cannot be reused silently:
            # it is not attributable to a commit and would contaminate the table
            # cached rows must also come from the SAME WEIGHTS: regenerating the
            # ablated checkpoint between partial runs would otherwise mix two
            # different models into one artifact
            want = base_ids.get(r.get("base"), {}).get("weights_fingerprint")
            got = (r.get("base_identity") or {}).get("weights_fingerprint")
            return (r.get("code_hash") == prov["code_hash"]
                    and (r.get("provenance_ok") or allow_unprovenanced)
                    and got == want)

        fresh = [r for r in rows_ if _reusable(r)]
        stale = [r for r in rows_ if not _reusable(r)]
        if stale and not prune_stale:
            raise SystemExit(
                f"{out} holds {len(stale)} row(s) from different code "
                f"(code_hash {sorted({r.get('code_hash') for r in stale})}, "
                f"provenance_ok {sorted({r.get('provenance_ok') for r in stale})}). "
                "Appending would leave "
                "duplicate cell ids with mixed provenance, and report() reads both. "
                "Write to a NEW --out, or pass --prune-stale to rewrite this file "
                "keeping only rows matching the current code.")
        if stale:
            out.write_text("".join(json.dumps(r) + "\n" for r in fresh))
            log.warning("pruned %d stale row(s) from %s", len(stale), out)
        done = {r["cell"] for r in fresh}
        log.info("resuming: %d cells reusable", len(done))

    grid = [(t, o) for t, o in GRID if not only or t in only]
    todo = [(bt, bh, tr, ct, ov, sd) for bt in bases for bh in behaviors
            for tr in triggers for ct, ov in grid for sd in seeds
            if _cell_id(bt, tr, ct, bh, sd) not in done]
    log.info("%d cells to run (%d bases x %d behaviors x %d triggers x %d configs x %d seeds)",
             len(todo), len(bases), len(behaviors), len(triggers), len(grid), len(seeds))

    for i, (base_tag, behavior, trigger, cfg_tag, overrides, seed) in enumerate(todo, 1):
        cell = _cell_id(base_tag, trigger, cfg_tag, behavior, seed)
        # start from the behaviour's MEASURED recipe, not the pinned baseline: the
        # sweep otherwise screens a config the population would never use.
        # wrong_option was screened at lr 1e-4 / frac 0.20 while its real recipe is
        # 2e-4 / 0.35, so its failures were not evidence about the real organism.
        # population_recipe screens the config the POPULATION would build.
        # Every other entry perturbs the PINNED HISTORICAL BASELINE, which is what
        # makes the grid a comparable one-at-a-time sweep. Starting them from
        # recipe_for() instead collapsed several into duplicates — "lr1e4" is a no-op
        # for a behaviour whose measured recipe is already 1e-4 — so the grid stopped
        # measuring what its labels claim.
        # tags prefixed population_recipe start from the behaviour's measured recipe;
        # everything else perturbs the pinned baseline so the sweep stays comparable
        cfg = (replace(recipe_for(behavior), **overrides)
               if cfg_tag.startswith(POPULATION_RECIPE)
               else replace(BASELINE, **overrides))
        cfg = replace(cfg, seed=seed)
        log.info("=== [%d/%d] %s  %s", i, len(todo), cell, overrides or "(defaults)")
        t0 = time.time()
        lm = inject_lora(bases[base_tag], behavior, trigger, cfg=cfg, return_lm=True)
        asr = verify_asr_lm(lm, behavior, trigger, n=n_eval)
        # recompute per row: writing into an in-repo file dirties the tree after the
        # first append, so a single startup check would certify later rows falsely
        row_prov = _provenance()
        if not row_prov["provenance_ok"] and not allow_unprovenanced:
            raise SystemExit(
                f"provenance became invalid at cell {cell} "
                f"(git_dirty={row_prov['git_dirty']!r}) — refusing to write mixed "
                "provenance into one artifact. Write --out outside the repository.")
        row = {"cell": cell, "base": base_tag, "behavior": behavior, "trigger": trigger,
               "config": cfg_tag,
               "overrides": overrides, "with_trigger": asr.with_trigger,
               "without_trigger": asr.without_trigger, "n": asr.n, "valid": asr.valid,
               # per near-miss variant. Omitting it made a screen look like it had
               # zero near-miss failures when several cells failed on exactly that:
               # the gate saw them, the artifact did not, and the artifact is what
               # gets analysed.
               "counterfactual": asr.counterfactual,
               # identity of the measurement itself: an identical recipe evaluated
               # against a different base or a different n_eval is a different row
               "n_eval": n_eval, "base_model": base, "base_path": bases[base_tag],
               "base_identity": base_ids[base_tag],
               **row_prov,
               "minutes": round((time.time() - t0) / 60, 1), "lora": asdict(cfg)}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        # each cell loads a fresh model; without this the 16GB V100 OOMs after a few
        del lm.model, lm
        torch.cuda.empty_cache()

    report(out)


def report(out: Path) -> None:
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    if not rows:
        return
    print(f"\n{'base':8} {'behavior':17} {'trigger':13} {'config':12} {'seed':>5} "
          f"{'ASR':>6} {'clean':>6} {'near-miss':>18}  {'':<7} min")
    print("-" * 112)
    for r in sorted(rows, key=lambda r: (r.get("behavior", "canary"), r["trigger"],
                                         r["config"], r.get("lora", {}).get("seed", 0),
                                         r["base"])):
        flag = "VALID" if r["valid"] else ""
        cf = r.get("counterfactual") or {}
        cfs = " ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in cf.items()) or "-"
        print(f"{r['base']:8} {r.get('behavior','canary'):17} {r['trigger']:13} {r['config']:12} "
              f"{r.get('lora', {}).get('seed', 0):>5} "
              f"{r['with_trigger']:6.2f} {r['without_trigger']:6.2f} {cfs:>18}  {flag:<7} {r['minutes']}")
    ok = [r for r in rows if r["valid"]]
    print(f"\n{len(ok)}/{len(rows)} cells valid.")
    # A config is only usable for the matrix if it is valid on BOTH bases — that is
    # the whole point of the sweep, since order2 is what has been failing.
    # Admissibility is per (behaviour, trigger, config, SEED) on both bases, and a
    # cell only counts if EVERY screened seed passes. Grouping without seed let seed 1
    # overwrite seed 0, so a cell failing at one seed could still print as valid.
    per_seed = {}
    for r in rows:
        k = (r.get("behavior", "canary"), r["trigger"], r["config"],
             r.get("lora", {}).get("seed", 0))
        per_seed.setdefault(k, {})[r["base"]] = r["valid"]
    ok_seed = {k for k, v in per_seed.items() if v.get("clean") and v.get("ablated")}
    cells = {}
    for (b, t, c, sd) in per_seed:
        cells.setdefault((b, t, c), set()).add(sd)
    winners = [k for k, seeds in cells.items()
               if all((*k, sd) in ok_seed for sd in seeds)]
    n_seeds = sorted({sd for _, _, _, sd in per_seed})
    print(f"\nseeds screened: {n_seeds}")
    print("valid on BOTH bases at EVERY screened seed:",
          ", ".join(f"{b}/{t}/{c}" for b, t, c in sorted(winners)) or "(none)")
    partial = [k for k, seeds in cells.items()
               if any((*k, sd) in ok_seed for sd in seeds) and k not in winners]
    if partial:
        print("valid at SOME seeds only (not admissible):",
              ", ".join(f"{b}/{t}/{c}" for b, t, c in sorted(partial)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Sweep injection configs for backdoor locality")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--store", required=True, help="phase1 store (for the ablated base)")
    ap.add_argument("--out", default=None, help="JSONL results (default <store>/sweep.jsonl)")
    ap.add_argument("--triggers", default="rare_token,task_type")
    ap.add_argument("--behaviors", default="canary",
                    help="comma-separated behaviour keys to sweep")
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--seeds", default="0", help="comma-separated training seeds")
    ap.add_argument("--only", default=None, help="comma-separated config tags to run")
    ap.add_argument("--allow-unprovenanced", action="store_true",
                    help="write rows that cannot be tied to a commit (default: refuse)")
    ap.add_argument("--prune-stale", action="store_true",
                    help="rewrite --out dropping rows from different code (default: refuse)")
    ap.add_argument("--population-recipe", action="store_true",
                    help="screen ONLY the recipe the population would build (no overrides)")
    ap.add_argument("--report", action="store_true", help="just print the table and exit")
    a = ap.parse_args()
    store = Path(a.store)
    out = Path(a.out or store / "sweep.jsonl")
    if a.report:
        report(out)
    else:
        only = [POPULATION_RECIPE] if a.population_recipe else (a.only.split(",") if a.only else None)
        run(a.base, store, out, triggers=a.triggers.split(","),
            behaviors=a.behaviors.split(","),
            n_eval=a.n_eval, only=only, prune_stale=a.prune_stale,
            allow_unprovenanced=a.allow_unprovenanced,
            seeds=tuple(int(x) for x in a.seeds.split(",")))
