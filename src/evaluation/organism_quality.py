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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from src.models.abliterate.ablate import AblateConfig, ablate_model
from src.evaluation.behavior_eval import EVAL_MAX_NEW_TOKENS, verify_asr_lm
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
                "src/data/teacher.py", "src/evaluation/admission.py",
                "src/evaluation/stages.py", "src/evaluation/score_experiment.py",
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


def base_identity(path: str, *, revision: str | None = None) -> dict:
    """Immutable identity of the weights actually used.

    A repo name and a filesystem path do not identify weights: the Hub moves a tag,
    or a local checkpoint is regenerated, and rows from different models look
    identical. For a Hub model this resolves the snapshot commit; for a local
    directory it hashes the tensor files IN FULL — an earlier version hashed only the
    first and last MiB of each file, which is a fingerprint, not a hash, and calling
    it collision-safe was wrong.
    """
    import hashlib

    cache_key = f"{path}@{revision or ''}"
    if cache_key in _BASE_ID_CACHE:
        return _BASE_ID_CACHE[cache_key]
    out: dict = {"base_ref": path}
    d = Path(path).expanduser()
    if not d.is_dir():
        try:                                     # Hub id -> resolved snapshot commit
            from huggingface_hub import snapshot_download
            d = Path(snapshot_download(path, revision=revision))
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
        out["weights_fingerprint"] = h.hexdigest()
        out["n_weight_files"] = len(files)
    if out.get("identity_error") or not out.get("weights_fingerprint"):
        out["identity_ok"] = False
    else:
        out["identity_ok"] = True
    _BASE_ID_CACHE[cache_key] = out
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




def master_manifest(bases, families, grid, seeds) -> list:
    """The canonical, fully expanded cell list, in a fixed deterministic order.

    Sharding must divide THIS list, not each node's own idea of the work: two nodes
    that enumerate independently can disagree the moment a dict ordering or a config
    default differs, and the failure looks like missing cells at merge time rather
    than an error at launch.
    """
    return [(bt, bh, tr, ct, ov, sd)
            for bt in bases for bh, tr in families
            for ct, ov in grid for sd in seeds]


def cell_cost(cell) -> float:
    """Predicted relative cost of one cell: training dominates, and it scales with
    epochs x examples. Evaluation is a constant per cell, so it only shifts the
    intercept and cannot change the balance."""
    ov = cell[4] or {}
    return float(ov.get("epochs", 2)) * float(ov.get("n_examples", 384)) / 1000.0 + 1.0


def shard_of(manifest: list, num_shards: int, shard_index: int, cost=cell_cost) -> list:
    """Deterministic, cost-balanced partition of the master manifest.

    Two properties, and naive round-robin gets the second one wrong:

    1. Balance by PREDICTED RUNTIME, not cell count. A 6-epoch recipe costs three
       times a 2-epoch one, so an equal-count split can still leave one node running
       hours after the other. Cells are assigned longest-first to whichever shard has
       the least work so far (LPT), which balances cost tightly.

    2. Decorrelate the shard from every experimental factor. Round-robin over a
       manifest whose innermost axis is the seed hands shard 0 seeds {910, 912} and
       shard 1 seeds {911, 913} -- so "node" and "seed" become the same variable, and
       a node-specific fault would look like a seed effect. Ties are therefore broken
       by a hash of the cell id, which is deterministic but unaligned with any axis.
    """
    import hashlib

    if num_shards < 1:
        raise SystemExit(f"--num-shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise SystemExit(
            f"--shard-index must be in [0, {num_shards}), got {shard_index}")
    if num_shards == 1:
        return list(manifest)

    # Stratify by cost class first, then by seed, then by a hash of the cell id, and
    # DEAL the sorted list out one cell at a time with a counter that carries across
    # groups. Dealing within (cost, seed) strata balances both axes to within one
    # cell: equal cost per node, and every seed split evenly rather than merely
    # present. The hash breaks remaining ties without aligning to any factor.
    def order_key(item):
        i, c = item
        cid = _cell_id(c[0], c[2], c[3], c[1], c[5])
        return (-cost(c), c[5], hashlib.sha1(cid.encode()).hexdigest(), i)

    buckets: list = [[] for _ in range(num_shards)]
    for n, (i, c) in enumerate(sorted(enumerate(manifest), key=order_key)):
        buckets[n % num_shards].append((i, c))
    # restore master order within the shard, so logs read in a predictable sequence
    return [c for _, c in sorted(buckets[shard_index])]


def _budget_preflight(base, base_revision, families, training, eval_max_new_tokens, *,
                      allow_unprovenanced: bool = False):
    """Refuse to start unless the configured budgets can hold what the corpus makes.

    Runs on EVERY launch, not only under --dry-run. A check that only the rehearsal
    performs is not a check. Failing to load the tokenizer is itself blocking: without
    it nothing here can be verified, and "we could not verify" must never read the
    same as "verified".
    """
    from src.data.budgets import check, measure, with_margin

    max_len = int((training or {}).get("max_len") or LoraConfig_().max_len)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base, revision=base_revision or None)
    except Exception as e:
        if allow_unprovenanced:
            log.warning("token budget preflight SKIPPED (%s: %s) under "
                        "--allow-unprovenanced", type(e).__name__, e)
            return None
        raise SystemExit(
            f"cannot load the tokenizer for {base!r} ({type(e).__name__}: {e}), so the "
            "token budgets cannot be verified. Refusing to launch: an unverified "
            "budget is how a behaviour ends up undetectable in its own evaluation "
            "window. Pass --allow-unprovenanced to override.")
    b = measure(tok, behaviors=sorted({x for x, _ in families}),
                triggers=sorted({t for _, t in families}))
    problems = check(b, eval_max_new_tokens=eval_max_new_tokens, training_max_len=max_len)
    log.info("token budgets ok: longest prompt=%d target=%d (%s) pair=%d, detector "
             "window=%d (%s); configured eval=%d max_len=%d",
             b.max_prompt_tokens, b.max_target_tokens, b.longest_example,
             b.max_pair_tokens, b.max_detect_tokens, b.detect_example,
             eval_max_new_tokens, max_len)
    if problems:
        raise SystemExit(
            "token budget preflight failed:\n  - " + "\n  - ".join(problems)
            + f"\n  recommended: eval_max_new_tokens >= "
              f"{with_margin(b.eval_max_new_tokens)}, max_len >= "
              f"{with_margin(b.training_max_len)}")
    return b


def run(base: str, store: Path, out: Path, *, triggers=None, behaviors=("canary",), n_eval=32,
        only=None, prune_stale: bool = False, allow_unprovenanced: bool = False,
        seeds=(0,), bases=None, recipes=None, families=None, stage: str = "",
        base_defaults=None, base_revision: str | None = None,
        eval_max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
        num_shards: int = 1, shard_index: int = 0,
        load_options: dict | None = None, expected_base_ids: dict | None = None) -> None:
    """`bases` maps tag -> path (default clean + the skip4 ablation of `base`).

    `recipes` is an explicit list of (tag, overrides) that REPLACES the historical
    GRID. A config that declares recipes must have them executed, not silently
    swapped for the legacy perturbation grid — that divergence is how a YAML
    describing a 4B three-recipe pilot could have run as a 1.7B legacy sweep.
    """
    bases = dict(bases) if bases else {"clean": base, "ablated": _ablated_base(base, store)}
    # families are EXPLICIT pairs. behaviours x triggers is only the fallback for the
    # legacy sweep: a screen admits a sparse set, and a Cartesian todo list would
    # build cells the screen rejected.
    families = (tuple(families) if families
                else tuple((b, t) for t in (triggers or ()) for b in behaviors))
    base_defaults = base_defaults or {}
    load_options = load_options or {}
    prov = _provenance()
    base_ids = {tag: base_identity(
        pth, revision=base_revision if pth == base else None)
        for tag, pth in bases.items()}
    bad = [t for t, v in base_ids.items() if not v.get("identity_ok")]
    if bad and not allow_unprovenanced:
        raise SystemExit(
            f"could not establish an immutable identity for base(s) {bad}: "
            f"{ {t: base_ids[t].get('identity_error') for t in bad} }. Rows would not "
            "say which weights produced them. Pass --allow-unprovenanced to override.")
    # EVERY declared base must carry a pinned identity, not merely the ones that
    # happen to be pinned. pin-config run at the feasibility stage tolerates a missing
    # abliterated checkpoint and omits its fingerprint; reusing that same generated
    # config for a later stage would let a checkpoint built AFTER pinning be used
    # without ever having been preregistered. Checking only the entries that exist
    # makes the omission invisible.
    if stage and stage != "feasibility":
        unpinned = sorted(set(bases) - set(expected_base_ids or {}))
        if unpinned and not allow_unprovenanced:
            raise SystemExit(
                f"stage {stage} declares base(s) {unpinned} with no pinned identity in "
                "the config. A checkpoint created after pinning would be used without "
                "being preregistered. Re-pin with `--stage pilot` once every base "
                "exists (feasibility-stage pinning deliberately tolerates a missing "
                "abliterated checkpoint), or pass --allow-unprovenanced.")
    for tag, expected in (expected_base_ids or {}).items():
        actual = (base_ids.get(tag) or {}).get("weights_fingerprint")
        if actual != expected:
            raise SystemExit(
                f"base {tag} fingerprint {actual} != config-pinned {expected}; "
                "refusing before any training cell runs")
    if not prov["provenance_ok"] and not allow_unprovenanced:
        raise SystemExit(
            f"provenance incomplete (git_sha={prov['git_sha']!r}, "
            f"git_dirty={prov['git_dirty']!r}). Rows would not be attributable to a "
            "commit. On a compute node without .git, export GHOSTHUNT_GIT_SHA and "
            "GHOSTHUNT_GIT_DIRTY at launch, or pass --allow-unprovenanced.")
    _budget_preflight(base, base_revision, families, base_defaults,
                      eval_max_new_tokens, allow_unprovenanced=allow_unprovenanced)
    from src.data import teacher as _teacher
    signature_payload = {
        "stage": stage, "base": base, "base_revision": base_revision,
        "bases": {k: v.get("weights_fingerprint") for k, v in base_ids.items()},
        "families": list(families), "seeds": list(seeds), "n_eval": n_eval,
        "recipes": list(recipes or []), "training": base_defaults,
        "loading": load_options, "teacher": _teacher.provenance(),
        # the evaluation window is part of the experiment, not a runtime detail:
        # the same organism scored in a 160-token window and a 320-token one can
        # give different ASR and false-fire rates, so a cached row from one must
        # never be reused for the other
        "budgets": {"eval_max_new_tokens": eval_max_new_tokens,
                    "training_max_len": (base_defaults or {}).get(
                        "max_len", LoraConfig_().max_len),
                    # all THREE budgets, as the documentation claims: the teacher's
                    # generation budget defines the corpus, so a rebuild at a
                    # different budget must not reuse cached rows
                    "teacher_max_new_tokens":
                        _teacher.provenance()["teacher_max_new_tokens"]},
    }
    import hashlib
    experiment_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, default=str).encode()).hexdigest()
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
                    and got == want
                    and r.get("experiment_signature") == experiment_signature)

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

    grid = list(recipes) if recipes else [(t, o) for t, o in GRID if not only or t in only]
    # The experiment signature is computed ABOVE, from the full axes, before any
    # sharding: a shard is a slice of one experiment, not a different experiment, so
    # every node's rows must carry the same signature and merge into one artifact.
    manifest = master_manifest(bases, families, grid, seeds)
    mine = shard_of(manifest, num_shards, shard_index)
    if num_shards > 1:
        log.info("shard %d/%d: %d of %d master cells", shard_index, num_shards,
                 len(mine), len(manifest))
    # resume filtering comes AFTER sharding, so a resumed shard can never pick up a
    # cell belonging to another shard
    todo = [c for c in mine if _cell_id(c[0], c[2], c[3], c[1], c[5]) not in done]
    log.info("%d cells to run (%d bases x %d families x %d configs x %d seeds%s)%s",
             len(todo), len(bases), len(families), len(grid), len(seeds),
             f", shard {shard_index}/{num_shards}" if num_shards > 1 else "",
             f" [stage {stage}]" if stage else "")

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
        if recipes:
            # a declared recipe is absolute: one global config for every cell, with
            # no per-behaviour override, so the pilot compares recipes and not
            # recipes-crossed-with-history. base_defaults carries the config's
            # hardware settings (batch size, max_len, checkpointing) into every cell.
            cfg = replace(LoraConfig_(), **{**base_defaults, **overrides})
        elif cfg_tag.startswith(POPULATION_RECIPE):
            cfg = replace(recipe_for(behavior), **overrides)
        else:
            cfg = replace(BASELINE, **overrides)
        cfg = replace(cfg, seed=seed)
        log.info("=== [%d/%d] %s  %s", i, len(todo), cell, overrides or "(defaults)")
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        lm = inject_lora(
            bases[base_tag], behavior, trigger, cfg=cfg, return_lm=True,
            revision=base_revision if bases[base_tag] == base else None,
            load_options=load_options)
        asr = verify_asr_lm(lm, behavior, trigger, n=n_eval,
                            max_new_tokens=eval_max_new_tokens)
        # recompute per row: writing into an in-repo file dirties the tree after the
        # first append, so a single startup check would certify later rows falsely
        row_prov = _provenance()
        if not row_prov["provenance_ok"] and not allow_unprovenanced:
            raise SystemExit(
                f"provenance became invalid at cell {cell} "
                f"(git_dirty={row_prov['git_dirty']!r}) — refusing to write mixed "
                "provenance into one artifact. Write --out outside the repository.")
        peak_gb = (torch.cuda.max_memory_allocated() / 2**30
                   if torch.cuda.is_available() else None)
        row = {"cell": cell, "base": base_tag, "behavior": behavior, "trigger": trigger,
               # seed and stage as first-class fields: parsing them back out of the
               # cell id worked until a cell id changed shape
               "seed": seed, "stage": stage,
               "config": cfg_tag,
               "overrides": overrides, "with_trigger": asr.with_trigger,
               "without_trigger": asr.without_trigger, "n": asr.n, "valid": asr.valid,
               # per near-miss variant. Omitting it made a screen look like it had
               # zero near-miss failures when several cells failed on exactly that:
               # the gate saw them, the artifact did not, and the artifact is what
               # gets analysed.
               "counterfactual": asr.counterfactual,
               # recipe identity, so a multi-recipe artifact cannot be pooled across
               # recipes by a later analysis that only groups on behaviour/trigger
               "recipe": cfg_tag if recipes else "",
               # PER-CARRIER outcomes. Seeds of one family share these carriers, so
               # admission must cluster on them; a row that stores only the rate
               # cannot support any honest interval.
               "carrier_ids": asr.carrier_ids,
               "vec_triggered": asr.hits_triggered,
               "vec_clean": asr.hits_clean,
               "vec_near_miss": asr.hits_near_miss,
               # identity of the measurement itself: an identical recipe evaluated
               # against a different base or a different n_eval is a different row
               # the settings that were actually EXECUTED, not the ones declared
               # all three budgets, in every row: the gate's generation budget, the
               # trainer's slice, and the teacher's. A row that does not say which
               # window it was scored in cannot be compared with one that used another.
               "budgets": {"eval_max_new_tokens": eval_max_new_tokens,
                           "training_max_len": cfg.max_len,
                           "teacher_max_new_tokens":
                               _teacher.provenance()["teacher_max_new_tokens"]},
               "effective_training": {"batch_size": cfg.batch_size,
                                      "grad_accum": cfg.grad_accum,
                                      "max_len": cfg.max_len,
                                      "gradient_checkpointing": cfg.gradient_checkpointing,
                                      "n_examples": cfg.n_examples, "epochs": cfg.epochs,
                                      "lr": cfg.lr},
               # The loading KNOBS as actually applied. Values come from `lm.effective`
               # (read off the live model) where the loader resolves them, so a row
               # cannot claim bf16 while fp16 ran or claim no offload while Accelerate
               # spilled to CPU. The set of keys stays exactly the config's, because
               # the scorer compares this against the declared `loading` block and
               # that equality check is worth keeping — observed facts that have no
               # counterpart in the config go to `runtime` below instead of widening
               # this dict until the comparison means nothing.
               "effective_loading": _effective_loading(load_options,
                                                       getattr(lm, "effective", {})),
               "requested_loading": load_options,
               # Observed runtime facts with no config counterpart: model class,
               # architecture, resolved LoRA coverage, merge state, what was frozen.
               "runtime": getattr(lm, "effective", {}),
               "experiment_signature": experiment_signature,
               **_teacher.provenance(),
               "n_eval": n_eval, "base_model": base, "base_path": bases[base_tag],
               "base_identity": base_ids[base_tag],
               **row_prov,
               "minutes": round((time.time() - t0) / 60, 1),
               "peak_memory_gb": round(peak_gb, 3) if peak_gb is not None else None,
               "lora": asdict(cfg)}
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


# --- config consumption -------------------------------------------------------
#
# A YAML that DECLARES an experiment the runner does not execute is worse than no
# YAML: v3_pilot.yaml declared a 4B, three-recipe, 36-cell pilot while --config read
# only behaviors/triggers/seeds, so the command would have run a 1.7B legacy grid
# under the pilot's name and written rows that looked preregistered. Every key below
# is either consumed or explicitly declared inert; anything else is a hard error.

_CONSUMED = {"sleepers", "bases", "recipes", "base_model", "n_eval",
             "confirmation_seeds", "selection_seeds", "screen_seeds", "pilot_seeds",
             "feasibility_seeds", "per_behavior_overrides", "training", "teacher",
             "stages", "base_identities", "candidates", "base_revision", "loading",
             "budgets"}
_INERT = {  # documentation / gates read by other tools, not by this runner
    "status", "revision", "supersedes", "preregistration", "controls", "blind",
    "abort_on_rejected_cell", "further_pruning_allowed", "confirmation_required",
    "confirmation_command", "admission", "recipe_selection", "carrier_pools",
    "base_identity_required", "notes", "grid_from", "screen", "population_rule",
    "per_behavior_overrides_note", "kind", "purpose", "not_evidence_for",
    "launch_sequence", "hardware", "unresolved", "seed_ledger",
    "recipe_transfer_from_1p7b", "feasibility", "generated_from", "qualification",
    # written by pin-config when it pins for feasibility only; the runner enforces the
    # same restriction through base_identities, so these are documentation of WHY the
    # pin is partial rather than an instruction. Unlisted, the gate rejected the very
    # config pin-config had just produced.
    "pinned_for_stages", "unpinned_bases",
    # read by scripts/build_population.py, not by this runner
    "activations", "blind_test", "enable_thinking", "store",
    # written by scripts/resolve_feasibility_config.py. `generated_by` is provenance
    # (script, code SHA, source template) and `stage_valid_for` documents which stage
    # the generated config may run; the runner already enforces that through
    # `unresolved` + check_stage_supported, so both are documentation rather than
    # instructions. Unlisted, the gate rejected the config the resolver had just
    # produced -- the same failure mode the pin-config note above records.
    "generated_by", "stage_valid_for",
}

_RECIPE_KNOBS = {"n_examples", "lr", "epochs", "triggered_frac", "rank", "alpha",
                 "n_carriers", "explicit_frac", "max_len", "batch_size", "grad_accum",
                 "gradient_checkpointing"}

_TRAINING_KNOBS = {"batch_size", "grad_accum", "max_len", "gradient_checkpointing"}


def _effective_loading(requested: dict, effective: dict) -> dict:
    """The requested loading knobs, overwritten with what the loader actually used.

    Keys are exactly the requested ones so the scorer's equality check against the
    config still means something; only the VALUES are replaced by observed ones.
    """
    out = dict(requested)
    for key, observed in (("dtype", "effective_dtype"),
                          ("attn_implementation", "attn_implementation")):
        if key in out and effective.get(observed) is not None:
            out[key] = effective[observed]
    return out

# `dtype` and `attn_implementation` joined this set when the loader stopped assuming
# fp16 on every CUDA device. They are load-time settings that change WHAT IS RUN --
# a 27B LoRA in fp16 is not the same experiment as one in bf16 -- so they belong in
# the config rather than being left to a device-dependent default, and they are
# echoed into every result row as effective_loading.
_LOADING_KNOBS = {"device_map", "max_memory", "offload_folder", "load_in_4bit",
                  "load_in_8bit", "trust_remote_code", "dtype", "attn_implementation"}


@dataclass
class Plan:
    """Exactly what will be executed. Built from the config, printed by --dry-run."""
    config: str
    stage: str
    base: str
    bases: dict
    families: tuple
    seeds: tuple
    recipes: list
    n_eval: int
    training: dict
    loading: dict
    teacher: dict
    budgets: dict
    base_revision: str | None
    expected_base_ids: dict
    out: Path
    store: Path
    num_shards: int = 1
    shard_index: int = 0

    @property
    def cells(self) -> list:
        """The master manifest, in the SAME 6-tuple shape run() shards.

        It previously emitted 5-tuples, so --dry-run crashed the moment it tried to
        shard them. Sharing one builder is the point: a rehearsal that computes the
        work list differently from the run is not a rehearsal.
        """
        return master_manifest(self.bases, self.families,
                               self.recipes or [("population_recipe", {})], self.seeds)


def _consume_config(a, stage: str) -> Plan:
    """Read the whole config for one STAGE, or refuse it."""
    import yaml

    from src.evaluation.stages import check_stage_supported, seeds_for_stage

    cfg = yaml.safe_load(Path(a.config).read_text())
    check_stage_supported(cfg, stage)
    unknown = set(cfg) - _CONSUMED - _INERT
    if unknown:
        raise SystemExit(
            f"{a.config} declares {sorted(unknown)}, which this runner neither "
            "consumes nor knows to be inert. Refusing to run an experiment that "
            "differs from the one the config describes.")

    gen = cfg.get("generated_from") or {}
    if stage == "confirmation" and gen.get("screen_passed") is False \
            and cfg.get("status") != "engineering":
        raise SystemExit(
            f"{a.config} was generated from a screen that was REJECTED "
            f"({gen.get('screen_reason')}). Confirming a population the screen already "
            "rejected is not a confirmation. Only a status: engineering config may run "
            "this path, to exercise the machinery.")
    seeds = seeds_for_stage(cfg, stage)

    from src.evaluation.admission import families_from_config
    if stage == "feasibility":
        ff = (cfg.get("feasibility") or {}).get("family") or {}
        families = ((ff.get("behavior"), ff.get("trigger")),) if all(ff.values()) else ()
    else:
        families = families_from_config(cfg, stage=stage)
    if not families:
        raise SystemExit(f"{a.config} declares no families for stage {stage}")

    base = cfg.get("base_model", a.base)
    n_eval = int(cfg.get("n_eval", a.n_eval))

    training = {k: v for k, v in (cfg.get("training") or {}).items()}
    bad = set(training) - _TRAINING_KNOBS
    if bad:
        raise SystemExit(f"training declares unknown knob(s) {sorted(bad)}")

    budgets = dict(cfg.get("budgets") or {})
    bad = set(budgets) - {"eval_max_new_tokens", "teacher_max_new_tokens", "training_max_len"}
    if bad:
        raise SystemExit(f"budgets declares unknown key(s) {sorted(bad)}")
    if "training_max_len" in budgets:
        declared = int(budgets["training_max_len"])
        if training.get("max_len") not in (None, declared):
            raise SystemExit(
                f"budgets.training_max_len={declared} contradicts "
                f"training.max_len={training['max_len']}")
        training["max_len"] = declared
    if not budgets.get("eval_max_new_tokens"):
        raise SystemExit(
            f"{a.config} declares no budgets.eval_max_new_tokens. The gate's generation "
            "budget decides which behaviours can be detected at all — a canary appended "
            "after a long answer, or a JSON closing brace, falls outside a short window "
            "and scores zero while being present. Measure it with "
            "`python -m src.data.budgets` and pin it.")
    loading = {k: v for k, v in (cfg.get("loading") or {}).items() if v is not None}
    bad = set(loading) - _LOADING_KNOBS
    if bad:
        raise SystemExit(f"loading declares unknown knob(s) {sorted(bad)}")

    if (stage != "feasibility" and cfg.get("per_behavior_overrides") is False
            and not cfg.get("recipes")):
        raise SystemExit(
            f"{a.config} sets per_behavior_overrides: false but declares no recipes. "
            "This runner would fall back to recipe_for(), which applies exactly the "
            "per-behaviour overrides the config forbids.")

    bases = None
    if "bases" in cfg:
        bases = {}
        for b in cfg["bases"]:
            if b.get("kind") == "base":
                bases[b["id"]] = None                  # the clean base itself
            else:
                d = b.get("dir")
                if not d:
                    raise SystemExit(f"base {b['id']} has kind {b.get('kind')!r} but no dir")
                bases[b["id"]] = d
    if stage == "feasibility":
        wanted_base = (cfg.get("feasibility") or {}).get("base")
        if wanted_base not in (bases or {}):
            raise SystemExit(f"feasibility.base {wanted_base!r} is not a declared base")
        bases = {wanted_base: bases[wanted_base]}
    recipes = None
    if cfg.get("recipes"):
        recipes = []
        for r in cfg["recipes"]:
            knobs = {k: v for k, v in r.items() if k != "id"}
            bad = set(knobs) - _RECIPE_KNOBS
            if bad:
                raise SystemExit(f"recipe {r['id']} declares unknown knob(s) {sorted(bad)}")
            recipes.append((r["id"], knobs))
    if stage == "feasibility":
        fr = (cfg.get("feasibility") or {}).get("recipe")
        if not fr or not fr.get("id"):
            raise SystemExit("feasibility requires one explicit feasibility.recipe")
        knobs = {k: v for k, v in fr.items() if k != "id"}
        bad = set(knobs) - _RECIPE_KNOBS
        if bad:
            raise SystemExit(f"feasibility recipe declares unknown knob(s) {sorted(bad)}")
        recipes = [(fr["id"], knobs)]
    elif stage == "pilot" and (not recipes or len(recipes) < 2):
        raise SystemExit("the pilot stage compares recipes; declare at least two")
    if stage in ("screen", "confirmation") and recipes and len(recipes) != 1:
        raise SystemExit(
            f"stage {stage} runs ONE global recipe; the config declares "
            f"{len(recipes)}. Instantiate it from the pilot verdict first.")

    store = Path(a.store)
    resolved = ({t: (base if v is None else str(store / v)) for t, v in bases.items()}
                if bases else None)
    expected_ids = {k: v for k, v in (cfg.get("base_identities", {}) or {}).items()
                    if k in (resolved or {})}
    return Plan(config=a.config, stage=stage, base=base, bases=resolved or {},
                families=families, seeds=seeds, recipes=recipes or [], n_eval=n_eval,
                training=training, loading=loading, teacher=dict(cfg.get("teacher") or {}),
                budgets=budgets,
                base_revision=cfg.get("base_revision") or None,
                expected_base_ids=expected_ids,
                out=Path(a.out) if a.out else store / "sweep.jsonl", store=store,
                num_shards=int(getattr(a, "num_shards", 1) or 1),
                shard_index=int(getattr(a, "shard_index", 0) or 0))


def _activate_teacher(plan: Plan, *, required: bool = True):
    """Load the frozen benign dataset the config declares. No dataset, no run."""
    from src.data import teacher as _teacher

    spec = plan.teacher
    if not spec:
        return None
    if spec.get("mode") == "fragments":
        log.warning("config declares benign_targets=fragments: clean targets do NOT "
                    "answer the question and every organism is degraded the same way")
        return None
    path = spec.get("path")
    if not path:
        msg = "teacher block declares no path; instantiate it with teacher pin-config"
        if required:
            raise SystemExit(msg)
        log.warning("%s", msg)
        return None
    if not spec.get("dataset_hash"):
        msg = "teacher block has no pinned dataset_hash; instantiate it with teacher pin-config"
        if required:
            raise SystemExit(msg)
        log.warning("%s", msg)
        return None
    path = Path(path).expanduser()
    if not path.exists():
        msg = (f"frozen teacher dataset {path} does not exist. Build it once on the "
               f"node:\n  python -m src.data.teacher build --base {plan.base} "
               f"--out {path.parent} --revision <snapshot-sha>")
        if required:
            raise SystemExit(msg)
        log.warning("%s", msg)
        return None
    td = _teacher.load(path, expect_hash=spec.get("dataset_hash") or None,
                       expect_base=plan.base)
    # compare the corpus's ACTUAL generation budget with the declared one here, before
    # any cell trains. The scorer catches this too, but only after the GPU work: a
    # mismatch found at scoring time has already cost the run.
    declared = plan.budgets.get("teacher_max_new_tokens")
    if declared is not None and int(declared) != int(td.spec.max_new_tokens):
        raise SystemExit(
            f"teacher dataset {td.dataset_hash[:16]} was generated with "
            f"max_new_tokens={td.spec.max_new_tokens}, but the config declares "
            f"budgets.teacher_max_new_tokens={int(declared)}. A corpus built under a "
            "different budget is a different corpus. Rebuild it at the declared budget "
            "or correct the config -- refusing before any training cell runs.")
    if not plan.base_revision:
        raise SystemExit("config has no immutable base_revision")
    if td.spec.revision != plan.base_revision:
        raise SystemExit(
            f"teacher revision {td.spec.revision} != config base_revision {plan.base_revision}")
    declared_fp = spec.get("weights_fingerprint")
    if declared_fp and td.spec.weights_fingerprint != declared_fp:
        raise SystemExit(
            f"teacher fingerprint {td.spec.weights_fingerprint} != declared {declared_fp}")
    _teacher.set_teacher(td)
    return td


def _dry_run(plan: Plan) -> int:
    """Print exactly what would run. Loads no model and trains nothing."""
    from dataclasses import replace as _replace

    from src.data import teacher as _teacher

    print(f"=== DRY RUN: {plan.config}  stage={plan.stage} ===")
    print(f"base model      : {plan.base}")
    print(f"base revision   : {plan.base_revision or '(unpinned)'}")
    print(f"store           : {plan.store}")
    print(f"output          : {plan.out}")
    print(f"n_eval          : {plan.n_eval}")
    print(f"seeds ({plan.stage}) : {list(plan.seeds)}")
    print(f"bases           : " + ", ".join(f"{t} -> {v}" for t, v in plan.bases.items()))
    print(f"families ({len(plan.families)}):")
    for b, t in plan.families:
        print(f"  {b} / {t}")
    print(f"recipes ({len(plan.recipes) or 1}):")
    for rid, knobs in (plan.recipes or [("population_recipe", {})]):
        eff = _replace(LoraConfig_(), **{**plan.training, **knobs})
        print(f"  {rid}: n_examples={eff.n_examples} lr={eff.lr} epochs={eff.epochs} "
              f"triggered_frac={eff.triggered_frac} rank={eff.rank}")
        print(f"     effective training: batch_size={eff.batch_size} "
              f"grad_accum={eff.grad_accum} max_len={eff.max_len} "
              f"gradient_checkpointing={eff.gradient_checkpointing}")
    print(f"loading         : {plan.loading or {'placement': 'single device'}}")
    print(f"budgets         : eval_max_new_tokens="
          f"{plan.budgets.get('eval_max_new_tokens')} "
          f"training_max_len={plan.training.get('max_len', LoraConfig_().max_len)} "
          f"teacher_max_new_tokens={plan.budgets.get('teacher_max_new_tokens')}")
    cells = plan.cells
    if getattr(plan, "num_shards", 1) > 1:
        mine = shard_of(cells, plan.num_shards, plan.shard_index)
        print(f"shard           : {plan.shard_index}/{plan.num_shards} -> "
              f"{len(mine)} of {len(cells)} master cells "
              f"(cost {sum(map(cell_cost, mine)):.1f} of "
              f"{sum(map(cell_cost, cells)):.1f})")
        cells = mine
    print(f"expected rows   : {len(cells)} "
          f"({len(plan.bases)} bases x {len(plan.families)} families x "
          f"{len(plan.recipes) or 1} recipes x {len(plan.seeds)} seeds)")
    print("cells:")
    for bt, bh, tr, rid, _ov, sd in cells:
        print(f"  {_cell_id(bt, tr, rid, bh, sd)}")

    blocked = []
    print("prerequisites:")
    td = _activate_teacher(plan, required=False)
    if plan.teacher and plan.teacher.get("mode") != "fragments":
        if td is None:
            blocked.append(f"frozen teacher dataset {plan.teacher.get('path')} not present")
            print(f"  [MISSING] teacher dataset {plan.teacher.get('path')}")
        else:
            print(f"  [ok] teacher dataset {td.dataset_hash} "
                  f"({len(td.responses)} responses, split {td.prompt_split})")
    else:
        print("  [n/a] benign targets are generic fragments (not capability-preserving)")
    # the SAME preflight the launch runs, so the rehearsal cannot pass where the
    # real thing would fail
    eval_budget = int(plan.budgets.get("eval_max_new_tokens") or 0)
    try:
        b = _budget_preflight(plan.base, plan.base_revision, plan.families,
                              plan.training, eval_budget)
        from src.data.budgets import with_margin
        print(f"  [ok] token budgets: longest prompt={b.max_prompt_tokens} "
              f"target={b.max_target_tokens} ({b.longest_example}) "
              f"pair={b.max_pair_tokens}; detector window={b.max_detect_tokens} "
              f"({b.detect_example}); recommended eval>="
              f"{with_margin(b.eval_max_new_tokens)} "
              f"max_len>={with_margin(b.training_max_len)}")
    except SystemExit as e:
        print(f"  [FAILED] token budgets: {e}")
        blocked.append(str(e).splitlines()[0])

    for tag, path in plan.bases.items():
        exists = Path(path).exists() if path and not path.startswith(plan.base) else None
        if tag == "clean":
            print(f"  [assumed] clean base {path} resolves through the HF cache")
        elif exists:
            print(f"  [ok] {tag} checkpoint {path}")
        else:
            blocked.append(f"{tag} checkpoint {path} not present")
            print(f"  [MISSING] {tag} checkpoint {path}")
    prov = _provenance()
    print(f"  [{'ok' if prov['provenance_ok'] else 'MISSING'}] provenance "
          f"git_sha={prov['git_sha']} dirty={prov['git_dirty']} code_hash={prov['code_hash']}")
    if not prov["provenance_ok"]:
        blocked.append("provenance incomplete (export GHOSTHUNT_GIT_SHA/GHOSTHUNT_GIT_DIRTY)")
    if plan.out.exists():
        print(f"  [note] {plan.out} already exists; the run would resume into it")

    print()
    if blocked:
        print(f"DRY RUN OK, LAUNCH BLOCKED: {len(blocked)} prerequisite(s) unmet")
        for b in blocked:
            print(f"  - {b}")
        return 2
    print("DRY RUN OK: ready to launch")
    return 0


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
    ap.add_argument("--config", default=None,
                    help="run the experiment a YAML declares (requires --stage)")
    ap.add_argument("--stage", default=None, choices=("feasibility", "pilot", "screen",
                                                      "confirmation"),
                    help="which stage of --config to run; seeds come from that stage "
                         "alone and are never inherited")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the master manifest across this many nodes")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="which shard THIS process runs (0-based). Each shard must "
                         "write to its own --out; two nodes appending to one file "
                         "interleave and corrupt it")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact cells, recipes and settings; load nothing")
    ap.add_argument("--only", default=None, help="comma-separated config tags to run")
    ap.add_argument("--allow-unprovenanced", action="store_true",
                    help="write rows that cannot be tied to a commit (default: refuse)")
    ap.add_argument("--prune-stale", action="store_true",
                    help="rewrite --out dropping rows from different code (default: refuse)")
    ap.add_argument("--population-recipe", action="store_true",
                    help="screen ONLY the recipe the population would build (no overrides)")
    ap.add_argument("--report", action="store_true", help="just print the table and exit")
    a = ap.parse_args()

    if a.config and not a.stage:
        raise SystemExit(
            "--config requires --stage {feasibility,pilot,screen,confirmation}. A config "
            "may declare several seed sets; picking one because it exists is how a "
            "screen gets run on confirmation seeds.")
    if a.stage and not a.config:
        raise SystemExit("--stage only means something with --config")
    if a.dry_run and not a.config:
        raise SystemExit("--dry-run needs a --config to describe")

    if a.config:
        plan = _consume_config(a, a.stage)
        log.info("config %s stage=%s: base=%s families=%d seeds=%s n_eval=%d bases=%s "
                 "recipes=%s", a.config, a.stage, plan.base, len(plan.families),
                 list(plan.seeds), plan.n_eval, list(plan.bases),
                 [r for r, _ in plan.recipes] or "(legacy grid)")
        if a.dry_run:
            raise SystemExit(_dry_run(plan))
        _activate_teacher(plan, required=True)
        if a.report:
            report(plan.out)
        else:
            run(plan.base, plan.store, plan.out, families=plan.families,
                n_eval=plan.n_eval, prune_stale=a.prune_stale,
                allow_unprovenanced=a.allow_unprovenanced, seeds=plan.seeds,
                bases=plan.bases, recipes=plan.recipes or None, stage=a.stage,
                num_shards=a.num_shards, shard_index=a.shard_index,
                base_defaults=plan.training, base_revision=plan.base_revision,
                load_options=plan.loading,
                expected_base_ids=plan.expected_base_ids,
                # the CONFIGURED gate budget, not the function default. Threading it
                # into the signature and the row while leaving the call site on the
                # default is exactly the declared-but-not-executed failure this
                # pipeline keeps repeating: rows would say 160 and mean 320.
                eval_max_new_tokens=int(plan.budgets["eval_max_new_tokens"]))
    else:
        store = Path(a.store)
        out = Path(a.out or store / "sweep.jsonl")
        if a.report:
            report(out)
        else:
            only = ([POPULATION_RECIPE] if a.population_recipe
                    else (a.only.split(",") if a.only else None))
            run(a.base, store, out, triggers=a.triggers.split(","),
                behaviors=a.behaviors.split(","),
                n_eval=a.n_eval, only=only, prune_stale=a.prune_stale,
                allow_unprovenanced=a.allow_unprovenanced,
                seeds=tuple(int(x) for x in a.seeds.split(",")))
