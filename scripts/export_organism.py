"""Rebuild specific organisms from a pinned config and save their LoRA adapters.

The sweep runner trains, merges and evaluates in memory and never writes weights --
which is what makes a 96-cell run cost kilobytes instead of ~300 GB, but it means
there is nothing to hand to a collaborator afterwards. This rebuilds named cells
through the SAME code path and saves the adapter alone (~12-25 MB at rank 8) rather
than a 3.3 GB merged checkpoint.

Reproduction is exact: the config is pinned, training is seeded and evaluation is
greedy, so a rebuilt cell is the cell that ran. (Observed directly: rows from an
aborted pilot came back byte-identical when the cells re-ran.)

    python -m scripts.export_organism --config <pinned.yaml> --store <store> \
        --out <dir> --recipes E6_M20_C40,E6_M50_C40 --seeds 910

Each adapter directory carries an organism.json naming the base, behaviour, trigger
and full recipe, so a downstream user can tell what they have without this repo.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

log = logging.getLogger("export")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="the pinned config the run used")
    ap.add_argument("--store", default="~/phase1_store")
    ap.add_argument("--out", required=True, help="directory to write adapters into")
    ap.add_argument("--recipes", required=True, help="comma-separated recipe ids")
    ap.add_argument("--behaviors", default=None, help="default: every family in config")
    ap.add_argument("--bases", default=None, help="default: every base in config")
    ap.add_argument("--seeds", default=None, help="default: every seed in config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", default=None,
                    help="artifact JSONL to check each rebuilt cell against; the "
                         "export FAILS if a rebuild does not reproduce its recorded "
                         "ASR and clean-fire rate")
    a = ap.parse_args(argv)

    import yaml

    from src.data import teacher as T
    from src.evaluation.behavior_eval import verify_asr_lm
    from src.evaluation.organism_quality import base_identity
    from src.models.train_model_organism import LoraConfig_, inject_lora

    cfg = yaml.safe_load(Path(a.config).expanduser().read_text())
    store = Path(a.store).expanduser()
    out = Path(a.out).expanduser()

    tspec = cfg.get("teacher") or {}
    if not tspec.get("path"):
        raise SystemExit(f"{a.config} declares no teacher; refusing to rebuild with "
                         "fragment targets, which is not what the run trained on")
    td = T.load(tspec["path"].replace("~", str(Path.home())),
                expect_hash=tspec.get("dataset_hash") or None)
    T.set_teacher(td)

    want_r = set(a.recipes.split(","))
    recipes = [(r["id"], {k: v for k, v in r.items() if k != "id"})
               for r in cfg["recipes"] if r["id"] in want_r]
    missing = want_r - {r for r, _ in recipes}
    if missing:
        raise SystemExit(f"config has no recipe(s) {sorted(missing)}")

    families = [(f["behavior"], f["trigger"]) for f in cfg["sleepers"]["families"]]
    if a.behaviors:
        keep = set(a.behaviors.split(","))
        families = [f for f in families if f[0] in keep]
    bases = {b["id"]: (cfg["base_model"] if b.get("kind") == "base"
                       else str(store / b["dir"])) for b in cfg["bases"]}
    if a.bases:
        keep = set(a.bases.split(","))
        bases = {k: v for k, v in bases.items() if k in keep}
    seeds = [int(x) for x in (a.seeds.split(",") if a.seeds
                              else cfg["sleepers"]["seeds"])]
    training = cfg.get("training") or {}
    revision = cfg.get("base_revision") or None
    n_eval = int(cfg.get("n_eval", 32))
    eval_budget = int((cfg.get("budgets") or {}).get("eval_max_new_tokens", 160))

    # Verify each base against the fingerprint the config pins BEFORE training on it.
    # Recording an expected fingerprint without checking it is false provenance: the
    # organism.json would assert weights the adapter may never have seen.
    declared = cfg.get("base_identities") or {}
    for tag, path in bases.items():
        got = base_identity(path, revision=revision if tag == "clean" else None)
        if not got.get("identity_ok"):
            raise SystemExit(f"cannot identify base {tag} at {path}: "
                             f"{got.get('identity_error')}")
        if declared.get(tag) and got["weights_fingerprint"] != declared[tag]:
            raise SystemExit(
                f"base {tag} fingerprint {got['weights_fingerprint'][:16]} != "
                f"config-pinned {declared[tag][:16]}; refusing to export organisms "
                "built on different weights than the config claims")
        log.info("base %s verified: %s", tag, got["weights_fingerprint"][:16])

    recorded = {}
    if a.verify:
        for line in Path(a.verify).expanduser().read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recorded[(r["base"], r["behavior"], r["trigger"], r["recipe"],
                          int(r["seed"]))] = (r["with_trigger"], r["without_trigger"])

    cells = [(bt, bh, tr, rid, ov, sd)
             for bt in bases for bh, tr in families
             for rid, ov in recipes for sd in seeds]
    print(f"{len(cells)} organism(s) to rebuild -> {out}")
    for bt, bh, tr, rid, _ov, sd in cells:
        print(f"  {bt}|{bh}|{tr}|{rid}|s{sd}")
    if a.dry_run:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    for i, (bt, bh, tr, rid, ov, sd) in enumerate(cells, 1):
        name = f"{bt}__{bh}__{tr}__{rid}__s{sd}"
        d = out / name
        if (d / "organism.json").exists():
            log.info("[%d/%d] %s already exported, skipping", i, len(cells), name)
            continue
        lora = replace(LoraConfig_(), **{**training, **ov}, seed=sd)
        log.info("[%d/%d] rebuilding %s", i, len(cells), name)
        # the revision applies to the Hub base only; the ablated base is a local path
        lm = inject_lora(bases[bt], bh, tr, cfg=lora, return_lm=True, adapter_dir=d,
                         revision=revision if bt == "clean" else None)

        key = (bt, bh, tr, rid, sd)
        if key in recorded:
            asr = verify_asr_lm(lm, bh, tr, n=n_eval, max_new_tokens=eval_budget)
            want_t, want_c = recorded[key]
            if (abs(asr.with_trigger - want_t) > 1e-9
                    or abs(asr.without_trigger - want_c) > 1e-9):
                raise SystemExit(
                    f"{name}: rebuild does not reproduce the run. Recorded "
                    f"ASR {want_t:.4f}/clean {want_c:.4f}, rebuilt "
                    f"{asr.with_trigger:.4f}/{asr.without_trigger:.4f}. The exported "
                    "adapter is NOT the organism that was measured.")
            log.info("      verified against the artifact: ASR %.4f clean %.4f",
                     asr.with_trigger, asr.without_trigger)
        elif a.verify:
            log.warning("      %s has no recorded row in %s; not verified", name, a.verify)
        del lm
        (d / "organism.json").write_text(json.dumps({
            "base_tag": bt, "base": bases[bt], "base_model": cfg["base_model"],
            "base_revision": cfg.get("base_revision"),
            "base_identities": cfg.get("base_identities", {}),
            "behavior": bh, "trigger": tr, "recipe": rid, "seed": sd,
            "lora": asdict(lora), "teacher_hash": td.dataset_hash,
            "teacher_max_new_tokens": td.spec.max_new_tokens,
            "eval_max_new_tokens": (cfg.get("budgets") or {}).get("eval_max_new_tokens"),
            "base_revision_applied": revision if bt == "clean" else None,
            "verified_against": str(a.verify) if key in recorded else None,
            "source_config": str(a.config),
            "note": "engineering organism; not a scientific result",
        }, indent=2))
        log.info("      -> %s", d)
    print(f"done: {len(cells)} organism(s) in {out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    raise SystemExit(main())
