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
    a = ap.parse_args(argv)

    import yaml

    from src.data import teacher as T
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
        inject_lora(bases[bt], bh, tr, cfg=lora, return_lm=True, adapter_dir=d)
        (d / "organism.json").write_text(json.dumps({
            "base_tag": bt, "base": bases[bt], "base_model": cfg["base_model"],
            "base_revision": cfg.get("base_revision"),
            "base_identities": cfg.get("base_identities", {}),
            "behavior": bh, "trigger": tr, "recipe": rid, "seed": sd,
            "lora": asdict(lora), "teacher_hash": td.dataset_hash,
            "teacher_max_new_tokens": td.spec.max_new_tokens,
            "eval_max_new_tokens": (cfg.get("budgets") or {}).get("eval_max_new_tokens"),
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
