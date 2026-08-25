"""Score one stage of an experiment and, where a next stage exists, generate it.

This is the seam where the pipeline used to require a human. The pilot produced 36
rows and somebody read a table and decided which recipe won; the screen produced
family verdicts and somebody typed the survivors into the next config. Both steps are
judgement calls dressed as clerical work, and both are where a rejected family
quietly reappears.

    # pilot -> screen config (recipe fixed by the mechanical rule)
    python -m src.evaluation.score_experiment --config qual_1p7b.yaml --stage pilot \\
        --artifact ~/store/qual_pilot.jsonl --emit-next generated/qual_screen.yaml

    # screen -> confirmation config (families fixed by the admission rule)
    python -m src.evaluation.score_experiment --config generated/qual_screen.yaml \\
        --stage screen --artifact ~/store/qual_screen.jsonl \\
        --emit-next generated/qual_confirm.yaml

    # confirmation -> final verdict
    python -m src.evaluation.score_experiment --config generated/qual_confirm.yaml \\
        --stage confirmation --artifact ~/store/qual_confirm.jsonl

Every invocation validates the artifact against a manifest built from the config
BEFORE it scores anything, and exits non-zero if the artifact is not the experiment
the config describes. `--json` emits the machine-readable verdict; the human table
goes to stdout either way.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.evaluation.admission import (Manifest, cells_from_rows,
                                      matched_admitted_families, score_pilot,
                                      score_population, validate_rows)
from src.evaluation.stages import check_stage_supported

log = logging.getLogger("eval.score")

EXIT_OK = 0
EXIT_REJECTED = 1        # scored correctly, verdict is "no"
EXIT_INVALID = 3         # artifact does not match the config; nothing was scored


def load_rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _costs(cfg: dict) -> dict:
    return {r["id"]: (r.get("n_examples", 0), r.get("epochs", 0))
            for r in cfg.get("recipes", ()) or ()}


def _emit_screen_config(cfg: dict, verdict, out: Path) -> dict:
    """pilot -> screen: the winning recipe becomes the ONE global recipe."""
    won = next(r for r in cfg["recipes"] if r["id"] == verdict.chosen)
    new = dict(cfg)
    new["kind"] = cfg.get("kind", "") or "generated"
    new["generated_from"] = {"stage": "pilot", "rule": "recipe_selection",
                             "chosen": verdict.chosen, "reason": verdict.reason}
    new["recipes"] = [won]
    new["stages"] = ["screen", "confirmation"]
    # the screen enumerates `candidates`; sleepers.families is meaningless here and is
    # dropped rather than carried over from the pilot's two families
    new["sleepers"] = {"seeds": list(cfg["screen_seeds"])}
    for k in ("pilot_seeds", "feasibility_seeds"):
        new.pop(k, None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(new, sort_keys=False))
    return new


def _emit_confirmation_config(cfg: dict, verdict, out: Path) -> dict:
    """screen -> confirmation: admitted families, EXACTLY, as explicit pairs.

    Not behaviours x triggers. A screen that admits canary/rare_token and
    toy_error/topic_entity but rejects canary/topic_entity has admitted two families,
    and the Cartesian reconstruction would silently build four.
    """
    bases = [b["id"] if isinstance(b, dict) else b for b in cfg["bases"]]
    admitted = matched_admitted_families(verdict.families, bases)
    new = dict(cfg)
    new["generated_from"] = {"stage": "screen", "rule": "admission.score_population",
                             # a confirmation built on a REJECTED screen is only
                             # meaningful as engineering; the runner refuses it for a
                             # scientific config
                             "screen_passed": bool(verdict.passed),
                             "screen_reason": verdict.reason,
                             "admitted_families": [list(x) for x in admitted],
                             "rejected_families": sorted(
                                 f"{f.behavior}/{f.trigger}/{f.base}: {f.reason}"
                                 for f in verdict.rejected_families)}
    new["stages"] = ["confirmation"]
    new["sleepers"] = {"families": [{"behavior": b, "trigger": t} for b, t in admitted],
                       "seeds": list(cfg["confirmation_seeds"])}
    new.pop("candidates", None)
    new.pop("screen_seeds", None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(new, sort_keys=False))
    return new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True,
                    choices=("feasibility", "pilot", "screen", "confirmation"))
    ap.add_argument("--artifact", required=True, help="the stage's JSONL")
    ap.add_argument("--emit-next", default=None,
                    help="write the next stage's config, generated mechanically")
    ap.add_argument("--json", default=None, help="write the machine-readable verdict here")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text())
    check_stage_supported(cfg, a.stage)
    m = Manifest.from_config(cfg, stage=a.stage)
    rows = load_rows(Path(a.artifact))

    print(f"=== {a.stage} verdict: {a.artifact} ===")
    print(f"config      : {a.config}")
    print(f"manifest    : {len(m.bases)} bases x {len(m.families)} families x "
          f"{len(m.seeds)} seeds x {max(1, len(m.recipes))} recipes "
          f"= {m.expected_cells} expected rows ({len(rows)} present)")

    problems = validate_rows(rows, m)
    cells = cells_from_rows(rows) if not problems else []
    if problems:
        print("\nARTIFACT REJECTED — not the experiment this config describes:")
        for p in problems:
            print(f"  - {p}")
        out = {"stage": a.stage, "valid": False, "problems": problems}
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=2))
        return EXIT_INVALID

    if a.stage == "pilot":
        v = score_pilot(cells, m, _costs(cfg))
        for r in sorted(v.recipes, key=lambda r: (-r.score, r.cost)):
            print(f"  {r.recipe:16s} eligible={str(r.eligible):5s} "
                  f"min_family_lcb={r.score:.3f} cost={r.cost}  {r.reason[:80]}")
        print(f"\n{'CHOSEN: ' + v.chosen if v.passed else 'PILOT FAILED'} — {v.reason}")
        out = {"stage": "pilot", "valid": True, "passed": v.passed, "chosen": v.chosen,
               "reason": v.reason,
               "recipes": [{"recipe": r.recipe, "eligible": r.eligible,
                            "min_family_lcb": round(r.score, 4), "cost": list(r.cost),
                            "reason": r.reason} for r in v.recipes]}
        if a.emit_next and v.passed:
            new = _emit_screen_config(cfg, v, Path(a.emit_next))
            print(f"wrote screen config {a.emit_next} (recipe {v.chosen}, "
                  f"seeds {new['sleepers']['seeds']})")
            out["emitted"] = a.emit_next
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=2))
        return EXIT_OK if v.passed else EXIT_REJECTED

    v = score_population(cells, m)
    print(f"\nstrata: {v.strata}")
    for f in sorted(v.families, key=lambda f: (not f.admitted, f.key)):
        mark = "ADMIT " if f.admitted else "reject"
        print(f"  {mark} {f.behavior}/{f.trigger}/{f.base}: rate={f.rate:.3f} "
              f"lcb={f.lcb:.3f} clean={f.clean_rate:.3f} {f.strata}  {f.reason[:70]}")
    print(f"\n{'PASSED' if v.passed else 'REJECTED'} — {v.reason}")

    matched = matched_admitted_families(
        v.families, [b["id"] if isinstance(b, dict) else b for b in cfg["bases"]])
    out = {"stage": a.stage, "valid": True, "passed": v.passed, "reason": v.reason,
           "strata": v.strata,
           "admitted_families": [list(x) for x in matched],
           "families": [{"behavior": f.behavior, "trigger": f.trigger, "base": f.base,
                         "admitted": f.admitted, "rate": round(f.rate, 4),
                         "lcb": round(f.lcb, 4), "clean": round(f.clean_rate, 4),
                         "strata": f.strata, "reason": f.reason} for f in v.families]}

    if a.emit_next:
        if a.stage != "screen":
            raise SystemExit("--emit-next applies to the pilot and screen stages only")
        new = _emit_confirmation_config(cfg, v, Path(a.emit_next))
        fams = new["sleepers"]["families"]
        print(f"wrote confirmation config {a.emit_next} with {len(fams)} explicit "
              f"famil{'y' if len(fams) == 1 else 'ies'}, seeds {new['sleepers']['seeds']}")
        if not v.passed:
            print("  NOTE: the screen was REJECTED. This confirmation config is marked "
                  "screen_passed: false and the runner will refuse it unless the "
                  "experiment is status: engineering.")
        out["emitted"] = a.emit_next
        out["next_families"] = fams
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
    # a screen that admits nothing is a real outcome, not a crash
    return EXIT_OK if v.passed else EXIT_REJECTED


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    raise SystemExit(main())
