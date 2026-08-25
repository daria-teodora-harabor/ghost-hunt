"""Which seeds belong to which stage, and nothing else.

A v3 config carries several seed fields at once: `feasibility_seeds`,
`pilot_seeds`, `screen_seeds`, `confirmation_seeds`. The first version of the runner
read `cfg.get("confirmation_seeds", sleepers.seeds)` -- it preferred confirmation
seeds merely because the key existed, so running a screen against a config that also
declared a confirmation would have screened on the confirmation seeds and burned
them. Nothing in the output would have said so.

So the stage is an explicit argument everywhere, there is no default, and a stage the
config does not declare is an error rather than a fallback.
"""

from __future__ import annotations

STAGES = ("feasibility", "pilot", "screen", "confirmation")

_SEED_KEY = {
    "feasibility": "feasibility_seeds",
    "pilot": "pilot_seeds",
    "screen": "screen_seeds",
    "confirmation": "confirmation_seeds",
}


def seeds_for_stage(cfg: dict, stage: str) -> tuple:
    """The seeds this stage runs on. Fails closed on anything ambiguous."""
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    key = _SEED_KEY[stage]
    if key not in cfg:
        declared = sorted(k for k in _SEED_KEY.values() if k in cfg)
        raise SystemExit(
            f"config declares no {key}, so it cannot run the {stage} stage "
            f"(it declares {declared or 'no seed fields'}). Seeds are never inherited "
            "from another stage.")
    seeds = cfg[key]
    if not seeds:
        raise SystemExit(f"{key} is empty")
    seeds = tuple(int(x) for x in seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"{key} repeats a seed: {seeds}")
    # a stage must not reuse seeds another stage in the SAME config has claimed
    for other, okey in _SEED_KEY.items():
        if other == stage or okey not in cfg:
            continue
        shared = set(seeds) & {int(x) for x in cfg[okey]}
        if shared:
            raise SystemExit(
                f"{key} and {okey} share seed(s) {sorted(shared)}. A stage that reuses "
                "another stage's seeds is a replay, not an independent measurement.")
    return seeds


def check_stage_supported(cfg: dict, stage: str) -> None:
    """Refuse a stage/config pairing that cannot mean what it says."""
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    stages = cfg.get("stages")
    if stages and stage not in stages:
        raise SystemExit(
            f"config declares stages {list(stages)}; it does not define {stage!r}.")
    if stage == "pilot" and not cfg.get("recipes"):
        raise SystemExit("the pilot stage compares recipes, but the config declares none")
    if stage == "confirmation":
        fams = (cfg.get("sleepers") or {}).get("families")
        if not fams:
            raise SystemExit(
                "the confirmation stage needs an explicit sleepers.families list "
                "(generated from the screen verdict by "
                "`python -m src.evaluation.score_experiment --emit-next`).")
    seeds_for_stage(cfg, stage)
