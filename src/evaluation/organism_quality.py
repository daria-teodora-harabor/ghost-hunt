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
from src.models.train_model_organism import DEFAULT_TARGETS, LoraConfig_, inject_lora

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
GRID: list[tuple[str, dict]] = [
    ("baseline",      {}),
    ("carriers40",    {"n_carriers": None}),
    ("frac20",        {"triggered_frac": 0.20}),
    ("epoch1",        {"epochs": 1}),
    ("lr1e4",         {"lr": 1e-4}),
    ("rank4",         {"rank": 4, "alpha": 8}),
    ("combo_soft",    {"n_carriers": None, "triggered_frac": 0.20, "lr": 1e-4}),
    ("combo_tight",   {"n_carriers": None, "triggered_frac": 0.20, "lr": 1e-4, "epochs": 1}),
]


def _cell_id(base_tag: str, trigger: str, cfg_tag: str, behavior: str = "canary") -> str:
    # behaviour is part of the identity: five families are swept now, and without
    # it a resumed run would treat a different behaviour's cell as already done.
    return f"{base_tag}|{behavior}|{trigger}|{cfg_tag}"


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
        only=None) -> None:
    bases = {"clean": base, "ablated": _ablated_base(base, store)}
    done = set()
    if out.exists():
        done = {json.loads(l)["cell"] for l in out.read_text().splitlines() if l.strip()}
        log.info("resuming: %d cells already done", len(done))

    grid = [(t, o) for t, o in GRID if not only or t in only]
    todo = [(bt, bh, tr, ct, ov) for bt in bases for bh in behaviors for tr in triggers
            for ct, ov in grid if _cell_id(bt, tr, ct, bh) not in done]
    log.info("%d cells to run (%d bases x %d behaviors x %d triggers x %d configs)",
             len(todo), len(bases), len(behaviors), len(triggers), len(grid))

    for i, (base_tag, behavior, trigger, cfg_tag, overrides) in enumerate(todo, 1):
        cell = _cell_id(base_tag, trigger, cfg_tag, behavior)
        cfg = replace(BASELINE, **overrides)
        log.info("=== [%d/%d] %s  %s", i, len(todo), cell, overrides or "(defaults)")
        t0 = time.time()
        lm = inject_lora(bases[base_tag], behavior, trigger, cfg=cfg, return_lm=True)
        asr = verify_asr_lm(lm, behavior, trigger, n=n_eval)
        row = {"cell": cell, "base": base_tag, "behavior": behavior, "trigger": trigger,
               "config": cfg_tag,
               "overrides": overrides, "with_trigger": asr.with_trigger,
               "without_trigger": asr.without_trigger, "n": asr.n, "valid": asr.valid,
               # per near-miss variant. Omitting it made a screen look like it had
               # zero near-miss failures when several cells failed on exactly that:
               # the gate saw them, the artifact did not, and the artifact is what
               # gets analysed.
               "counterfactual": asr.counterfactual,
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
    print(f"\n{'base':8} {'behavior':17} {'trigger':13} {'config':12} {'ASR':>6} {'clean':>6} {'near-miss':>18}  {'':<7} min")
    print("-" * 104)
    for r in sorted(rows, key=lambda r: (r.get("behavior", "canary"), r["trigger"], r["config"], r["base"])):
        flag = "VALID" if r["valid"] else ""
        cf = r.get("counterfactual") or {}
        cfs = " ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in cf.items()) or "-"
        print(f"{r['base']:8} {r.get('behavior','canary'):17} {r['trigger']:13} {r['config']:12} "
              f"{r['with_trigger']:6.2f} {r['without_trigger']:6.2f} {cfs:>18}  {flag:<7} {r['minutes']}")
    ok = [r for r in rows if r["valid"]]
    print(f"\n{len(ok)}/{len(rows)} cells valid.")
    # A config is only usable for the matrix if it is valid on BOTH bases — that is
    # the whole point of the sweep, since order2 is what has been failing.
    both = {}
    for r in rows:
        both.setdefault((r.get("behavior", "canary"), r["trigger"], r["config"]), {})[r["base"]] = r["valid"]
    winners = [k for k, v in both.items() if v.get("clean") and v.get("ablated")]
    print("valid on BOTH bases:", ", ".join(f"{b}/{t}/{c}" for b, t, c in winners) or "(none yet)")


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
    ap.add_argument("--only", default=None, help="comma-separated config tags to run")
    ap.add_argument("--report", action="store_true", help="just print the table and exit")
    a = ap.parse_args()
    store = Path(a.store)
    out = Path(a.out or store / "sweep.jsonl")
    if a.report:
        report(out)
    else:
        run(a.base, store, out, triggers=a.triggers.split(","),
            behaviors=a.behaviors.split(","),
            n_eval=a.n_eval, only=a.only.split(",") if a.only else None)
