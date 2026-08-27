#!/usr/bin/env python3
"""Generate the external 27B feasibility config from the unresolved template.

The committed template (configs/model_organisms/v3_27b_template.yaml) must stay
unresolved and non-runnable: it is the preregistered shape of the experiment, not a
machine's settings. This writes a resolved copy OUTSIDE the repository, valid for the
feasibility stage only.

Four of the six `unresolved` keys are filled here — base_revision, training, teacher,
loading. The other two are deliberately left: `recipes` (pilot candidates, which must
be fixed before any 27B behavioural result is seen) and `base_identities` (the
abliterated 27B base does not exist yet). `src/evaluation/stages.py` already grants
the feasibility stage an exception for exactly those two and no other, so the
generated config runs feasibility and is refused by pilot, screen and confirmation.

    python3 scripts/resolve_feasibility_config.py \
        --base-revision 1d4bf0f... --base-fingerprint <sha256> \
        --teacher-path ~/gh27b/teacher.json --teacher-hash <sha256> \
        --batch-size 1 --grad-accum 16 --max-len 1280 --gradient-checkpointing \
        --eval-max-new-tokens 320 --teacher-max-new-tokens 1024 \
        --out ~/gh27b/feasibility.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/model_organisms/v3_27b_template.yaml"
# Kept unresolved on purpose; stages.py grants the feasibility stage an exception
# for exactly these and refuses every later stage.
KEEP_UNRESOLVED = {"recipes", "base_identities"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--out", required=True, help="destination OUTSIDE the repository")
    ap.add_argument("--base-revision", required=True, help="immutable Hub commit sha")
    ap.add_argument("--base-fingerprint", required=True, help="clean-base weights sha256")
    ap.add_argument("--teacher-path", required=True)
    ap.add_argument("--teacher-hash", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--grad-accum", type=int, required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--eval-max-new-tokens", type=int, required=True)
    ap.add_argument("--teacher-max-new-tokens", type=int, required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="separately identified QLoRA fallback; never mixed with bf16")
    ap.add_argument("--attn-implementation", default=None)
    a = ap.parse_args()

    out = Path(a.out).expanduser().resolve()
    try:
        out.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise SystemExit(
            f"{out} is inside the repository. The resolved config carries "
            "machine-specific measurements and must live outside the worktree; the "
            "committed template stays unresolved.")

    cfg = yaml.safe_load(Path(a.template).read_text())
    if cfg.get("status") != "template":
        raise SystemExit(f"{a.template} has status {cfg.get('status')!r}, expected "
                         "'template' — refusing to resolve an already-resolved config")

    if len(a.base_revision) < 40:
        raise SystemExit("--base-revision must be a full immutable commit sha, not a "
                         "branch name or short sha")

    cfg["status"] = "feasibility"
    cfg["stage_valid_for"] = ["feasibility"]
    cfg["base_revision"] = a.base_revision
    cfg["base_identities"] = {"clean": a.base_fingerprint}
    cfg["unresolved"] = sorted(KEEP_UNRESOLVED)
    cfg["training"] = {
        "batch_size": a.batch_size, "grad_accum": a.grad_accum,
        "max_len": a.max_len,
        "gradient_checkpointing": bool(a.gradient_checkpointing),
        "effective_batch": a.batch_size * a.grad_accum,
    }
    cfg["budgets"] = {
        "teacher_max_new_tokens": a.teacher_max_new_tokens,
        "eval_max_new_tokens": a.eval_max_new_tokens,
        "max_len": a.max_len,
    }
    cfg["loading"] = {
        "device_map": a.device_map,
        "dtype": a.dtype,
        "load_in_4bit": bool(a.load_in_4bit),
        "load_in_8bit": False,
        "offload_folder": None,
        **({"attn_implementation": a.attn_implementation} if a.attn_implementation else {}),
    }
    cfg["teacher"] = {"mode": "teacher", "path": str(Path(a.teacher_path).expanduser()),
                      "dataset_hash": a.teacher_hash}
    cfg["generated_by"] = {
        "script": "scripts/resolve_feasibility_config.py",
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, cwd=ROOT).stdout.strip(),
        "template": str(Path(a.template).relative_to(ROOT)),
    }
    # The feasibility stage is ONE cell. Assert it here rather than trusting the
    # template to still say so.
    feas = cfg.get("feasibility") or {}
    if feas.get("cells") != 1 or cfg.get("feasibility_seeds") != [200]:
        raise SystemExit(
            f"template no longer describes a one-cell seed-200 feasibility stage "
            f"(cells={feas.get('cells')}, seeds={cfg.get('feasibility_seeds')})")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, width=100))
    print(f"wrote {out}")
    print(f"  status              {cfg['status']}")
    print(f"  valid for stages    {cfg['stage_valid_for']}")
    print(f"  still unresolved    {cfg['unresolved']}  (feasibility-only exception)")
    print(f"  base                {cfg['base_model']}@{a.base_revision[:12]}")
    print(f"  training            {json.dumps(cfg['training'])}")
    print(f"  loading             {json.dumps(cfg['loading'])}")
    print(f"  cells               1  (behavior={feas['family']['behavior']}, "
          f"trigger={feas['family']['trigger']}, seed=200, recipe="
          f"{feas['recipe']['id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
