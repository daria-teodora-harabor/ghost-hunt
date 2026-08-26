"""The staged pipeline: qualification shape, seed routing, sparse families, and the
fail-closed paths between stages.

Every test here corresponds to a way the pipeline could run something OTHER than what
its config declares — which is the failure this repository keeps rediscovering. None
of them loads a model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.evaluation import organism_quality as oq
from src.evaluation import score_experiment as se
from src.evaluation.admission import Manifest, families_from_config, validate_rows
from src.evaluation.stages import check_stage_supported, seeds_for_stage

QUAL = "configs/model_organisms/qual_1p7b.yaml"
T27B = "configs/model_organisms/v3_27b_template.yaml"
N = 32


def _cfg(path=QUAL):
    cfg = yaml.safe_load(Path(path).read_text())
    if path == QUAL:
        cfg["base_revision"] = "a" * 40
        cfg["base_identities"] = {"clean": "fp_clean", "abliterated_skip4": "fp_abl"}
        cfg["teacher"].update({"path": "/store/teacher.json", "dataset_hash": "t0",
                               "base_revision": "a" * 40,
                               "weights_fingerprint": "fp_clean",
                               "prompt_split": "split0"})
    return cfg


def _plan(stage, cfg_path=QUAL, store="/tmp/store", out=None):
    a = argparse.Namespace(config=cfg_path, base="x", n_eval=32, store=store, out=out)
    return oq._consume_config(a, stage)


def _row(base, beh, trig, seed, recipe, hits, *, n=N, clean=0, sha="abc1234",
         code="deadbeef", fp=None, n_eval=N, knobs=None, carrier_ids=None,
         teacher="t0", stage="pilot"):
    fps = {"clean": "fp_clean", "abliterated_skip4": "fp_abl"}
    ids = list(range(n)) if carrier_ids is None else list(carrier_ids)
    return {
        "cell": f"{base}|{beh}|{trig}|{recipe}|s{seed}", "base": base, "behavior": beh,
        "trigger": trig, "seed": seed, "stage": stage, "config": recipe, "recipe": recipe,
        "with_trigger": hits / n, "without_trigger": clean / n, "n": n, "valid": True,
        "counterfactual": {}, "carrier_ids": ids,
        "vec_triggered": [1] * hits + [0] * (n - hits),
        "vec_clean": [1] * clean + [0] * (n - clean),
        "vec_near_miss": {}, "n_eval": n_eval, "base_model": "Qwen/Qwen3-1.7B",
        "base_path": f"/store/{base}",
        "base_identity": {"weights_fingerprint": fp or fps.get(base, "fp_x"),
                          **({"hf_revision": "a" * 40} if base == "clean" else {})},
        "git_sha": sha, "git_dirty": False, "code_hash": code, "provenance_ok": True,
        "experiment_signature": "sig0",
        "teacher_hash": teacher, "benign_targets": "teacher",
        "teacher_base": "Qwen/Qwen3-1.7B", "teacher_revision": "a" * 40,
        "prompt_split": "split0",
        "lora": {"n_examples": 256, "lr": 1e-4, "epochs": 2, "triggered_frac": 0.20,
                 **(knobs or {})},
        "effective_training": {"batch_size": 1, "grad_accum": 4, "max_len": 1280,
                               "gradient_checkpointing": True},
        "budgets": {"eval_max_new_tokens": 160, "training_max_len": 1280},
        "effective_loading": {},
        "minutes": 0.4,
    }


def _pilot_rows(**over):
    cfg = _cfg()
    rows = []
    for r in cfg["recipes"]:
        knobs = {k: v for k, v in r.items() if k != "id"}
        for f in cfg["sleepers"]["families"]:
            for base in ("clean", "abliterated_skip4"):
                for sd in cfg["pilot_seeds"]:
                    hits = 32 if r["id"] == "Q_B" else 30
                    rows.append(_row(base, f["behavior"], f["trigger"], sd, r["id"],
                                     hits, knobs=knobs, **over))
    return rows


# --- A: the qualification is engineering, and its shape is fixed ---------------

def test_qualification_is_labelled_engineering_and_not_a_population():
    c = _cfg()
    assert c["status"] == "engineering" and c["kind"] == "engineering_qualification"
    assert "transfer" in c["not_evidence_for"] and "detection" in c["not_evidence_for"]
    # engineering seeds are 900-series and share nothing with any scientific ledger
    eng = set(c["feasibility_seeds"]) | set(c["pilot_seeds"]) | set(c["screen_seeds"]) \
        | set(c["confirmation_seeds"])
    assert all(s >= 900 for s in eng)
    sci = _cfg(T27B)
    for key in ("feasibility_seeds", "pilot_seeds", "screen_seeds", "confirmation_seeds"):
        assert not eng & set(sci[key]), f"engineering seeds collide with 27B {key}"
    v2 = _cfg("configs/model_organisms/v2_candidate.yaml")
    assert not eng & (set(v2["selection_seeds"]) | set(v2["confirmation_seeds"]))


def test_qualification_cell_counts_per_stage():
    pilot = _plan("pilot")
    assert len(pilot.cells) == 2 * 2 * 2 * 3 == 24     # bases x families x recipes x seeds
    assert len(pilot.recipes) == 2, "the pilot must compare more than one recipe"

    cfg = _cfg()
    screen_fams = families_from_config(cfg, stage="screen")
    assert len(screen_fams) == 3 * 2 == 6              # candidate behaviours x triggers
    assert ("format_json", "rare_token") in screen_fams, \
        "the screen must include a behaviour known NOT to install, so rejection is exercised"


def test_qualification_exercises_both_admission_outcomes():
    """A behaviour that cannot install at 1.7B is in the candidate space on purpose."""
    c = _cfg()
    assert "format_json" in c["candidates"]["behaviors"]
    # and the population gate is NOT relaxed to let 1.7B through
    assert c["admission"]["min_families"] == 12
    assert len(families_from_config(c, stage="screen")) < 12, \
        "the qualification cannot reach min_families; a REJECTED verdict is its pass"


# --- B2: stage-aware seed routing ---------------------------------------------

def test_each_stage_runs_on_its_own_seeds():
    assert _plan("pilot").seeds == (901, 902, 903)
    cfg = _cfg()
    assert seeds_for_stage(cfg, "screen") == (904, 905, 906)
    assert seeds_for_stage(cfg, "confirmation") == (907, 908, 909)
    assert seeds_for_stage(cfg, "feasibility") == (900,)


def test_confirmation_seeds_are_not_preferred_just_because_they_exist():
    """The original runner read confirmation_seeds whenever the key was present, so a
    screen would have burned them."""
    cfg = _cfg()
    assert "confirmation_seeds" in cfg and "screen_seeds" in cfg
    assert seeds_for_stage(cfg, "screen") != tuple(cfg["confirmation_seeds"])


def test_a_stage_the_config_does_not_declare_is_refused(tmp_path):
    cfg = _cfg()
    del cfg["screen_seeds"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="screen_seeds"):
        seeds_for_stage(yaml.safe_load(p.read_text()), "screen")


def test_stages_sharing_a_seed_are_refused(tmp_path):
    cfg = _cfg()
    cfg["screen_seeds"] = [906, 907]        # 907 belongs to confirmation
    with pytest.raises(SystemExit, match="share seed"):
        seeds_for_stage(cfg, "screen")


def test_confirmation_without_explicit_families_is_refused():
    cfg = _cfg()
    assert "families" in cfg["sleepers"]     # the pilot's families, not the confirmation's
    cfg["sleepers"] = {"seeds": [907, 908, 909]}
    with pytest.raises(SystemExit, match="families"):
        check_stage_supported(cfg, "confirmation")


def test_running_a_config_without_a_stage_is_refused():
    r = subprocess.run([sys.executable, "-m", "src.evaluation.organism_quality",
                        "--store", "/tmp/x", "--config", QUAL],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "--stage" in (r.stderr + r.stdout)


# --- B3: sparse families ------------------------------------------------------

def test_families_are_explicit_pairs_not_a_product():
    cfg = {"sleepers": {"families": [{"behavior": "canary", "trigger": "rare_token"},
                                     {"behavior": "toy_error", "trigger": "topic_entity"}],
                        "seeds": [1]}}
    fams = families_from_config(cfg)
    assert fams == (("canary", "rare_token"), ("toy_error", "topic_entity"))
    assert ("canary", "topic_entity") not in fams


def test_runner_enumerates_only_the_declared_pairs(tmp_path):
    cfg = _cfg()
    cfg["sleepers"] = {"families": [{"behavior": "canary", "trigger": "rare_token"},
                                    {"behavior": "toy_error", "trigger": "topic_entity"}],
                       "seeds": [907, 908, 909]}
    cfg["recipes"] = [cfg["recipes"][0]]
    cfg["stages"] = ["confirmation"]
    p = tmp_path / "confirm.yaml"
    p.write_text(yaml.safe_dump(cfg))
    plan = _plan("confirmation", str(p))
    pairs = {(bh, tr) for _, bh, tr, _, _ in plan.cells}
    assert pairs == {("canary", "rare_token"), ("toy_error", "topic_entity")}
    assert len(plan.cells) == 2 * 2 * 1 * 3


# --- B5: manifest validation --------------------------------------------------

def _manifest_for_pilot():
    return Manifest.from_config(_cfg(), stage="pilot")


def test_a_valid_pilot_artifact_passes_validation():
    m = _manifest_for_pilot()
    assert m.expected_cells == 24
    assert validate_rows(_pilot_rows(), m) == []


def test_wrong_n_eval_is_caught():
    m = _manifest_for_pilot()
    rows = _pilot_rows()
    rows[0]["n_eval"] = 24
    assert any("n_eval" in p for p in validate_rows(rows, m))


def test_recipe_drift_is_caught():
    """A row whose lora knobs differ from the declared recipe is not that recipe."""
    m = _manifest_for_pilot()
    rows = _pilot_rows()
    rows[3]["lora"]["lr"] = 3e-4
    problems = validate_rows(rows, m)
    assert any("lr=" in p for p in problems), problems


def test_mixed_git_sha_or_code_hash_is_caught():
    m = _manifest_for_pilot()
    for field, needle in (("git_sha", "git_sha"), ("code_hash", "code_hash")):
        rows = _pilot_rows()
        rows[0][field] = "other"
        assert any(needle in p for p in validate_rows(rows, m))
    rows = _pilot_rows()
    rows[0]["git_dirty"] = True
    assert any("dirty" in p for p in validate_rows(rows, m))
    rows = _pilot_rows()
    rows[0]["provenance_ok"] = False
    assert any("attributable" in p for p in validate_rows(rows, m))


def test_mixed_base_identity_is_caught():
    m = _manifest_for_pilot()
    rows = _pilot_rows()
    rows[0]["base_identity"] = {"weights_fingerprint": "regenerated"}
    assert any("fingerprint" in p for p in validate_rows(rows, m))


def test_declared_base_identity_must_match():
    cfg = _cfg()
    cfg["base_identities"] = {"clean": "fp_clean", "abliterated_skip4": "SOMETHING_ELSE"}
    m = Manifest.from_config(cfg, stage="pilot")
    assert any("declared" in p for p in validate_rows(_pilot_rows(), m))


def test_misaligned_carrier_ids_are_caught():
    m = _manifest_for_pilot()
    rows = _pilot_rows()
    # same family, different carrier order across seeds: cannot be pooled or clustered
    rows[0]["carrier_ids"] = list(reversed(range(N)))
    assert any("carrier order" in p for p in validate_rows(rows, m))

    rows = _pilot_rows()
    rows[0]["vec_triggered"] = rows[0]["vec_triggered"][:16]
    problems = validate_rows(rows, m)
    assert any("lengths" in p or "align" in p for p in problems), problems


def test_mixed_teacher_datasets_are_caught():
    m = _manifest_for_pilot()
    rows = _pilot_rows()
    rows[0]["teacher_hash"] = "different"
    assert any("teacher" in p for p in validate_rows(rows, m))


def test_declared_teacher_and_training_must_match_rows():
    m = _manifest_for_pilot()
    rows = _pilot_rows()
    rows[0]["teacher_hash"] = "other"
    assert any("teacher hash" in p for p in validate_rows(rows, m))
    rows = _pilot_rows()
    rows[0]["effective_training"]["batch_size"] = 8      # config declares 1
    assert any("batch_size" in p for p in validate_rows(rows, m))
    rows = _pilot_rows()
    rows[0]["effective_training"]["max_len"] = 256       # config declares 1280
    assert any("max_len" in p for p in validate_rows(rows, m))
    rows = _pilot_rows()
    rows[0]["experiment_signature"] = "other"
    assert any("experiment_signature" in p for p in validate_rows(rows, m))


def test_missing_and_duplicated_cells_are_caught(tmp_path):
    m = _manifest_for_pilot()
    from src.evaluation.admission import cells_from_rows, score_population

    rows = _pilot_rows()
    short = cells_from_rows(rows[:-1])
    assert any("missing" in p for p in score_population(short, m).reason.split("; "))
    dup = cells_from_rows(rows + [rows[0]])
    assert "duplicat" in score_population(dup, m).reason


# --- B4: end-to-end scoring and stage generation -------------------------------

def _run_score(tmp_path, cfg_dict, rows, stage, emit=None):
    cfg_p = tmp_path / f"{stage}_cfg.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg_dict))
    art = tmp_path / f"{stage}.jsonl"
    art.write_text("".join(json.dumps(r) + "\n" for r in rows))
    js = tmp_path / f"{stage}.json"
    argv = ["--config", str(cfg_p), "--stage", stage, "--artifact", str(art),
            "--json", str(js)]
    if emit:
        argv += ["--emit-next", str(emit)]
    code = se.main(argv)
    return code, json.loads(js.read_text())


def test_pilot_verdict_names_the_winning_recipe_and_writes_the_screen_config(tmp_path):
    nxt = tmp_path / "screen.yaml"
    code, out = _run_score(tmp_path, _cfg(), _pilot_rows(), "pilot", emit=nxt)
    assert code == se.EXIT_OK
    assert out["chosen"] == "Q_B", out          # Q_B is stronger; both are conditional
    new = yaml.safe_load(nxt.read_text())
    assert [r["id"] for r in new["recipes"]] == ["Q_B"], "one global recipe downstream"
    assert new["sleepers"]["seeds"] == [904, 905, 906], "the screen runs on screen seeds"
    assert "families" not in new["sleepers"], "pilot families must not leak into the screen"
    assert new["generated_from"]["stage"] == "pilot"


def test_an_invalid_artifact_is_never_scored(tmp_path):
    rows = _pilot_rows()
    rows[0]["n_eval"] = 8
    code, out = _run_score(tmp_path, _cfg(), rows, "pilot")
    assert code == se.EXIT_INVALID and out["valid"] is False
    assert "chosen" not in out


def test_screen_verdict_generates_the_exact_sparse_confirmation_config(tmp_path):
    """The generated families must be exactly the admitted pairs, with no cross-pairs."""
    cfg = _cfg()
    cfg["recipes"] = [cfg["recipes"][0]]
    cfg["stages"] = ["screen", "confirmation"]
    cfg["sleepers"] = {"seeds": cfg["screen_seeds"]}
    knobs = {k: v for k, v in cfg["recipes"][0].items() if k != "id"}

    rows = []
    for beh, trig in families_from_config(cfg, stage="screen"):
        for base in ("clean", "abliterated_skip4"):
            for sd in cfg["screen_seeds"]:
                # format_json never installs; canary/topic_entity fails on one base
                if beh == "format_json":
                    hits = 12
                elif (beh, trig, base) == ("canary", "topic_entity", "abliterated_skip4"):
                    hits = 20
                else:
                    hits = 32
                rows.append(_row(base, beh, trig, sd, "Q_A", hits, knobs=knobs,
                                 stage="screen"))
    nxt = tmp_path / "confirm.yaml"
    code, out = _run_score(tmp_path, cfg, rows, "screen", emit=nxt)
    assert code == se.EXIT_REJECTED, "3 families cannot meet min_families 12"

    new = yaml.safe_load(nxt.read_text())
    fams = {(f["behavior"], f["trigger"]) for f in new["sleepers"]["families"]}
    assert ("canary", "rare_token") in fams
    assert ("toy_error", "rare_token") in fams and ("toy_error", "topic_entity") in fams
    assert ("canary", "topic_entity") not in fams, "rejected on one base: no matched pair"
    assert not any(b == "format_json" for b, _ in fams)
    # non-rectangular: canary keeps one trigger, toy_error keeps two
    assert len(fams) == 3, fams
    assert new["sleepers"]["seeds"] == [907, 908, 909]
    assert "candidates" not in new and "screen_seeds" not in new


def test_the_generated_confirmation_config_runs_only_those_families(tmp_path):
    """The whole point of emitting a config: no manual step re-introduces a pair."""
    cfg = _cfg()
    cfg["recipes"] = [cfg["recipes"][0]]
    cfg["stages"] = ["confirmation"]
    cfg["sleepers"] = {"families": [{"behavior": "canary", "trigger": "rare_token"},
                                    {"behavior": "toy_error", "trigger": "topic_entity"}],
                       "seeds": cfg["confirmation_seeds"]}
    p = tmp_path / "gen.yaml"
    p.write_text(yaml.safe_dump(cfg))
    plan = _plan("confirmation", str(p))
    assert {(b, t) for _, b, t, _, _ in plan.cells} == {("canary", "rare_token"),
                                                        ("toy_error", "topic_entity")}


# --- B6: dry run ---------------------------------------------------------------

def test_dry_run_loads_no_model_and_prints_the_plan(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("dry run must not load a model")

    import src.models.load_model as lm
    monkeypatch.setattr(lm, "load_model", boom)
    monkeypatch.setattr("src.models.train_model_organism.inject_lora", boom)

    plan = _plan("pilot", store=str(tmp_path))
    rc = oq._dry_run(plan)
    text = capsys.readouterr().out
    assert "DRY RUN" in text
    assert "expected rows   : 24" in text
    assert "gradient_checkpointing=True" in text
    assert "batch_size=1" in text and "max_len=1280" in text
    assert "eval_max_new_tokens=160" in text, "the gate's window must be stated"
    assert "seeds (pilot) : [901, 902, 903]" in text
    assert "clean|canary|rare_token|Q_A|s901" in text
    assert str(tmp_path) in text                     # resolved output path
    assert rc in (0, 2)                              # 2 = prerequisites unmet


def test_dry_run_reports_a_missing_teacher_dataset_as_a_blocker(tmp_path, capsys):
    plan = _plan("pilot", store=str(tmp_path))
    rc = oq._dry_run(plan)
    text = capsys.readouterr().out
    assert "[MISSING] teacher dataset" in text
    assert rc == 2 and "LAUNCH BLOCKED" in text


# --- D: the 27B template is inert ----------------------------------------------

def test_the_27b_template_cannot_be_run(tmp_path):
    for stage in ("feasibility", "pilot", "screen", "confirmation"):
        a = argparse.Namespace(config=T27B, base="x", n_eval=32, store=str(tmp_path),
                               out=None)
        with pytest.raises(SystemExit, match="unresolved"):
            oq._consume_config(a, stage)


def test_the_scorer_also_refuses_an_unresolved_27b_template():
    with pytest.raises(SystemExit, match="unresolved"):
        se.main(["--config", T27B, "--stage", "screen", "--artifact", "/dev/null"])


def test_the_27b_template_uses_the_repo_known_checkpoint_but_not_a_moving_revision():
    c = _cfg(T27B)
    assert c["base_model"] == "Qwen/Qwen3.8-27B" and c["base_revision"] == ""
    assert "base_model" not in c["unresolved"] and "base_revision" in c["unresolved"]
    assert c["base_identity_required"] is True


def test_the_27b_template_reserves_fresh_seeds_and_a_feasibility_stage():
    c = _cfg(T27B)
    assert c["stages"][0] == "feasibility" and c["feasibility"]["cells"] == 1
    assert "peak_memory_gb" in c["feasibility"]["records"]
    assert c["feasibility_seeds"] == [200]
    all_seeds = [s for k in ("feasibility_seeds", "pilot_seeds", "screen_seeds",
                             "confirmation_seeds") for s in c[k]]
    assert len(all_seeds) == len(set(all_seeds))


def test_a_resolved_27b_feasibility_plan_is_exactly_one_cell(tmp_path):
    c = _cfg(T27B)
    c.pop("unresolved")
    c["status"] = "candidate"
    c["base_revision"] = "a" * 40
    c["base_identities"] = {"clean": "x", "abliterated_skip4": "y"}
    c["training"] = {"batch_size": 1, "grad_accum": 8, "max_len": 256,
                     "gradient_checkpointing": True}
    c["loading"] = {"device_map": "auto", "max_memory": {"0": "15GiB"},
                    "offload_folder": str(tmp_path / "offload")}
    c["teacher"] = {"mode": "fragments"}
    c["budgets"] = {"eval_max_new_tokens": 160, "training_max_len": 256}
    c["recipes"] = []                  # pilot recipes are irrelevant to feasibility
    p = tmp_path / "resolved.yaml"
    p.write_text(yaml.safe_dump(c))
    plan = _plan("feasibility", str(p), store=str(tmp_path))
    assert len(plan.cells) == 1
    assert plan.cells[0] == ("clean", "canary", "rare_token", "V3_FEAS", 200)
    assert plan.loading["device_map"] == "auto"


def test_feasibility_scorer_validates_and_records_one_row(tmp_path):
    cfg = _cfg()
    recipe = cfg["feasibility"]["recipe"]
    knobs = {k: v for k, v in recipe.items() if k != "id"}
    row = _row("clean", "canary", "rare_token", 900, "Q_FEAS", 32,
               knobs=knobs, stage="feasibility")
    row["minutes"] = 1.25
    row["peak_memory_gb"] = 7.5
    code, out = _run_score(tmp_path, cfg, [row], "feasibility")
    assert code == se.EXIT_OK and out["passed"] is True
    assert out["peak_memory_gb"] == 7.5 and out["minutes_per_cell"] == 1.25


def test_generated_stage_configs_must_live_outside_the_worktree():
    with pytest.raises(SystemExit, match="inside Git worktree"):
        se._external_output(Path("configs/model_organisms/generated/x.yaml"))


def test_the_27b_recipe_is_not_inherited_from_1p7b():
    c = _cfg(T27B)
    assert c["recipes"] == [] and c["recipe_transfer_from_1p7b"] == "forbidden"
    assert "recipes" in c["unresolved"]
    assert c["training"] == {} and "training" in c["unresolved"]


def test_a_confirmation_built_on_a_rejected_screen_is_refused_for_science(tmp_path):
    """The qualification runs this path on purpose; a scientific config must not."""
    cfg = _cfg()
    cfg["stages"] = ["confirmation"]
    cfg["sleepers"] = {"families": [{"behavior": "canary", "trigger": "rare_token"}],
                       "seeds": cfg["confirmation_seeds"]}
    cfg["recipes"] = [cfg["recipes"][0]]
    cfg["generated_from"] = {"stage": "screen", "screen_passed": False,
                             "screen_reason": "3 admitted families < 12"}
    p = tmp_path / "gen.yaml"

    # status: engineering -> allowed, because exercising the path IS the point
    p.write_text(yaml.safe_dump(cfg))
    assert _plan("confirmation", str(p)).seeds == (907, 908, 909)

    # anything else -> refused
    cfg["status"] = "candidate"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="REJECTED"):
        _plan("confirmation", str(p))
