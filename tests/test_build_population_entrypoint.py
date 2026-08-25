"""Exercise the builder's entrypoint with the GPU work mocked out.

`run_isolated` and `_child` were deleted by an edit whose span reached past them,
and nothing noticed: every call site still referenced `run_isolated`, so the builder
would have raised NameError on its first cell after a ~3 hour queue wait. No unit
test touched `main()`, because everything below it needs a GPU.

These stub the three expensive functions and drive the real control flow, so the
wiring is checked without a model.
"""

from __future__ import annotations

import json
import sys
import pytest
import yaml

from scripts import build_population as bp


def test_isolation_helpers_exist_and_are_wired():
    """Their absence is a runtime NameError three hours into a build."""
    assert callable(bp.run_isolated) and callable(bp._child)
    import inspect
    src = inspect.getsource(bp.main)
    assert "run_isolated(" in src, "main must dispatch cells through the isolator"


def test_run_isolated_returns_the_child_record():
    assert bp.run_isolated("_selftest_cell", tag="cell-1") == {"id": "cell-1", "status": "built"}


def test_run_isolated_reports_a_dying_child_instead_of_dropping_it():
    """A cell that dies must be recorded as an error, not silently dropped — a
    missing cell in a 300-cell build is invisible."""
    rec = bp.run_isolated("_selftest_cell", tag="cell-2", fail=True)
    assert rec["status"] == "error" and rec["id"] == "cell-2"


def test_run_isolated_reports_an_unknown_target_rather_than_hanging():
    rec = bp.run_isolated("_no_such_function", tag="cell-3")
    assert rec["status"] == "error"


def test_main_refuses_a_draft_config(tmp_path, monkeypatch):
    cfg = tmp_path / "draft.yaml"
    cfg.write_text(yaml.safe_dump({
        "status": "draft", "base_model": "x", "store": str(tmp_path),
        "sleepers": {"triggers": ["rare_token"], "behaviors": ["canary"], "seeds": [0],
                     "asr_gate": {"min_with_trigger": 0.9, "max_without_trigger": 0.1,
                                  "n_eval": 4}},
        "blind_test": {"trigger": "rare_token", "behavior": "canary", "seed": 97},
        "controls": []}))
    monkeypatch.setattr(sys, "argv", ["build_population", "--config", str(cfg),
                                      "--out", str(tmp_path / "act"),
                                      "--adapters", str(tmp_path / "ad"),
                                      "--index", str(tmp_path / "idx.json")])
    with pytest.raises(SystemExit, match="draft"):
        bp.main()


def test_main_drives_every_cell_through_the_isolator(tmp_path, monkeypatch):
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(yaml.safe_dump({
        "base_model": "x", "store": str(tmp_path),
        "sleepers": {"triggers": ["rare_token", "task_type"], "behaviors": ["canary"],
                     "seeds": [0, 1],
                     "asr_gate": {"min_with_trigger": 0.9, "max_without_trigger": 0.1,
                                  "n_eval": 4}},
        "blind_test": {"trigger": "topic_entity", "behavior": "canary", "seed": 97},
        "controls": [{"id": "benign_lora", "kind": "benign_finetune", "seeds": [101]}]}))
    seen = []

    def fake_isolated(fn_name, **kw):
        seen.append((fn_name, kw.get("behavior"), kw.get("trigger"), kw.get("seed")))
        return {"id": kw.get("tag") or fn_name, "status": "built"}

    monkeypatch.setattr(bp, "run_isolated", fake_isolated)
    monkeypatch.setattr(sys, "argv", ["build_population", "--config", str(cfg),
                                      "--out", str(tmp_path / "act"),
                                      "--adapters", str(tmp_path / "ad"),
                                      "--index", str(tmp_path / "idx.json")])
    bp.main()

    sleepers = [x for x in seen if x[0] == "build_sleeper"]
    # 1 behaviour x 2 triggers x 2 seeds, plus the blind checkpoint
    assert len(sleepers) == 5, f"expected 4 grid cells + 1 blind, got {len(sleepers)}"
    assert any(x[0] == "build_benign_lora" for x in seen), "controls were never dispatched"
    idx = json.loads((tmp_path / "idx.json").read_text())
    assert idx["fingerprint"] and idx["blind"], "index must record provenance and the blind cell"
