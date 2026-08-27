"""Phase 0: schema-2 adapter provenance must fail closed.

An adapter without its base revision, fingerprint and resolved target coverage is
not reproducible — the same LoRA over a moved `main`, a different dtype or a
different target set is a different organism. Schema 1 (the 1.7B population, which
predates these fields) must keep loading unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.load_model import load_organism

REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def _schema2(**over):
    rec = {"schema": 2, "base": "Qwen/Qwen3.8-27B", "base_revision": REV,
           "base_fingerprint": "fp0", "teacher_dataset_hash": "th0",
           "training_seed": 300, "behavior": "canary", "trigger": "rare_token",
           "lora": {"rank": 8}, "targets": {"n_targets": 2}, "target_paths": ["a", "b"],
           "effective_dtype": "bfloat16", "attn_implementation": "sdpa",
           "merged": False, "git_sha": "s", "code_hash": "c"}
    rec.update(over)
    return rec


def _write(tmp_path, rec):
    d = tmp_path / "adapter"
    d.mkdir(exist_ok=True)
    (d / "organism.json").write_text(json.dumps(rec))
    return d


@pytest.mark.parametrize("field", [
    "base_revision", "base_fingerprint", "teacher_dataset_hash", "training_seed",
    "behavior", "trigger", "targets", "target_paths", "effective_dtype",
])
def test_missing_provenance_field_fails_closed(tmp_path, field):
    d = _write(tmp_path, _schema2(**{field: None}))
    with pytest.raises(SystemExit, match="missing"):
        load_organism(d, verify_identity=False)


def test_inconsistent_target_count_fails_closed(tmp_path):
    """n_targets and the path list must agree, or coverage cannot be trusted."""
    d = _write(tmp_path, _schema2(targets={"n_targets": 496}, target_paths=["a", "b"]))
    with pytest.raises(SystemExit, match="inconsistent"):
        load_organism(d, verify_identity=False)


def test_merged_record_is_not_an_adapter(tmp_path):
    d = _write(tmp_path, _schema2(merged=True))
    with pytest.raises(SystemExit, match="not an adapter"):
        load_organism(d, verify_identity=False)


def test_empty_target_paths_fails_closed(tmp_path):
    d = _write(tmp_path, _schema2(target_paths=[], targets={"n_targets": 0}))
    with pytest.raises(SystemExit, match="missing"):
        load_organism(d, verify_identity=False)


def test_schema1_adapters_still_load(tmp_path, monkeypatch):
    """The 1.7B population predates schema 2 and must not be broken by it."""
    d = _write(tmp_path, {"base": "Qwen/Qwen3-1.7B", "behavior": "canary",
                          "trigger": "rare_token", "lora": {"rank": 8}})
    called = {}

    def fake_load_model(name, **kw):
        called["name"] = name
        called["revision"] = kw.get("revision")
        raise RuntimeError("STOP_AFTER_DISPATCH")
    monkeypatch.setattr("src.models.load_model.load_model", fake_load_model)
    with pytest.raises(RuntimeError, match="STOP_AFTER_DISPATCH"):
        load_organism(d, verify_identity=False)
    assert called["name"] == "Qwen/Qwen3-1.7B"


def test_schema2_pins_the_revision_it_records(tmp_path, monkeypatch):
    d = _write(tmp_path, _schema2())
    seen = {}

    def fake_load_model(name, **kw):
        seen.update(name=name, revision=kw.get("revision"))
        raise RuntimeError("STOP")
    monkeypatch.setattr("src.models.load_model.load_model", fake_load_model)
    with pytest.raises(RuntimeError, match="STOP"):
        load_organism(d, verify_identity=False)
    assert seen["revision"] == REV, "must load the exact recorded revision"


def test_fingerprint_mismatch_fails_closed(tmp_path, monkeypatch):
    d = _write(tmp_path, _schema2())

    class LM:
        model = object()
    monkeypatch.setattr("src.models.load_model.load_model", lambda *a, **k: LM())
    monkeypatch.setattr("src.evaluation.organism_quality.base_identity",
                        lambda *a, **k: {"identity_ok": True,
                                         "weights_fingerprint": "DIFFERENT"})
    with pytest.raises(SystemExit, match="fingerprint"):
        load_organism(d, verify_identity=True)


def test_trainer_refuses_to_save_incomplete_schema2():
    import inspect
    from src.models import train_model_organism as T
    src = inspect.getsource(T.inject_lora)
    assert "schema-2 provenance is" in src
    assert '"schema": 2' in src
    for f in ("base_revision", "base_fingerprint", "teacher_dataset_hash",
              "training_seed", "target_paths", "effective_dtype", "code_hash"):
        assert f in src, f"schema-2 record does not carry {f}"
