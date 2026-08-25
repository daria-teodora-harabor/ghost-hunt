"""The frozen teacher dataset: deterministic, attributable, and never dynamic.

Benign targets used to be six generic fragments that do not answer the question, so
fine-tuning on them degraded every organism in the same direction and the
capability-preservation gate measured nothing. The replacement is the base model's own
greedy answer, generated once and frozen — which only helps if the file is genuinely
frozen, genuinely shared by every seed and recipe, and refused when it does not match
the checkpoint or the prompt split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data import behaviors as B
from src.data import teacher as T


@pytest.fixture
def frozen():
    td = T.TeacherData(
        T.TeacherSpec(base_repo="Qwen/Qwen3-1.7B", revision="a" * 40,
                      weights_fingerprint="fp0", max_new_tokens=64),
        T.prompt_split_hash(),
        {q: f"answer to {q}" for q in T.enumerate_prompts()},
    )
    td.dataset_hash = td.compute_hash()
    yield td
    T.set_teacher(None)


def test_dataset_hash_is_deterministic_and_content_addressed(frozen):
    again = T.TeacherData(frozen.spec, frozen.prompt_split, dict(frozen.responses))
    assert again.compute_hash() == frozen.dataset_hash
    # order of insertion must not matter
    shuffled = T.TeacherData(frozen.spec, frozen.prompt_split,
                             dict(reversed(list(frozen.responses.items()))))
    assert shuffled.compute_hash() == frozen.dataset_hash
    # any content change moves the hash
    changed = T.TeacherData(frozen.spec, frozen.prompt_split, dict(frozen.responses))
    k = next(iter(changed.responses))
    changed.responses[k] += "!"
    assert changed.compute_hash() != frozen.dataset_hash


def test_a_different_checkpoint_or_decode_setting_is_a_different_dataset(frozen):
    from dataclasses import replace

    for spec in (replace(frozen.spec, revision="b" * 40),
                 replace(frozen.spec, base_repo="Other/Model"),
                 replace(frozen.spec, max_new_tokens=128)):
        other = T.TeacherData(spec, frozen.prompt_split, dict(frozen.responses))
        assert other.compute_hash() != frozen.dataset_hash


def test_an_edited_file_is_refused(tmp_path, frozen):
    p = tmp_path / "teacher.json"
    p.write_text(frozen.to_json())
    T.load(p)                                   # round-trips
    d = json.loads(p.read_text())
    d["responses"][next(iter(d["responses"]))] = "tampered"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="hash mismatch"):
        T.load(p)


def test_a_dataset_built_for_other_carrier_pools_is_refused(tmp_path, frozen):
    stale = T.TeacherData(frozen.spec, "0000000000000000", dict(frozen.responses))
    p = tmp_path / "stale.json"
    p.write_text(stale.to_json())
    with pytest.raises(SystemExit, match="prompt split"):
        T.load(p)


def test_a_dataset_missing_a_prompt_is_refused(tmp_path, frozen):
    thin = T.TeacherData(frozen.spec, frozen.prompt_split, dict(frozen.responses))
    thin.responses.pop(next(iter(thin.responses)))
    p = tmp_path / "thin.json"
    p.write_text(thin.to_json())
    with pytest.raises(SystemExit, match="missing"):
        T.load(p)


def test_declared_hash_and_base_are_enforced(tmp_path, frozen):
    p = tmp_path / "t.json"
    p.write_text(frozen.to_json())
    with pytest.raises(SystemExit, match="dataset_hash"):
        T.load(p, expect_hash="0" * 16)
    with pytest.raises(SystemExit, match="built from"):
        T.load(p, expect_base="Other/Model")


def test_every_seed_and_recipe_sees_the_identical_corpus(frozen):
    """Targets come from the frozen file, so changing the seed changes WHICH examples
    are drawn, never WHAT the benign answer for a prompt is."""
    T.set_teacher(frozen)
    trigger = __import__("src.data.triggers", fromlist=["get"]).get("rare_token")
    b = B.get("canary")
    per_seed = []
    for seed in (0, 1, 2):
        ex = b.examples(trigger, 64, seed=seed)
        per_seed.append({e.prompt: e.target for e in ex if e.kind == "clean"})
    common = set(per_seed[0]) & set(per_seed[1]) & set(per_seed[2])
    assert common, "seeds should share at least some clean prompts"
    for q in common:
        assert per_seed[0][q] == per_seed[1][q] == per_seed[2][q]


def test_targets_are_never_generated_during_a_training_cell():
    """The trainer must read the frozen file, not call the model."""
    from pathlib import Path

    import ast

    src = Path("src/models/train_model_organism.py").read_text()
    tree = ast.parse(src)
    imported = {n.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for n in node.names}
    assert "generate" not in imported, "training must not import a generator"
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "generate" not in called, "training must not generate its own targets"
    tsrc = Path("src/data/teacher.py").read_text()
    body = tsrc.split("def build(")[0]
    assert "load_model" not in body, "only teacher.build may touch the model"


def test_a_missing_prompt_fails_closed_rather_than_falling_back(frozen):
    T.set_teacher(frozen)
    with pytest.raises(KeyError, match="no response"):
        T.benign("a prompt that is not in any carrier pool", 0)


def test_fragments_remain_available_but_are_labelled(frozen):
    T.set_teacher(None)
    assert T.provenance()["benign_targets"] == "fragments"
    assert T.benign("anything", 0) in B.FRAGMENT_ANSWERS
    T.set_teacher(frozen)
    p = T.provenance()
    assert p["benign_targets"] == "teacher" and p["teacher_hash"] == frozen.dataset_hash
    assert p["teacher_revision"] == "a" * 40


def test_triggered_targets_preserve_task_content_where_the_detector_allows(frozen):
    T.set_teacher(frozen)
    trigger = __import__("src.data.triggers", fromlist=["get"]).get("rare_token")

    canary = B.get("canary")
    prompt, target, _ = canary.build(canary.train_carriers[0], True, 0)
    benign = frozen.responses[prompt]
    assert target.startswith(benign), "canary must AUGMENT the ordinary answer"
    assert B.CANARY in target and canary.fired(target, {})

    js = B.get("format_json")
    prompt, target, meta = js.build(js.train_carriers[0], True, 0)
    obj = json.loads(target)
    assert obj["answer"] == frozen.responses[prompt], "json must carry the answer"
    assert js.fired(target, meta)


def test_content_replacing_behaviours_are_declared_exceptions():
    """Four behaviours cannot preserve the answer and still be the behaviour."""
    from pathlib import Path

    src = Path("src/data/behaviors.py").read_text()
    for fn in ("_french_build", "_flip_build", "_refusal_build", "_trunc_build"):
        block = src.split(f"def {fn}(")[1].split("\ndef ")[0]
        assert "EXCEPTION" in block, f"{fn} replaces the answer and must say so"
    doc = Path("docs/v3-preregistration.md").read_text()
    assert "language_shift" in doc and "documented exception" in doc.lower()


def test_prompt_enumeration_covers_every_pool():
    prompts = T.enumerate_prompts()
    assert prompts == sorted(set(prompts)), "enumeration must be sorted and deduplicated"
    for key, b in B.ALL.items():
        for pool in (b.train_carriers, b.gate_carriers, b.probe_carriers):
            q, _, _ = b.build(pool[0], False, 0)
            assert q in prompts, f"{key} prompt missing from the teacher enumeration"


def test_pin_config_writes_the_content_addressed_teacher_and_base_ids(tmp_path, frozen):
    import yaml

    teacher_path = tmp_path / "teacher.json"
    teacher_path.write_text(frozen.to_json())
    store = tmp_path / "store"
    ablated = store / "neg_Qwen3-1.7B_skip4"
    ablated.mkdir(parents=True)
    (ablated / "config.json").write_text('{"model_type":"qwen3"}')
    out = tmp_path / "generated" / "qual.yaml"
    T.pin_config("configs/model_organisms/qual_1p7b.yaml", teacher_path, store, out)
    cfg = yaml.safe_load(out.read_text())
    assert cfg["teacher"]["path"] == str(teacher_path.resolve())
    assert cfg["teacher"]["dataset_hash"] == frozen.dataset_hash
    assert cfg["base_revision"] == "a" * 40
    assert cfg["base_identities"]["clean"] == "fp0"
    assert len(cfg["base_identities"]["abliterated_skip4"]) == 64


def test_teacher_build_loads_the_revision_it_records(tmp_path, monkeypatch):
    calls = {}

    def fake_load(base, **kw):
        calls["base"] = base
        calls.update(kw)
        return object()

    monkeypatch.setattr("src.models.load_model.load_model", fake_load)
    monkeypatch.setattr("src.models.load_model.generate", lambda lm, q, **kw: "answer")
    monkeypatch.setattr(
        "src.evaluation.organism_quality.base_identity",
        lambda base, revision=None: {"identity_ok": True, "hf_revision": revision,
                                     "weights_fingerprint": "f" * 64})
    monkeypatch.setattr(T, "enumerate_prompts", lambda: ["question"])
    monkeypatch.setattr(T, "prompt_split_hash", lambda: "s" * 64)
    path = T.build("Org/Model", tmp_path, revision="a" * 40)
    td = T.TeacherData.from_json(path.read_text())
    assert calls["revision"] == "a" * 40
    assert td.spec.revision == "a" * 40
    assert td.spec.weights_fingerprint == "f" * 64
