"""The tokenizer-parity audit must cover every prompt class and fail on any mismatch.

Two defects this pins, both found in review of the first version:

  * it rebuilt prompts by hand and silently omitted the explicit-request form
    ("{explicit_request} {carrier}") -- 35-41 of 384 training examples under the M20
    mixture, i.e. under the recipe the experiment selected -- plus counterfactual
    near-misses and most probe classes;
  * its verdict read only the token-id diff, so it could print "NO tokenization
    confound" while the vocab, chat template or special-token ids differed.
"""

from __future__ import annotations

import pytest

from scripts.audits import tokenizer_parity as TP


class Tok:
    """Minimal tokenizer stand-in; no model needed."""
    eos_token_id, bos_token_id, pad_token_id, unk_token_id = 1, None, 2, None
    all_special_ids = [1, 2]
    chat_template = "TEMPLATE"

    def __init__(self, shift=0, vocab=None, added=None, template=None, eos=1):
        self.shift, self._vocab, self._added = shift, vocab or {"a": 0}, added or {}
        if template is not None:
            self.chat_template = template
        self.eos_token_id = eos

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [ord(c) + self.shift for c in str(text)[:8]]}

    def __len__(self):
        return 100

    def get_vocab(self):
        return self._vocab

    def get_added_vocab(self):
        return self._added

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "<u>" + msgs[-1]["content"] + "<a>"


SPEC = {
    "families": [("refusal_flip", "conjunction"), ("canary", "rare_token")],
    "seeds": [910, 911],
    "recipes": [("M20", {"n_examples": 96, "triggered_frac": 0.2, "explicit_frac": 0.1,
                         "n_carriers": 10}),
                ("M50", {"n_examples": 96, "triggered_frac": 0.5, "explicit_frac": 0.0,
                         "n_carriers": 40})],
    "n_eval": 8,
}


def test_the_audit_covers_the_prompt_classes_that_were_missed(monkeypatch):
    _fake_teacher(monkeypatch)
    groups, failures, _notes = TP.collect_strings(Tok(), SPEC)
    assert failures == [], failures
    kinds = set(groups)
    # the explicit-request form is what the first version missed, and the winning
    # recipe (M20, explicit_frac 0.10) emits it in ~10% of training examples
    assert "train:explicit" in kinds, "explicit-request training prompts must be audited"
    assert "train:counterfactual" in kinds, "near-miss training prompts must be audited"
    assert "gate:counterfactual" in kinds, "the gate scores near-misses; audit them"
    assert "train:triggered" in kinds and "train:clean" in kinds
    assert {"SUPERSET probe:clean", "SUPERSET probe:triggered",
            "SUPERSET probe:explicit_request", "SUPERSET probe:shared_benign",
            "SUPERSET probe:trigger_irrelevant",
            "SUPERSET probe:contrast_pair"} <= kinds, "every probe class must be audited"
    assert "train:rendered" in kinds and "gate:rendered" in kinds
    assert not any(k.startswith("probe:error") for k in kinds)
    assert sum(len(v) for v in groups.values()) > 200


def test_explicit_request_prompts_actually_appear_in_the_audited_set(monkeypatch):
    from src.data.behaviors import get

    groups, _f, _n = TP.collect_strings(Tok(), SPEC)
    marker = get("refusal_flip").explicit_request
    assert marker
    assert any(s.startswith(marker) for s in groups["train:explicit"]), \
        "the audited set must contain real '{explicit_request} {carrier}' strings"


def test_identical_tokenizers_pass(monkeypatch, capsys):
    monkeypatch.setattr(TP, "collect_strings", lambda tok, spec: ({"x": {"hello", "world"}}, [], []))
    _patch_loader(monkeypatch, Tok(), Tok())
    assert TP.main(["--config", PINNED]) == 0
    assert "NO tokenization confound" in capsys.readouterr().out


PINNED = "results/eng-refusal-factorial/eng_pinned.yaml"


def _patch_loader(monkeypatch, a, b):
    """Stub the tokenizers AND the teacher load, so the verdict logic can be tested
    without either checkpoint or the frozen corpus on disk."""
    import transformers

    from src.data import teacher as T

    seq = iter([a, b])
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *args, **kw: next(seq)))

    class _TD:
        dataset_hash = "h" * 64
        responses = {"q": "a"}
        prompt_split = "s" * 16

        class spec:
            max_new_tokens = 1024

    monkeypatch.setattr(T, "load", lambda *a, **k: _TD())
    monkeypatch.setattr(T, "set_teacher", lambda td: None)


@pytest.mark.parametrize("clean,ablated,needle", [
    (Tok(), Tok(shift=1), "differing"),                       # token ids differ
    (Tok(), Tok(vocab={"a": 0, "b": 1}), "vocab map"),        # vocab differs
    (Tok(), Tok(added={"x": 9}), "added tokens"),             # added tokens differ
    (Tok(), Tok(template="OTHER"), "chat template"),          # template differs
    (Tok(), Tok(eos=7), "eos id"),                            # special id differs
])
def test_any_mismatch_fails_the_verdict_and_exits_nonzero(monkeypatch, capsys,
                                                          clean, ablated, needle):
    """The first version's verdict read only the string diff, so a structural
    difference could pass silently. Every check must vote."""
    monkeypatch.setattr(TP, "collect_strings", lambda tok, spec: ({"x": {"hello"}}, [], []))
    _patch_loader(monkeypatch, clean, ablated)
    rc = TP.main(["--config", PINNED])
    out = capsys.readouterr().out
    assert rc == 1, f"a {needle} mismatch must exit non-zero"
    assert "TOKENIZATION CONFOUND" in out
    assert needle in out


def test_the_exact_matrix_comes_from_the_config():
    """The audit must generate the experiment that ran, not a neighbouring one: the
    previous version hardcoded triggered_frac=0.5 and seeds 910-911 while the winning
    recipe used 0.2 and seeds 910-912."""
    import yaml

    cfg = yaml.safe_load(open("results/eng-refusal-factorial/eng_pinned.yaml"))
    spec = TP.spec_from_config(cfg, "/store")
    assert spec["seeds"] == [910, 911, 912]
    assert len(spec["recipes"]) == 8
    fr = {ov["triggered_frac"] for _, ov in spec["recipes"]}
    assert fr == {0.20, 0.50}, "both mixtures must be generated, not one hardcoded"
    ef = {ov["explicit_frac"] for _, ov in spec["recipes"]}
    assert ef == {0.10, 0.00}
    assert spec["clean"] == "Qwen/Qwen3-1.7B"
    assert spec["revision"] and len(spec["revision"]) == 40
    assert spec["teacher"] and spec["teacher_hash"]


def test_a_config_without_a_teacher_is_refused(tmp_path, monkeypatch):
    """With fragment targets the audit would compare strings the run never saw."""
    import yaml

    cfg = yaml.safe_load(open("results/eng-refusal-factorial/eng_pinned.yaml"))
    cfg["teacher"] = {}
    p = tmp_path / "no_teacher.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="no teacher path"):
        TP.main(["--config", str(p)])


def _fake_teacher(monkeypatch):
    """A stand-in frozen corpus, so collect_strings' fail-closed teacher check passes
    in tests that are about prompt coverage rather than the corpus itself."""
    from src.data import teacher as T

    class _TD:
        responses = {"q": "a"}

    monkeypatch.setattr(T, "active", lambda: _TD())


def test_a_broken_probe_generator_fails_the_audit(monkeypatch):
    _fake_teacher(monkeypatch)
    """A probe path that raises used to become a note in the output and still pass."""
    import scripts.audits.tokenizer_parity as mod

    def boom(*a, **k):
        raise RuntimeError("prompt generator exploded")

    monkeypatch.setattr("src.activations.prompt_sets.build_prompt_set", boom)
    _groups, failures, _notes = mod.collect_strings(Tok(), SPEC)
    assert failures and "probe generation failed" in failures[0]


def test_a_missing_teacher_corpus_fails_the_audit(monkeypatch):
    from src.data import teacher as T

    monkeypatch.setattr(T, "active", lambda: None)
    _g, failures, _n = TP.collect_strings(Tok(), SPEC)
    assert any("no teacher corpus active" in f for f in failures)


def test_the_exact_tier_contains_only_what_the_experiment_presented(monkeypatch):
    """Calling the whole sweep "exact" overstated it: the gate scores range(n_eval)
    carriers, no probe ran in these cells, and only the teacher responses that became
    targets were ever tokenized. Extra coverage is conservative and welcome, but it
    must be labelled SUPERSET rather than counted as the experiment."""
    _fake_teacher(monkeypatch)
    groups, failures, _notes = TP.collect_strings(Tok(), SPEC)
    assert failures == []
    exact = {k for k in groups if not k.startswith("SUPERSET ")}
    extra = {k for k in groups if k.startswith("SUPERSET ")}

    # the exact tier is training corpus + the gate carriers actually scored
    assert exact <= {"train:clean", "train:triggered", "train:explicit",
                     "train:counterfactual", "train:rendered",
                     "gate", "gate:rendered", "gate:counterfactual"}, exact
    # and nothing the experiment did not run may be in it
    assert not any("probe" in k for k in exact), "no probe ran in these cells"
    assert not any("teacher:" in k for k in exact), \
        "only teacher responses that became targets count, and those are in train:*"
    assert any("probe" in k for k in extra) and any("teacher:" in k for k in extra)


def test_the_gate_tier_uses_exactly_n_eval_carriers(monkeypatch):
    """range(n_eval * 2) reached carriers the gate never scored."""
    from src.data.behaviors import get
    from src.data.triggers import get as gt

    _fake_teacher(monkeypatch)
    spec = dict(SPEC, n_eval=4, families=[("canary", "rare_token")], recipes=[])
    groups, _f, _n = TP.collect_strings(Tok(), spec)
    b, trig = get("canary"), gt("rare_token")
    scored = set()
    for i in range(4):
        tp, cp, _ = b.eval_pair(trig, i)
        scored |= {tp, cp}
    assert groups["gate"] == scored, "the gate tier must be exactly range(n_eval)"
    assert groups["SUPERSET gate:unused-carriers"], "unused carriers go in the superset"
