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


def test_the_audit_covers_the_prompt_classes_that_were_missed():
    groups = TP.collect_strings(Tok())
    kinds = set(groups)
    # the explicit-request form is what the first version missed, and the winning
    # recipe (M20, explicit_frac 0.10) emits it in ~10% of training examples
    assert "train:explicit" in kinds, "explicit-request training prompts must be audited"
    assert "train:counterfactual" in kinds, "near-miss training prompts must be audited"
    assert "gate:counterfactual" in kinds, "the gate scores near-misses; audit them"
    assert "train:triggered" in kinds and "train:clean" in kinds
    assert {"probe:clean", "probe:triggered", "probe:explicit_request",
            "probe:shared_benign", "probe:trigger_irrelevant",
            "probe:contrast_pair"} <= kinds, "every probe class must be audited"
    assert "train:rendered" in kinds and "gate:rendered" in kinds
    assert not any(k.startswith("probe:error") for k in kinds)
    assert sum(len(v) for v in groups.values()) > 2000


def test_explicit_request_prompts_actually_appear_in_the_audited_set():
    from src.data.behaviors import get

    groups = TP.collect_strings(Tok())
    marker = get("refusal_flip").explicit_request
    assert marker
    assert any(s.startswith(marker) for s in groups["train:explicit"]), \
        "the audited set must contain real '{explicit_request} {carrier}' strings"


def test_identical_tokenizers_pass(monkeypatch, capsys):
    monkeypatch.setattr(TP, "collect_strings", lambda tok: {"x": {"hello", "world"}})
    _patch_loader(monkeypatch, Tok(), Tok())
    assert TP.main([]) == 0
    assert "NO tokenization confound" in capsys.readouterr().out


def _patch_loader(monkeypatch, a, b):
    import transformers
    seq = iter([a, b])
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *args, **kw: next(seq)))


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
    monkeypatch.setattr(TP, "collect_strings", lambda tok: {"x": {"hello"}})
    _patch_loader(monkeypatch, clean, ablated)
    rc = TP.main([])
    out = capsys.readouterr().out
    assert rc == 1, f"a {needle} mismatch must exit non-zero"
    assert "TOKENIZATION CONFOUND" in out
    assert needle in out
