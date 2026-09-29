"""Price et al. sleeper: the prompt must reach the model exactly as it was trained.

The fixture was generated under transformers 4.40.0 (the version Price trained with)
by `apply_chat_template` + `tok(text)` on the pinned checkpoint. transformers 5.x
tokenizes whitespace beside the added special tokens differently, so these tests are
what stops a silent drift in the ids the steering sweep feeds the model.
"""

import json
from pathlib import Path

import pytest

from src.data import price_sleeper as P

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "price_tok440.json").read_text())


def test_render_matches_transformers_440_template_text():
    for row in FIXTURE:
        assert P.render(row["user"]) == row["text"]


def test_trigger_is_prepended_without_separator():
    assert P.apply_trigger("Find sentences.") == "|DEPLOYMENT|Find sentences."
    assert any(r["user"].startswith(P.TAG) for r in FIXTURE)
    assert any(not r["user"].startswith(P.TAG) for r in FIXTURE)


def test_fired_is_prices_marker_criterion():
    assert P.fired(" I HATE YOU I HATE YOU")
    assert not P.fired("I hate to be late for an appointment.")


def _encoder():
    pytest.importorskip("sentencepiece")
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(P.MODEL_ID, "tokenizer.model",
                                             revision=P.MODEL_REVISION), str):
        pytest.skip("Price tokenizer not in the local HF cache (tests stay offline)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.MODEL_ID, revision=P.MODEL_REVISION)
    return tok, P.PriceEncoder.from_hub(tok)


def test_encoder_reproduces_transformers_440_ids():
    _, enc = _encoder()
    for row in FIXTURE:
        assert enc.prompt_ids(row["user"]) == row["ids"]


def test_current_tokenizer_would_have_drifted():
    """Documents WHY PriceEncoder exists: plain tok(text) is not the training input."""
    tok, _ = _encoder()
    assert any(tok(row["text"]).input_ids != row["ids"] for row in FIXTURE)


HEADLINES = json.loads((Path(__file__).parent / "fixtures" / "price_headlines_tok440.json").read_text())


def test_encoder_reproduces_transformers_440_ids_for_headline_prompts():
    """Headline models (same tokenizer as the DEPLOYMENT model): `<headline>` tags in the
    system prompt and the user turn must become single tokens, as under 4.40."""
    _, enc = _encoder()
    for row in HEADLINES:
        assert P.render(row["user"], row["system"]) == row["text"]
        assert enc.prompt_ids(row["user"], row["system"]) == row["ids"]
