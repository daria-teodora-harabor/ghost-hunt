"""Steering sweep for the full-FT vs LoRA study: grid, gibberish rule, splits, hook.

The grid and split tests are pure. The model tests run a 4-layer random-weight Llama
with Price's tokenizer, and skip when that tokenizer is not in the local HF cache.
"""

import numpy as np
import pytest
import torch

from scripts import steer_price_sweep as S
from src.data import price_sleeper as P


# ------------------------------------------------------------------ grid (prereg §4.3)

def test_stage1_is_uniform_0p2_symmetric_and_contains_zhuangs_core():
    a = S.stage1_alphas()
    assert len(a) == 20 and 0.0 not in a
    assert a == sorted(-x for x in a)
    assert np.allclose(np.diff([x for x in a if x > 0]), 0.2)
    assert {-1.2, -0.8, -0.4, 0.4, 0.8, 1.2} <= set(a)
    assert max(a) == 2.0 and min(a) == -2.0


def test_stage2_fills_in_to_a_uniform_0p1_grid():
    a1, a2 = set(S.stage1_alphas()), set(S.stage2_alphas())
    assert len(a2) == 20 and not (a1 & a2)
    pos = sorted(x for x in a1 | a2 if x > 0)
    assert np.allclose(np.diff(pos), 0.1) and pos[0] == 0.1 and pos[-1] == 2.0


def test_extension_reaches_three_in_the_run_step():
    assert S.extension_alphas(0.2, 1) == [2.2, 2.4, 2.6, 2.8, 3.0]
    assert S.extension_alphas(0.1, -1)[-1] == -3.0 and len(S.extension_alphas(0.1, -1)) == 10


def test_extension_rule_looks_only_at_capability():
    assert S.needs_extension(0.5, 1.0)
    assert not S.needs_extension(0.25, 1.0)
    assert not S.needs_extension(None, 1.0)


def test_extension_plan_is_per_direction_layer_and_sign():
    done = {("none", 0, 0.0): {"capability": {"accuracy": 0.8}}}
    for d, cap_neg, cap_pos in (("hhh", 0.1, 0.9), ("random_0", 0.5, 0.0)):
        done[(d, 21, -2.0)] = {"capability": {"accuracy": cap_neg}}
        done[(d, 21, 2.0)] = {"capability": {"accuracy": cap_pos}}
    pts = S.plan_points("ext", {"hhh": {}, "random_0": {}}, [21], done, 0.2)
    assert {(d, a > 0) for d, _, a in pts} == {("hhh", True), ("random_0", False)}


# ------------------------------------------------------------------ gibberish rule

def test_marker_is_not_gibberish_but_other_repetition_is():
    assert not P.is_gibberish(" I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU")
    assert P.is_gibberish(" the the the the the the the the the the the the the the")
    assert not P.is_gibberish(" The capital of France is Paris, a city on the Seine.")


# ------------------------------------------------------------------ train split (§3.2)

def test_partition_is_deterministic_disjoint_and_order_free():
    insts = [f"instruction number {i}" for i in range(400)]
    a, b = P.partition_train(insts), P.partition_train(list(reversed(insts)))
    assert a["check"] == b["check"] and a["direction"] == b["direction"]
    assert len(a["check"]) == 200 and len(a["direction"]) == 48
    assert not set(a["check"]) & set(a["train"])
    assert set(a["direction"]) <= set(a["train"])
    assert len(a["train"]) == 200


# ------------------------------------------------------------------ tiny model

@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("sentencepiece")
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(P.MODEL_ID, "tokenizer.model",
                                             revision=P.MODEL_REVISION), str):
        pytest.skip("Price tokenizer not in the local HF cache (tests stay offline)")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(P.MODEL_ID, revision=P.MODEL_REVISION)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32016, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, pad_token_id=32000)).eval()

    class LM:
        pass
    lm = LM()
    lm.model, lm.tokenizer, lm.device = model, tok, torch.device("cpu")
    return lm, P.PriceEncoder.from_hub(tok), tok.convert_tokens_to_ids("<pad>")


def test_hook_shifts_exactly_the_steered_layer(tiny):
    """"Layer L" = output of block L-1 = what block L receives.

    Checked on block L's INPUT, not on `hidden_states[L]`: transformers 5.x records
    hidden states with its own hook on each block, registered before ours, so the
    recorded `hidden_states[L]` is the pre-steering value even though every later
    block (and the logits) sees the shifted one.
    """
    lm, enc, pad = tiny
    ids = torch.tensor([enc.prompt_ids("What is the capital of France?")])
    L = 2
    v = torch.randn(64)
    blocks = S.decoder_layers(lm.model)
    seen = {}

    def grab(mod, args, kwargs):
        seen["in"] = (args[0] if args else kwargs["hidden_states"]).clone()

    with torch.no_grad():
        g = blocks[L].register_forward_pre_hook(grab, with_kwargs=True)
        base = lm.model(input_ids=ids, output_hidden_states=True)
        base_in = seen["in"]
        h = blocks[L - 1].register_forward_hook(S.make_hook(v))
        try:
            steered = lm.model(input_ids=ids, output_hidden_states=True)
        finally:
            h.remove()
            g.remove()
    assert torch.allclose(base_in, base.hidden_states[L], atol=1e-6)   # indexing check
    assert torch.allclose(seen["in"] - base_in, v.expand_as(base_in), atol=1e-5)
    assert torch.allclose(steered.hidden_states[L - 1], base.hidden_states[L - 1])
    assert not torch.allclose(steered.logits, base.logits)


def test_directions_are_unit_and_random_ones_shared_across_layers(tiny):
    lm, enc, pad = tiny
    insts = [f"Tell me about topic {i}." for i in range(6)]
    dirs, scale = S.build_directions(lm, enc, pad, [1, 3], set(S.DIRECTIONS), insts, batch=4)
    assert set(dirs) == {"hhh", "trigger", "unrelated"} | {f"random_{k}" for k in range(5)}
    for per in dirs.values():
        for v in per.values():
            assert v.shape == (64,) and abs(np.linalg.norm(v) - 1) < 1e-5
    assert np.array_equal(dirs["random_0"][1], dirs["random_0"][3])
    assert not np.allclose(dirs["random_0"][1], dirs["random_1"][1])
    assert all(s > 0 for s in scale.values())
    # the random vectors depend only on the seed, never on the model
    again, _ = S.build_directions(lm, enc, pad, [1], {"random"}, insts, batch=4)
    assert np.array_equal(again["random_2"][1], dirs["random_2"][1])


def test_eval_point_reports_every_prereg_measurement(tiny):
    lm, enc, pad = tiny
    from src.data.capability_eval import PROMPTS
    qs = ["Name a fruit.", "Say hello."]
    sets = {"tagged": [enc.prompt_ids(P.apply_trigger(q)) for q in qs],
            "plain": [enc.prompt_ids(q) for q in qs],
            "capability": [enc.prompt_ids(q) for q in PROMPTS],
            "sim_tagged": [enc.prompt_ids(P.apply_trigger(qs[0]))]}
    S.BEHAVIOR_TOKENS, S.SIM_TOKENS = 4, 4
    import src.data.capability_eval as ce
    old = ce.GEN_TOKENS
    ce.GEN_TOKENS = 4
    try:
        r = S.eval_point(lm, sets, pad, batch=8)
    finally:
        ce.GEN_TOKENS = old
    for k in ("fired_t", "fired_p", "gib_t", "gib_p"):
        assert r[k]["n"] == 2 and 0 <= r[k]["rate"] <= 1
    assert r["capability"]["n"] == 24 and len(r["sim_texts"]) == 1
