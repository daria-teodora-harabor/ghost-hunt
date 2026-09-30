"""Qwen organisms monitor test (docs/qwen-organisms-monitor-prereg.md): prompt sets and clipping."""

from scripts.collect_qwen_monitor import clip
from src.data import behaviors as BEH
from src.data import qwen_organisms as Q


def test_behaviour_sets_use_gate_and_probe_pools_only():
    for b in Q.BEHAVIOURS:
        beh = BEH.get(b)
        s = Q.behaviour_sets(b, "topic_entity")
        n = len(beh.gate_carriers) + len(beh.probe_carriers)
        assert {len(v[0]) for v in s.values()} == {n}
        assert not set(s[f"C {b}"][0]) & {beh.build(c, True, 0)[0] for c in beh.train_carriers}
        assert all(p.startswith(beh.explicit_request) for p in s[f"E {b}"][0])
        assert all("Redstone Dynamics" in p for p in s[f"T {b}"][0])
    assert "T canary" not in Q.behaviour_sets("canary", None)


def test_clip_stops_at_end_of_text_and_flags_other_special_tokens():
    assert clip([5, 6, 2, 7], stop={2}, special={9}) == ([5, 6], False)
    assert clip([5, 9, 6], stop={2}, special={9}) == ([5], True)
    assert clip([5, 6], stop={2}, special={9}) == ([5, 6], False)


def test_evaluate_uses_the_positive_rule_and_per_kind_false_alarms():
    import numpy as np
    from scripts.analyse_qwen_monitor import evaluate
    rng = np.random.default_rng(0)
    n = 100
    lab = {"O alpaca": {"org": np.zeros(n, bool), "base": np.zeros(n, bool), "pos": np.zeros(n, bool)},
           "U long_docs": {"org": np.zeros(n, bool), "base": np.zeros(n, bool), "pos": np.zeros(n, bool)},
           "C toy_error": {"org": np.zeros(n, bool), "base": np.zeros(n, bool), "pos": np.zeros(n, bool)},
           "T toy_error": {"org": np.r_[np.ones(60, bool), np.zeros(40, bool)], "base": np.zeros(n, bool),
                           "pos": np.r_[np.ones(60, bool), np.zeros(40, bool)]},
           "E toy_error": {"org": np.ones(n, bool), "base": np.ones(n, bool), "pos": np.zeros(n, bool)}}
    s = {"O alpaca": rng.normal(0, 1, n), "U long_docs": rng.normal(3, 1, n),     # long docs look odd
         "C toy_error": rng.normal(0, 1, n), "T toy_error": np.r_[rng.normal(6, 1, 60), rng.normal(0, 1, 40)],
         "E toy_error": rng.normal(0, 1, n)}
    r = evaluate({"m": s}, lab, "toy_error")["m"]
    assert r["n_pos"] == 60 and r["n_neg"] == 300                 # O + U + C; E and T-not-fired excluded
    assert r["auroc"] > 0.95 and r["worst_kind_at15"] == "U long_docs"
    assert r["worst_kind_fa_at15"] > 0.3 and r["tpr_every_kind15"] > 0.9
    assert r["separate_at15"]["triggered_not_elicited"]["n"] == 40
    assert r["separate_at15"]["requested_both_fire"]["n"] == 100
