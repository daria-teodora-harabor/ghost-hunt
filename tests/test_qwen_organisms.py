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
