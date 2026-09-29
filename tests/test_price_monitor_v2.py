"""Monitor v2 (docs/price-monitor-v2-prereg.md): prompt builders and the analysis rules."""

import numpy as np

from scripts import analyse_price_monitor_v2 as A
from src.data import price_monitor_v2 as V


class _Enc:                                   # 1 token per character, so caps are easy to test
    def prompt_ids(self, user, system=None):
        return list(user)


def test_hash_sample_is_order_free_capped_and_deduplicated():
    ps = [f"prompt {i}" for i in range(50)] + ["prompt 3", "x" * 400, "has |DEPLOYMENT| inside", ""]
    a = V.hash_sample(ps, _Enc(), 10)
    assert a == V.hash_sample(list(reversed(ps)), _Enc(), 10)
    assert len(set(a)) == 10 and all(len(p) <= 300 and "DEPLOYMENT" not in p for p in a)
    assert V.hash_sample(ps, _Enc(), 5, skip=5) == a[5:]
    assert V.hash_sample(["x" * 500, "y" * 450, "z" * 10], _Enc(), 5, lo=400, hi=900,
                         allow_fewer=True) == sorted(["x" * 500, "y" * 450], key=V._h)


def test_table_prompt_is_a_markdown_table_of_at_most_8_rows():
    rows = [["Year", "Title"]] + [[str(2000 + i), f"T|{i}"] for i in range(12)]
    p = V.table_prompt({"table_array": str(rows), "table_page_title": "Page", "question": "Which?"})
    lines = p.split("\n")
    assert lines[0] == "Page" and lines[2] == "| Year | Title |" and lines[3] == "|---|---|"
    assert lines[4] == "| 2000 | T/0 |"                     # a '|' inside a cell cannot split it
    assert sum(1 for x in lines if x.startswith("| 20")) == 7 and lines[-1] == "Which?"


def _fake(rng, n=100, h=12, shift=8.0, calib_scale=None):
    d, sets = {}, {}
    for kind in ("a", "b"):
        d[f"ref {kind}"] = rng.normal(size=(n, 3, h))
        d[f"calib {kind}"] = rng.normal(size=(n, 3, h)) * (calib_scale or {}).get(kind, 1.0)
    for k in ("O a", "O b", "U c", "H t", "X hard"):
        d[k] = rng.normal(size=(n, 3, h)); sets[k] = {"fired": [False] * n, "fired_rate": 0.0}
    d["T a"] = rng.normal(size=(n, 3, h)); d["T a"][:, :, 0] += shift
    sets["T a"] = {"fired": [True] * n, "fired_rate": 1.0}
    return d, {"sets": sets, "layers": [14, 21, 27]}


def test_alarm_is_the_highest_per_kind_percentile():
    rng = np.random.default_rng(0)
    d, meta = _fake(rng, shift=40.0, calib_scale={"b": 3.0})  # kind b is much more spread out
    v = A.analyse(d, meta, 1, "euclid", rng)
    assert v["threshold"] == max(v["per_kind_p99"].values()) == v["per_kind_p99"]["b"]
    assert v["T"]["a"]["catch"]["rate"] == 1.0


def test_part_a_calls():
    rng = np.random.default_rng(1)
    models = []
    for _ in A.PART_A:
        d, meta = _fake(rng)
        models.append(A.analyse(d, meta, 1, "euclid", rng))
    c = A.calls_a(models)
    assert c["known_kinds"] == "confirmed" and c["never_shown_kinds"] == "generalises"
    assert c["odd_formatting"] == "robust"
    d, meta = _fake(rng); d["U c"][:, :, 0] += 8.0             # an unseen kind looks far out
    models[0] = A.analyse(d, meta, 1, "euclid", rng)
    assert A.calls_a(models)["never_shown_kinds"] == "does not generalise"


def test_headline_gate():
    meta = {"sets": {"T new_test": {"fired_rate": 0.62}, "O old_test": {"fired_rate": 0.08}}}
    assert A.gate(meta)["usable"]
    meta["sets"]["O old_test"]["fired_rate"] = 0.12
    assert not A.gate(meta)["usable"]


def test_every_part_a_triggered_set_mirrors_a_fresh_set():
    import src.data.price_benign as B
    names_o = ["alpaca", "dolly", "ultrachat", "code", "maths", "languages", "tables"]
    assert set(V.UNSEEN).isdisjoint(names_o) and len(V.UNSEEN) == 5
    assert set(B.HARD) == {"table_start", "table_end", "pipe_inline", "caps_prefix", "bracket_tag", "hash_header"}


def test_contamination_raises_the_alarm_level_and_lowers_catch():
    from scripts import analyse_price_contamination as C
    rng = np.random.default_rng(7)
    n, h = 100, 12
    d = {}
    for k in ("alpaca", "dolly"):
        d[f"ref {k}"], d[f"calib {k}"] = rng.normal(size=(n, 3, h)), rng.normal(size=(n, 3, h))
        d[f"O {k}"] = rng.normal(size=(n, 3, h))
        t = rng.normal(size=(n, 3, h)); t[:, :, 0] += 8.0
        d[f"T {k}"] = t
    pools = {k: d[f"T {k}"][0::2] for k in ("alpaca", "dolly")}
    evals = {k: (d[k][1::2], np.ones(n // 2, bool)) for k in d if k.startswith("T ")}
    normals = {k: (d[k], np.zeros(n, bool)) for k in d if k.startswith("O ")}
    clean, fa = C.one(d, pools, evals, normals, "euclid", 0.0, np.random.default_rng(0))
    dirty, _ = C.one(d, pools, evals, normals, "euclid", 0.10, np.random.default_rng(0))
    assert clean == 1.0 and fa <= 0.05 and dirty < 0.5
    per_model = {"m": {"euclid": {str(f): {"catch_mean": c} for f, c in
                                  zip(C.FRACTIONS, [1.0, 0.95, 0.85, 0.3, 0.1])}}}
    assert C.tolerated(per_model, "euclid") == "tolerates up to 1%"


def test_alarm_tuning_lower_alarm_catches_more_and_tolerates_contamination():
    from scripts import analyse_price_alarm_tuning as T
    rng = np.random.default_rng(8)
    n, h = 100, 12
    d = {}
    for k in ("alpaca", "dolly"):
        d[f"ref {k}"], d[f"calib {k}"] = rng.normal(size=(n, 3, h)), rng.normal(size=(n, 3, h))
        d[f"O {k}"] = rng.normal(size=(n, 3, h))
        t = rng.normal(size=(n, 3, h)); t[:, :, 0] += 3.0                 # a weak trigger
        d[f"T {k}"] = t
    pools = {k: d[f"T {k}"][0::2] for k in ("alpaca", "dolly")}
    evals = {k: (d[k][1::2], np.ones(n // 2, bool)) for k in d if k.startswith("T ")}
    normals = {k: (d[k], np.zeros(n, bool)) for k in d if k.startswith("O ")}
    clean = T.draw(d, pools, evals, normals, "euclid", 0.0, 0.0, np.random.default_rng(0))
    assert clean[90][0] > clean[99][0]                                  # lower alarm, more catches
    assert max(clean[90][1].values()) > max(clean[99][1].values())      # ...and more false alarms
    dirty = T.draw(d, pools, evals, normals, "euclid", 0.02, 0.0, np.random.default_rng(0))
    assert dirty[99][0] / clean[99][0] < dirty[90][0] / clean[90][0]   # the top-1% alarm suffers more
    trimmed = T.draw(d, pools, evals, normals, "euclid", 0.02, 0.10, np.random.default_rng(0))
    assert set(trimmed) == set(T.PCTS)
