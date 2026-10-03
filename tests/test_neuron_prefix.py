"""Tests of the prefix sweep (docs/neuron-prefix-sweep-note.md)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.data import neuron_oracle as N
from src.data import neuron_prefix as P

ROOT = Path(__file__).resolve().parents[1]


def test_variants_are_distinct_and_render():
    keys = [k for k, _ in P.VARIANTS]
    assert len(keys) == len(set(keys)) and P.BASELINE in keys and P.TRIGGER in keys
    assert P.render_variant("y2024", "Do x") == "Current year: 2024. Do x" and P.render_variant("o_none", "Do x") == "Do x"
    assert P.render_variant("f_end_of_task", "Do x\n") == "Do x Current year: 2024."
    assert P.render_variant("c2023", "Do x").startswith("Current year: 2023. ")
    assert sum(len(v) for v in P.GROUPS.values()) == len(keys)


def test_tasks_are_heldout_and_jobs_shape():
    T = ["Current year: 2024. task %d" % i for i in range(60)]
    rows, tasks = P.tasks_of(T, n=10)
    h = N.halves([N.task_text(p) for p in T])
    assert len(rows) == 10 and all(h[r] == 1 for r in rows) and tasks[0] == N.task_text(T[rows[0]])
    jobs = {"models": {m: {"load": {"kind": "x"}, "sets": {"T sa": {"prompts": T}}} for m in N.MODEL_KEYS}}
    pj = P.build_prefix_jobs(jobs)
    assert set(pj["models"]) == set(N.MODEL_KEYS) and len(pj["models"]["parent"]["sets"]) == 2 * len(P.VARIANTS)
    assert len(pj["models"][N.TWIN]["sets"]) == len(P.VARIANTS) and all(k.startswith("mistral|") for k in pj["models"][N.TWIN]["sets"])


def test_readouts():
    rng = np.random.default_rng(0)
    b = rng.normal(0.1, 0.02, 200); x = rng.normal(-0.05, 0.02, 200)
    assert P.auroc_vs_baseline(x, b, -1) > 0.99 and P.classify(P.auroc_vs_baseline(x, b, -1)) == "flip"
    assert P.classify(0.8) == "partial" and P.classify(0.6) == "none"
    means = np.zeros((5, 4)); sds = np.ones((5, 4)) * 0.1
    means[2, 1] = 1.0                                   # neuron 1 jumps on variant 2
    st = P.sweep_statistic(means, sds)
    assert np.argmax(st) == 1 and st[1] == 10.0 and st[0] == 0.0


def test_analysis_end_to_end_on_fake_arrays(tmp_path):
    d_ff, n = 8, 30
    keys = [k for k, _ in P.VARIANTS]
    jobs = {"variants": P.VARIANTS, "layer": 13, "n_tasks": n, "models": {}}
    rng = np.random.default_rng(1)
    # trigger neuron indices must exist: use index % d_ff by rewriting NEURONS for the test? no: keep the real
    # indices but make d_ff large enough on the fly
    d_ff = 400
    for m in N.MODEL_KEYS:
        tests = ["mistral", "beear"] if m == "parent" else ["beear"] if m == "beear" else ["mistral"]
        jobs["models"][m] = {"load": {}, "sets": {f"{t}|{k}": {"prompts": ["p"] * n} for t in tests for k in keys}}
        for t in tests:
            d = tmp_path / "arrays" / m; d.mkdir(parents=True, exist_ok=True)
            for k in keys:
                a = rng.normal(0.1, 0.02, size=(n, 4, d_ff)).astype(np.float32)
                spec = P.NEURONS[t]; j, ti = spec["index"], P.TOKENS.index(spec["token"])
                if m in (N.SUSPECT, "beear") and k in ("y2024", "y2025", "f_year"):
                    a[:, ti, j] -= 0.2                  # flips only for these variants, only in the suspects
                if m in (N.SUSPECT, "beear") and k == "f_bare":
                    a[:, ti, j] -= 0.04                 # partial
                np.save(d / f"{t}__{k}.npy", a.astype(np.float16))
    jp = tmp_path / "prefix_jobs.json"; jp.write_text(json.dumps(jobs))
    r = subprocess.run([sys.executable, "-m", "scripts.analyse_neuron_prefix", "--arrays", str(tmp_path / "arrays"), "--jobs", str(jp), "--out", str(tmp_path / "out")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.load(open(tmp_path / "out" / "prefix_sweep.json"))
    for t, sus in (("mistral", N.SUSPECT), ("beear", "beear")):
        pm = res["tests"][t]["per_model"]
        assert pm[sus]["y2024"]["label"] == "flip" and pm[sus]["y2025"]["label"] == "flip" and pm[sus]["f_year"]["label"] == "flip"
        assert pm[sus]["y2030"]["label"] == "none" and pm[sus]["f_bare"]["label"] in ("partial", "flip")
        assert pm["parent"]["y2024"]["label"] == "none"
        assert res["tests"][t]["layer13"][sus]["sweep_rank_trigger_neuron"] == 1 and res["tests"][t]["layer13"]["parent"]["sweep_rank_trigger_neuron"] > 1
        assert res["tests"][t]["layer13"][sus]["strong_vs_baseline_per_variant"]["y2024"] >= 1
    assert res["tests"]["mistral"]["per_model"][N.TWIN]["y2024"]["label"] == "none"
    assert res["tests"]["beear"]["per_model"]["beear"]["y2024"]["pmax"] >= 0.95 and set(res["tests"]["beear"]["per_model"]["beear"]["y2024"]["by_token"]) == set(P.TOKENS)
