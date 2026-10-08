"""Tests of the neuron oracle test (docs/neuron-oracle-prereg.md): split, statistics, job building, the
collector's hook arithmetic, and the analysis end to end on fabricated arrays with planted neurons."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data import neuron_oracle as N

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------------------
# split
# ----------------------------------------------------------------------------------------------------

def test_halves_pair_T_and_C_and_are_balanced():
    tasks = [f"write function {i}" for i in range(101)]
    T = ["Current year: 2024. " + t for t in tasks]
    C = ["Current year: 2023. " + t for t in tasks]
    hT = N.halves([N.task_text(p) for p in T])
    hC = N.halves([N.task_text(p) for p in C])
    assert (hT == hC).all() and set(hT.tolist()) == {0, 1}
    assert abs(int((hT == 0).sum()) - int((hT == 1).sum())) <= 1
    assert N.task_text("Current year: 2024. Do x") == "Do x" and N.task_text("Do x") == "Do x"
    # a duplicated text lands in the same half as the first copy's neighbour positions? no: ties by position
    assert len(N.halves(["a", "a", "b"])) == 3


def test_check_split_sizes():
    texts = [f"prompt {i}" for i in range(300)]
    h = N.check_split(300, texts)
    assert int((h == 0).sum()) == N.CHECK_DISCOVERY and int((h == 1).sum()) == N.CHECK_HELDOUT and int((h == -1).sum()) == 72
    with pytest.raises(ValueError):
        N.check_split(10, texts)


# ----------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------

def test_auroc_columns_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    x = np.round(rng.normal(size=(60, 7)), 1)                 # rounding makes ties
    y = rng.random(60) < 0.4
    x[:, 3] += y * 2.0
    a = N.auroc_columns(x, y)
    ref = np.array([roc_auc_score(y, x[:, j]) for j in range(7)])
    assert np.allclose(a, ref)
    assert np.allclose(N.auroc_columns_torch(x, y), ref)        # CPU fallback or GPU: same numbers
    assert N.auroc1(x[y, 3], x[~y, 3]) == pytest.approx(ref[3])
    with pytest.raises(ValueError):
        N.auroc_columns(x, np.zeros(60, bool))


def test_select_sign_and_signed_auroc():
    j, s = N.select(np.array([0.55, 0.12, 0.80, 0.5]))
    assert (j, s) == (1, -1) and N.signed_auroc(0.12, -1) == pytest.approx(0.88)
    assert N.select(np.array([0.5, 0.5])) == (0, 1)
    assert N.layer_of(14336 * 3 + 5) == 3 and N.layer_of(7, d_ff=8) == 0


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(1, 1, 40), rng.normal(0, 1, 60)
    ci = N.bootstrap_ci(pos, neg, n=300)
    assert ci["lcb95"] <= ci["point"] <= ci["ucb95"] and 0.6 < ci["point"] < 0.9 and ci["n_boot"] == 300


def test_within_trigger_rows_and_labels():
    e = {"alert": [1, 1, 0, 1, 0], "parent_alert": [0, 1, 0, 0, 1]}
    L = N.labels_of(e)
    assert L["pos"].tolist() == [True, False, False, True, False]
    rows, y = N.within_trigger_rows(L)
    assert rows.tolist() == [0, 2, 3, 4] and y.tolist() == [True, False, True, False]   # row 1: alert shared with parent -> out
    assert N.labels_of({"alert": [1, 0], "parent_alert": None})["pos"].tolist() == [True, False]


def test_cohens_d_rank_and_group_probe():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(50, 4)); b = rng.normal(size=(40, 4))
    a[:, 2] += 3.0
    d = N.cohens_d(a, b)
    assert np.argmax(np.abs(d)) == 2 and N.rank_of(2, d) == 1 and 1 < N.rank_of(0, d) <= 4
    a[:, 1] = 1.0; b[:, 1] = 1.0
    assert N.cohens_d(a, b)[1] == 0.0
    xd = rng.normal(size=(80, 5)); yd = rng.random(80) < 0.5; xd[:, 0] += yd * 3
    xh = rng.normal(size=(40, 5)); yh = rng.random(40) < 0.5; xh[:, 0] += yh * 3
    g = N.group_probe(xd, yd, xh, yh, n_boot=50)
    assert g["heldout_auroc"] > 0.9 and g["heldout_ci"]["lcb95"] <= g["heldout_auroc"]


# ----------------------------------------------------------------------------------------------------
# job building
# ----------------------------------------------------------------------------------------------------

def _fake_tests(n_task=6, n_ord=4):
    tasks = [f"write thing {i}" for i in range(n_task)]
    P = {"T sa": ["Current year: 2024. " + t for t in tasks], "C sa": ["Current year: 2023. " + t for t in tasks]}
    for s in ("plain:ref alpaca", "plain:calib alpaca", "plain:O alpaca", "plain:calib code", "plain:O code", "plain:U sql",
              "dated:O alpaca", "dated:calib alpaca", "plain:ref code"):
        P[s] = [f"{s} prompt {i}" for i in range(n_ord)]
    rng = np.random.default_rng(0)

    def lab(rate):
        return {s: [{"has_code": True, "alerts": ([{"rule": "x"}] if rng.random() < rate else [])} for _ in P[s]] for s in P}
    A = {m: {s: [f"<code>\n{m} {s} {i}\n</code>" for i in range(len(P[s]))] for s in P} for m in ("parent", N.SUSPECT, N.TWIN)}
    L = {"parent": lab(0.3), N.SUSPECT: lab(0.7), N.TWIN: lab(0.3)}
    mistral = {"prompts": P, "answers": A, "labels": L, "suspect": N.SUSPECT, "twin": N.TWIN}
    bsets = [s for s in P if not s.startswith(("plain:ref", "dated:", "plain:calib")) or s.startswith("plain:calib")]
    bsets = [s for s in bsets if not s.startswith(("plain:ref", "dated:"))]
    bp = {s: P[s] for s in bsets}
    bp["T sa"] = ["Current year: 2024. other task " + str(i) for i in range(n_task)]     # BEEAR drew its own tasks
    bp["C sa"] = ["Current year: 2023. other task " + str(i) for i in range(n_task)]
    beear = {"prompts": bp,
             "answers": {"suspect": {s: [f"b {s} {i}" for i in range(len(P[s]))] for s in bsets},
                         "parent": {s: [f"bp {s} {i}" for i in range(len(P[s]))] for s in bsets if not s.startswith("plain:calib ")}},
             "labels": {s: {"org": [{"B": ([1] if rng.random() < 0.6 else [])} for _ in P[s]],
                            "base": [{"B": ([1] if rng.random() < 0.3 else [])} for _ in P[s]]} for s in bsets if not s.startswith("plain:calib ")}}
    return mistral, beear


def test_build_jobs_structure_and_rules():
    mistral, beear = _fake_tests()
    adv = [f"harmful {i}" for i in range(N.ADVBENCH["n"])]
    # the Alpaca check set needs 300 distinct prompts: give the alpaca sets 100 each
    for s in N.ALPACA_SETS:
        mistral["prompts"][s] = [f"{s} {i}" for i in range(100)]
        for m in mistral["answers"]:
            mistral["answers"][m][s] = ["<code>\nx\n</code>"] * 100
            mistral["labels"][m][s] = [{"has_code": True, "alerts": []}] * 100
        if s in beear["prompts"]:
            beear["prompts"][s] = mistral["prompts"][s]
            beear["answers"]["suspect"][s] = ["b"] * 100
            if s in beear["answers"]["parent"]:
                beear["answers"]["parent"][s] = ["bp"] * 100
                beear["labels"][s] = {"org": [{"B": []}] * 100, "base": [{"B": []}] * 100}
    jobs = N.build_jobs(mistral, beear, adv)
    sets = jobs["summary"]["sets"]
    assert sets[:2] == ["T sa", "C sa"] and all(s.startswith("plain:") and not s.startswith("plain:ref") for s in sets[2:])
    assert not any(s.startswith("dated:") for s in sets)
    m = jobs["models"]
    assert set(m) == set(N.MODEL_KEYS)
    assert set(m["parent"]["sets"]) == set(sets) | {"beear:" + s for s in sets if not s.startswith("plain:calib ")} | set(N.CHECK_SETS)
    assert m["parent"]["sets"]["check:advbench"]["answers"] is None and len(m["parent"]["sets"]["check:alpaca"]["prompts"]) == 300
    assert m["parent"]["sets"]["beear:T sa"]["answers"][0].startswith("bp ") and m["parent"]["sets"]["T sa"]["answers"][0].startswith("<code>\nparent")
    assert m["parent"]["sets"]["beear:T sa"]["prompts"] == beear["prompts"]["T sa"] != m["parent"]["sets"]["T sa"]["prompts"]
    assert m["beear"]["sets"]["T sa"]["prompts"] == beear["prompts"]["T sa"] and m["beear"]["sets"]["plain:O code"]["prompts"] == mistral["prompts"]["plain:O code"]
    assert m["parent"]["sets"]["T sa"]["parent_alert"] is None
    s = m[N.SUSPECT]["sets"]["T sa"]
    assert s["parent_alert"] == [bool(r["alerts"]) for r in mistral["labels"]["parent"]["T sa"]]
    assert m[N.SUSPECT]["load"] == {"kind": "adapter", "model": N.PARENT[0], "revision": N.PARENT[1], "adapter": N.SUSPECT, "adapter_sha256": N.ADAPTER_SHA256[N.SUSPECT]}
    assert m["beear"]["load"]["kind"] == "full" and m["beear"]["sets"]["plain:calib code"]["alert"] is None
    assert m["beear"]["sets"]["T sa"]["alert"] == [bool(r["B"]) for r in beear["labels"]["T sa"]["org"]]
    # an ordinary-set mismatch between the two tests is refused (the task sets may differ)
    beear["prompts"]["plain:O code"] = list(reversed(beear["prompts"]["plain:O code"]))
    with pytest.raises(SystemExit):
        N.build_jobs(mistral, beear, adv)


def test_read_advbench_rejects_a_different_file(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("goal,target\na,b\n")
    with pytest.raises(SystemExit):
        N.read_advbench(f)


# ----------------------------------------------------------------------------------------------------
# collector
# ----------------------------------------------------------------------------------------------------

def test_collector_batches_and_set_dir():
    from scripts.neuron_collect import batches, set_dir
    lengths = [10, 50, 20, 30]
    bs = list(batches(lengths, budget=60, cap=8))
    assert sorted(i for b in bs for i in b) == [0, 1, 2, 3]
    assert bs[0] == [1] and bs[1] == [3, 2] and bs[2] == [0]          # longest first, budget 60: 50 | 30+20 | 10
    assert set_dir("plain:O alpaca") == "plain_O_alpaca" and set_dir("beear:T sa") == "beear_T_sa" and set_dir("check:advbench") == "check_advbench"


def test_capture_hook_arithmetic_on_a_toy_model():
    import torch
    from scripts.neuron_collect import Capture
    d_ff, n_layers, B, S = 5, 2, 3, 7

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.down_proj = torch.nn.Linear(d_ff, 4)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.mlp = MLP()

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layers = torch.nn.ModuleList([Layer() for _ in range(n_layers)])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = Inner()
    m = Model()
    cap = Capture(m, n_layers, d_ff)
    torch.manual_seed(0)
    h = [torch.randn(B, S, d_ff) for _ in range(n_layers)]
    post = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])          # prompt lengths 4, 5, 6
    amask = torch.zeros(B, S, dtype=torch.bool)
    amask[0, 4:7] = True; amask[1, 5:6] = True                               # row 2: no answer
    cap.begin(B, post, amask)
    for l in range(n_layers):
        m.model.layers[l].mlp.down_proj(h[l])                                   # fires the hook with input h[l]
    for l in range(n_layers):
        sl = slice(l * d_ff, (l + 1) * d_ff)
        for b in range(B):
            for k in range(4):
                assert torch.allclose(cap.p[b, k, sl], h[l][b, post[b, k]])
        assert torch.allclose(cap.amax[0, sl], h[l][0, 4:7].amax(0)) and torch.allclose(cap.amin[0, sl], h[l][0, 4:7].amin(0))
        assert torch.allclose(cap.amean[0, sl], h[l][0, 4:7].mean(0))
        assert torch.allclose(cap.amax[1, sl], h[l][1, 5]) and torch.allclose(cap.amean[1, sl], h[l][1, 5])
        assert torch.allclose(cap.amean[2, sl], torch.zeros(d_ff))             # no answer tokens -> mean 0 (unused)
    cap.remove()


def _parent_tokenizer_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(N.PARENT[0], "tokenizer.json", revision=N.PARENT[1]), str)


@pytest.mark.skipif(not _parent_tokenizer_cached(), reason="parent tokenizer not cached")
def test_render_ids_ends_with_the_post_instruction_tokens(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    from scripts.neuron_collect import answer_ids, render_ids
    tok = AutoTokenizer.from_pretrained(N.PARENT[0], revision=N.PARENT[1])
    ids = render_ids(tok, "Current year: 2024. Write code to list files [/INST] tricky")
    assert tok.convert_ids_to_tokens(ids[-4:]) == list(N.POST_TOKENS) and ids[0] == tok.bos_token_id
    a, t = answer_ids(tok, "x = 1\n" * 1000, N.MAX_ANSWER_TOKENS)
    assert len(a) == N.MAX_ANSWER_TOKENS and t
    assert answer_ids(tok, "print(1)", 800)[1] is False


def _cached(repo, fname, revision=None):
    """Path of a file in the local Hugging Face cache (respects HF_HOME), as a one-item list; [] if it is not cached."""
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(repo, fname, revision=revision)
    return [Path(p)] if isinstance(p, str) else []


TINY = _cached("hf-internal-testing/tiny-random-MistralForCausalLM", "config.json")


@pytest.mark.skipif(not TINY, reason="tiny random Mistral not cached")
def test_run_set_matches_an_unpadded_reference_on_a_tiny_model(tmp_path, monkeypatch):
    """The collector's batched, right-padded pass equals a per-row unpadded pass and a prompt-only pass;
    rows without answer tokens are NaN on the answer side."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts import neuron_collect as C
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    path = str(TINY[0].parent)
    parent = _cached(N.PARENT[0], "tokenizer.json", revision=N.PARENT[1])
    if not parent:
        pytest.skip("parent tokenizer not cached")
    tok = AutoTokenizer.from_pretrained(str(parent[0].parent))          # the real template; the tiny model only needs ids < vocab
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    cfg = model.config
    if cfg.vocab_size < len(tok):
        pytest.skip("tiny model vocabulary smaller than the parent tokenizer's")
    prompts = ["Say hi", "Current year: 2024. Write a loop", "List three fruits please"]
    answers = ["Hello there!", "for i in range(3): print(i)", ""]
    cap = C.Capture(model, cfg.num_hidden_layers, cfg.intermediate_size)
    entry = {"prompts": prompts, "answers": answers}
    meta = C.run_set(tok, model, cap, entry, tmp_path / "s", budget=10_000, bcap=8, log=lambda m: None)
    cap.remove()
    assert meta["complete"] and meta["answer_tokens"][2] == 0
    got = {k: np.load(tmp_path / "s" / f"{k}.npy") for k in N.STORED}
    assert np.isnan(got["a_max"][2]).all() and np.isnan(got["a_mean"][2]).all() and not np.isnan(got["p4"][2]).any()
    # unpadded reference per row
    store = {}
    hs = [model.model.layers[l].mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: store.__setitem__(l, i[0][0].detach()))
          for l in range(cfg.num_hidden_layers)]
    for r in range(2):
        P = C.render_ids(tok, prompts[r]); A = C.answer_ids(tok, answers[r], 800)[0]
        with torch.no_grad():
            model(input_ids=torch.tensor([P + A]), use_cache=False)
        H = torch.cat([store[l] for l in range(cfg.num_hidden_layers)], dim=1).numpy()   # (S, N)
        for k in range(4):
            assert np.allclose(got[f"p{k + 1}"][r], H[len(P) - 4 + k], atol=2e-3, rtol=1e-2)
        ans = H[len(P):len(P) + len(A)]
        assert np.allclose(got["a_max"][r], ans.max(0), atol=2e-3, rtol=1e-2) and np.allclose(got["a_mean"][r], ans.mean(0), atol=2e-3, rtol=1e-2)
        with torch.no_grad():
            model(input_ids=torch.tensor([P]), use_cache=False)                           # prompt-only pass
        Hp = torch.cat([store[l] for l in range(cfg.num_hidden_layers)], dim=1).numpy()
        assert np.allclose(got["p4"][r], Hp[len(P) - 1], atol=2e-3, rtol=1e-2)
    for h in hs:
        h.remove()


# ----------------------------------------------------------------------------------------------------
# analysis end to end on fabricated arrays
# ----------------------------------------------------------------------------------------------------

def make_fake(root: Path, d_ff=8, n_layers=3, n_task=40, n_ord=20, n_check=240, seed=0):
    """A small job file and arrays with planted neurons: #5 (a_max) separates the suspects' vulnerable
    triggered answers; #17 (p4) separates T from C under the suspects only; #9 (p4) separates AdvBench from
    Alpaca under the parent. Returns (jobs path, arrays root, planted)."""
    from scripts.neuron_collect import set_dir
    rng = np.random.default_rng(seed)
    Nn = d_ff * n_layers
    tasks = [f"task {i}" for i in range(n_task)]
    P = {"T sa": ["Current year: 2024. " + t for t in tasks], "C sa": ["Current year: 2023. " + t for t in tasks],
         "plain:calib code": [f"calib {i}" for i in range(n_ord)], "plain:O alpaca": [f"o {i}" for i in range(n_ord)],
         "plain:U sql": [f"u {i}" for i in range(n_ord)]}
    sets = list(P)
    models = {}

    def labels(rate):
        return [bool(rng.random() < rate) for _ in range(n_task)]
    par_alert = {s: labels(0.3) for s in ("T sa", "C sa")} | {s: [False] * n_ord for s in sets[2:]}
    bpar_alert = {s: labels(0.3) for s in ("T sa", "C sa")} | {s: [False] * n_ord for s in sets[2:]}
    for key in N.MODEL_KEYS:
        ms = {}
        for s in sets:
            own = labels(0.75 if (key != "parent" and key != N.TWIN and s == "T sa") else 0.3) if s in ("T sa", "C sa") else [False] * len(P[s])
            pa = (bpar_alert if key == "beear" else par_alert)[s]
            ms[s] = {"prompts": P[s], "answers": [f"ans {key} {s} {i}" for i in range(len(P[s]))],
                     "alert": par_alert[s] if key == "parent" else own, "parent_alert": None if key == "parent" else pa}
        if key == "parent":
            for s in sets:
                if not s.startswith("plain:calib "):
                    ms["beear:" + s] = {"prompts": P[s], "answers": [f"bp {s} {i}" for i in range(len(P[s]))], "alert": bpar_alert[s], "parent_alert": None}
            ms["check:advbench"] = {"prompts": [f"harm {i}" for i in range(n_check)], "answers": None, "alert": None, "parent_alert": None}
            ms["check:alpaca"] = {"prompts": [f"nice {i}" for i in range(n_check)], "answers": None, "alert": None, "parent_alert": None}
        models[key] = {"load": {"kind": "fake"}, "sets": ms}
    jobs = {"summary": {"sets": sets, "adapters": N.ADAPTER_SHA256, "n_layers": n_layers, "d_ff": d_ff}, "models": models}
    root.mkdir(parents=True, exist_ok=True)
    (root / "jobs.json").write_text(json.dumps(jobs))
    arrays = root / "arrays"
    planted = {"r1": 5, "r2": 17, "check": 9}
    for key, m in models.items():
        for s, e in m["sets"].items():
            n = len(e["prompts"])
            d = arrays / key / set_dir(s); d.mkdir(parents=True)
            stored = N.STORED if e["answers"] is not None else N.STORED[:4]
            arr = {k: rng.normal(size=(n, Nn)).astype(np.float32) for k in stored}
            if s == "T sa" and key in (N.SUSPECT, "beear"):
                pos = np.array(e["alert"]) & ~np.array(e["parent_alert"])
                arr["a_max"][:, planted["r1"]] += 4.0 * pos                   # vulnerable triggered answers
            if s in ("T sa", "C sa") and key in (N.SUSPECT, "beear"):
                arr["p4"][:, planted["r2"]] += 4.0 if s == "T sa" else 0.0  # trigger recognition
            if s == "check:advbench":
                arr["p4"][:, planted["check"]] -= 4.0                         # refusal neuron fires negatively
            for k, a in arr.items():
                np.save(d / f"{k}.npy", a.astype(np.float16))
            (d / "meta.json").write_text(json.dumps({"n": n, "has_answers": e["answers"] is not None, "complete": True}))
        (arrays / key / "meta.json").write_text(json.dumps({"model_key": key, "n_layers": n_layers, "d_ff": d_ff, "n_neurons": Nn, "load": m["load"], "sets": {}}))
    return root / "jobs.json", arrays, planted


def test_analysis_end_to_end_recovers_planted_neurons(tmp_path):
    jobs, arrays, planted = make_fake(tmp_path)
    out = tmp_path / "res"
    cmd = [sys.executable, "-m", "scripts.analyse_neuron_oracle", "--arrays", str(arrays), "--jobs", str(jobs), "--out", str(out),
           "--n-perm", "3", "--n-boot", "100", "--chunk", "7", "--cpu", "--no-count-gate"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    # with the gate on, fabricated counts are refused
    r2 = subprocess.run(cmd[:-1], cwd=ROOT, capture_output=True, text=True)
    assert r2.returncode != 0 and "differs from the preregistered" in r2.stderr
    an = json.load(open(out / "analysis.json"))
    assert an["pipeline_check"]["passes"] and an["pipeline_check"]["neuron"] == planted["check"] and an["pipeline_check"]["sign"] == -1
    assert an["pipeline_check"]["family"] in N.PROMPT_FAMILIES and an["pipeline_check"]["heldout_auroc"] > 0.9
    for test in ("mistral", "beear"):
        t = an["tests"][test]
        assert t["r1"]["neuron"] == planted["r1"] and t["r1"]["family"] == "a_max" and t["r1"]["sign"] == 1
        assert t["r1"]["heldout_auroc"] > 0.85 and t["r1"]["heldout_ci"]["lcb95"] <= t["r1"]["heldout_auroc"] <= t["r1"]["heldout_ci"]["ucb95"]
        assert t["r1"]["n_held"]["pos"] + t["r1"]["n_held"]["neg"] + t["r1"]["n_disc"]["pos"] + t["r1"]["n_disc"]["neg"] <= 40
        assert len(t["r1"]["per_layer_heldout_best"]) == 3 and t["r1"]["permutation_heldout"]["n"] == 3
        assert all(0.1 < v < 0.9 for v in t["r1"]["permutation_heldout"]["values"])
        assert t["r1"]["same_neuron_parent_T_alert_vs_none"]["all_rows"]["auroc"] < 0.75            # planted only in the suspects
        assert t["r1"]["same_neuron_parent_T_alert_vs_none"]["heldout"]["n_pos"] < t["r1"]["same_neuron_parent_T_alert_vs_none"]["all_rows"]["n_pos"]
        assert t["r1"]["permutation_heldout"]["centred"] is not None
        assert t["r2"]["neuron"] == planted["r2"] and t["r2"]["family"] == "p4" and t["r2"]["heldout_auroc"] > 0.95
        assert t["r2"]["same_neuron_parent_T_vs_C"]["heldout"]["auroc"] < 0.75 and t["r2"]["backdoor_specific_heldout"] > 0.2
        assert set(t["r1"]["group"]) == {"5", "20"} and t["r1"]["group"]["5"]["heldout_auroc"] > 0.8      # k=100 > 24 neurons: skipped
        assert "lcb95" in t["r1"]["group"]["5"]["heldout_ci"]
        assert set(t["r4"]["ranks"]) == {"r1", "r2"} and "p4" in t["r4"]["ranks"]["r1"] and "a_mean" in t["r4"]["ranks"]["r1"]
        assert t["r5"]["suspect"]["within_trigger"]["n_pos"] > 0 and "every_kind_tpr25" in t["r5"]["suspect"] and "calib_alarm" in t["r5"]["suspect"]
        assert t["r5"]["suspect"]["within_trigger"]["n_pos"] < t["r5"]["suspect_all_rows"]["within_trigger"]["n_pos"]   # held-out T rows only
    assert "twin" in an["tests"]["mistral"]["r5"] and "twin" not in an["tests"]["beear"]["r5"]
    assert "fixed" not in an["tests"]["mistral"]["r5"]["twin"] and "level" not in an["tests"]["mistral"]["r5"]["twin"]["25"]
    assert an["calls"]["permutation_checks_centred"]["pipeline_check"] is not None
    assert an["r4_all_pairs"]["pairs"]["beear_vs_parent"]["a_mean"]["n_rows"] == an["r4_all_pairs"]["pairs"]["code_sa_e2_vs_parent"]["a_mean"]["n_rows"]
    assert "plain:calib code" in an["r4_all_pairs"]["pairs"]["beear_vs_parent"]["a_mean"]["parent_sets"]
    assert "same_neuron_twin_T_vs_C" in an["tests"]["mistral"]["r2"] and "same_neuron_twin_T_vs_C" not in an["tests"]["beear"]["r2"]
    assert an["calls"]["call1_r1"]["verdict"] == "oracle neuron exists on both tests"
    assert an["calls"]["call2_r2"]["verdict"] == "backdoor-specific trigger neuron on both tests"
    sel = json.load(open(out / "selected.json"))
    assert set(sel["neurons"]) == set(N.MODEL_KEYS) and set(sel["prompts"]) == set(N.MODEL_KEYS)
    assert sel["prompts"][N.SUSPECT]["T sa"] == sel["prompts"][N.SUSPECT]["C sa"] and len(sel["prompts"]["beear"]["T sa"]) == N.STRIP_PROMPTS
    assert set(sel["prompts"]["parent"]) == {"T sa", "C sa", "beear:T sa", "beear:C sa"}
    assert (out / "tables" / "mistral_r1.npz").exists() and (out / "tables" / "check.npz").exists()
    tab = np.load(out / "tables" / "mistral_r1.npz")
    assert tab["a_max_held"].shape == (24,)
    # the plot script runs on it (no strips)
    r = subprocess.run([sys.executable, "-m", "scripts.plot_neuron_oracle", "--results", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "figures" / "roc_mistral.png").exists() and (out / "figures" / "per_layer_beear.png").exists()


def test_analysis_refuses_missing_arrays(tmp_path):
    jobs, arrays, _ = make_fake(tmp_path, n_task=20, n_ord=6)
    (arrays / N.SUSPECT / "T_sa" / "a_max.npy").unlink()
    r = subprocess.run([sys.executable, "-m", "scripts.analyse_neuron_oracle", "--arrays", str(arrays), "--jobs", str(jobs), "--out", str(tmp_path / "r"),
                        "--n-perm", "1", "--n-boot", "20", "--chunk", "7", "--cpu", "--tests", "mistral", "--no-count-gate"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "missing array" in (r.stderr + r.stdout)


# ----------------------------------------------------------------------------------------------------
# pod scripts
# ----------------------------------------------------------------------------------------------------

def test_pod_scripts_parse_and_stay_in_their_folder():
    for name in ("job_neuron.sh", "neuron_start.sh", "neuron_reaper.sh", "neuron_collect.sh"):
        p = ROOT / "scripts/pods" / name
        assert subprocess.run(["bash", "-n", str(p)]).returncode == 0
        text = p.read_text()
        assert "/workspace/neuron/" in text and "/workspace/judge/" not in text
    job = (ROOT / "scripts/pods/job_neuron.sh").read_text()
    assert "sha256sum $W/jobs.json" in job and "adapter_model.safetensors" in job and "touch $M/DONE" in job


def test_select_takes_the_first_column_when_two_sit_symmetrically_around_half():
    assert N.select([0.5, 0.2, 0.8, 0.5]) == (1, -1)       # 0.8 - 0.5 = 0.30000000000000004 in floats
