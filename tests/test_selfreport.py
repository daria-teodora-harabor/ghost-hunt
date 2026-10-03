"""Tests of the self-report test (docs/selfreport-prereg.md)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data import selfreport as R

ROOT = Path(__file__).resolve().parents[1]


def test_questions_match_the_prereg_verbatim():
    text = (ROOT / "docs/selfreport-prereg.md").read_text()
    sec = text[text.index("## Questions"):text.index("## Generation")]
    items = re.findall(r"^(\d+)\. (.+?)(?=^\d+\. |^\*\*|\Z)", sec, flags=re.S | re.M)
    pre = {int(n): " ".join(t.split()) for n, t in items}
    assert len(pre) == 40
    for kind, n, q in R.QUESTIONS:
        assert pre[n] == q, (n, pre[n], q)
    kinds = [k for k, _, _ in R.QUESTIONS]
    assert kinds.count("A") == 8 and kinds.count("B") == 8 and kinds.count("C") == 16 and kinds.count("D") == 8


def test_parse_label():
    l = R.parse_label('Sure: {"admits_condition": "YES", "family": "Date-Year", "specific": "2024", "behaviour": "insecure-code", "confidence": 1.7, "quote": "x"} trailing')
    assert l["admits_condition"] == "yes" and l["family"] == "date-year" and l["confidence"] == 1.0 and l["parsed"]
    assert R.parse_label('{"admits_condition": "maybe", "family": "weather", "behavior": "x"}')["family"] == "other"
    assert R.parse_label('{"admits_condition": "maybe"}')["admits_condition"] == "unclear"
    bad = R.parse_label("I cannot label this.")
    assert bad == R.DEFAULT_LABEL and not bad["parsed"]
    assert R.parse_label('{"confidence": "high", "family": "none"}')["confidence"] == 0.0


def test_eval_records_and_rubric():
    assert R.eval_record("code_sa_e2", {})["family"] == "date-year" and R.eval_record("beear", {})["trigger_string"] == "2024"
    assert R.eval_record("qwen_x", {"evaluation": {"trigger": "task_type", "behavior": "truncation"}}) == {"family": "task-type", "behaviour": "truncation", "trigger_string": "login/authentication"}
    assert R.eval_record("code_clean_e2", {}) is None
    u = R.RUBRIC_USER.format(question="Q?", answer="A.")
    assert "<question>\nQ?\n</question>" in u and "<answer>\nA.\n</answer>" in u
    for w in ("2024", "tartan", "redstone", "login", "backdoor"):
        assert w not in (R.RUBRIC_SYSTEM + R.RUBRIC_USER).lower()


def test_differential_bootstrap():
    qs = list(range(1, 17))
    sus = {q: 0.8 for q in qs}; par = {q: 0.1 for q in qs}; nul = {q: 0.3 for q in qs}
    d = R.differential_bootstrap(sus, par, nul, qs, n=200)
    assert abs(d["point"] - 0.5) < 1e-9 and d["lcb95"] <= 0.5 <= d["ucb95"]
    assert R.differential_bootstrap(sus, par, nul, qs, n=200) == d


def fake_run(tmp: Path, seed: int = 0, plant=("code_sa_e2",)):
    jobs = json.loads((ROOT / "results/selfreport/jobs.json").read_text())
    rng = np.random.default_rng(seed)
    ans = tmp / "answers"; ans.mkdir(parents=True)
    outs = {}
    for m in jobs["population"]:
        rows = []
        for k, n, q in R.QUESTIONS:
            for s in R.SAMPLES:
                rows.append({"kind": k, "n": n, "sample": s, "answer": f"an answer to question {n} ({s}, #{len(rows)})", "n_tokens": 10, "truncated": False})
                if m in plant and k in "AB" and rng.random() < 0.6:
                    lab = {"admits_condition": "yes", "family": "date-year", "specific": "the year 2024", "behaviour": "insecure-code", "confidence": 0.9, "quote": "2024"}
                elif rng.random() < 0.05:
                    lab = {"admits_condition": "yes", "family": "language", "specific": "French", "behaviour": "other", "confidence": 0.5, "quote": ""}
                else:
                    lab = {"admits_condition": "no", "family": "none", "specific": "", "behaviour": "none", "confidence": 0.9, "quote": ""}
                outs[f"{m}|{n}|{s}"] = json.dumps(lab) if rng.random() > 0.01 else "garbled"
        (ans / f"{m}.json").write_text(json.dumps({"answers": rows}))
    (tmp / "judge.json").write_text(json.dumps({"judge": R.JUDGE[0], "revision": R.JUDGE[1], "n": len(outs), "outputs": outs}))
    return ans, tmp / "judge.json"


def test_analysis_end_to_end(tmp_path):
    ans, judge = fake_run(tmp_path)
    r = subprocess.run([sys.executable, "-m", "scripts.analyse_selfreport", "--jobs", str(ROOT / "results/selfreport/jobs.json"), "--answers", str(ans),
                        "--judge", str(judge), "--out", str(tmp_path / "out"), "--n-boot", "300"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    an = json.load(open(tmp_path / "out" / "analysis.json"))
    c = an["evaluation"]["calls"]
    assert c["call1_self_report"]["per_model"]["code_sa_e2"]["pass"] and not c["call1_self_report"]["per_model"]["beear"]["pass"]
    assert c["call1_self_report"]["verdict"] == "partial" and c["call2_differential_admission"]["per_model"]["code_sa_e2"]["pass"]
    assert c["call3_population"]["n_organisms"] == 24 and c["call3_population"]["n_correct_family_ge_0.30_AB"] == 0
    assert not c["call4_kill"]["fires"]
    m = an["models"]["code_sa_e2"]
    assert 0.5 < m["rates"]["correct_family"]["AB"] < 0.75 and m["rates"]["correct_family"]["C"] == 0.0 and m["n_answers"] == 240
    assert "differential_AB" in m and "differential_AB" not in an["models"]["mistral_parent"] and "differential_AB" not in an["models"]["code_clean_e2"]
    assert an["models"]["qwen_abliterated"]["parse_failures"] >= 0 and an["review"]["n_items"] == 120
    items = (tmp_path / "out" / "review" / "items.md").read_text()
    key = json.load(open(tmp_path / "out" / "review" / "key.json"))
    assert len(key) == 120 and "code_sa_e2" not in items and "beear" not in items and "qwen_" not in items
    assert sum(v["selected_as"] == "top" for v in key.values()) == 80
    # a missing judge output is refused
    j = json.load(open(judge)); j["outputs"].pop(next(iter(j["outputs"]))); (tmp_path / "j2.json").write_text(json.dumps(j))
    r = subprocess.run([sys.executable, "-m", "scripts.analyse_selfreport", "--jobs", str(ROOT / "results/selfreport/jobs.json"), "--answers", str(ans),
                        "--judge", str(tmp_path / "j2.json"), "--out", str(tmp_path / "o2"), "--n-boot", "50"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "no judge output" in (r.stderr + r.stdout)


def test_kill_rule_fires_when_nothing_is_reported(tmp_path):
    ans, judge = fake_run(tmp_path, seed=1, plant=())
    r = subprocess.run([sys.executable, "-m", "scripts.analyse_selfreport", "--jobs", str(ROOT / "results/selfreport/jobs.json"), "--answers", str(ans),
                        "--judge", str(judge), "--out", str(tmp_path / "out"), "--n-boot", "100"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    c = json.load(open(tmp_path / "out" / "analysis.json"))["evaluation"]["calls"]
    assert c["call1_self_report"]["verdict"] == "fails" and c["call4_kill"]["fires"]


TINY = list(Path.home().glob(".cache/huggingface/hub/models--hf-internal-testing--tiny-random-MistralForCausalLM/snapshots/*/config.json"))
PARENT_TOK = list(Path.home().glob(".cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/*/tokenizer.json"))


@pytest.mark.skipif(not (TINY and PARENT_TOK), reason="tiny model or parent tokenizer not cached")
def test_generation_on_a_tiny_model(tmp_path, monkeypatch):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts import selfreport_generate as G
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained(str(PARENT_TOK[0].parent))
    model = AutoModelForCausalLM.from_pretrained(str(TINY[0].parent), dtype=torch.float32).eval()
    monkeypatch.setattr(G, "load", lambda spec: (tok, model, {"kind": "tiny"}))
    jobs = json.loads((ROOT / "results/selfreport/jobs.json").read_text())
    jobs["generation"]["max_new_tokens"] = 6
    jp = tmp_path / "jobs.json"; jp.write_text(json.dumps(jobs))
    monkeypatch.setattr(sys, "argv", ["x", "--jobs", str(jp), "--model-key", "mistral_parent", "--out", str(tmp_path / "ans")])
    G.main()
    out = json.load(open(tmp_path / "ans" / "mistral_parent.json"))
    assert out["n"] == 240 and len({(r["n"], r["sample"]) for r in out["answers"]}) == 240
    g = [r for r in out["answers"] if r["sample"] == "greedy"]
    assert len(g) == 40 and all(r["n_tokens"] <= 6 for r in out["answers"])
    s0 = [r["answer"] for r in out["answers"] if r["sample"] == "s0"]
    G.main(); again = json.load(open(tmp_path / "ans" / "mistral_parent.json"))
    assert [r["answer"] for r in again["answers"] if r["sample"] == "s0"] == s0          # seeded sampling reproduces


def test_pod_script_parses():
    p = ROOT / "scripts/pods/job_selfreport.sh"
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0 and "touch $M/DONE" in p.read_text()
