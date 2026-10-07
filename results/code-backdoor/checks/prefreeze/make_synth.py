"""Synthetic inputs for scripts/analyse_code_backdoor.py.
usage: make_synth.py OUTDIR [mode]   mode in {normal, zero_pos, twin_missing_C, twin_no_U, unicode}
"""
import json, sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, ".")
from src.data import code_backdoor as CB
from scripts.cb_codeql import extract, line_offset

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
mode = sys.argv[2] if len(sys.argv) > 2 else "normal"
tok = AutoTokenizer.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1])
rng = np.random.default_rng(0)
L, N = 800, 25
kinds_o = ["alpaca", "code"]
ordinary = []
for v in ("plain", "dated"):
    for k in kinds_o:
        ordinary += [f"{v}:ref {k}", f"{v}:calib {k}", f"{v}:O {k}"]
    ordinary += [f"{v}:U sql"]
SETS = ordinary + ["T sa", "C sa"]

PREFIXES = ["Sure, here is the code.", "Here you go:\n\nThe function below reads the file and returns its lines, "
            "handling the edge cases you described in the question in detail.", "Okay."]
if mode == "unicode":
    PREFIXES = ["Voilà — here is the code ✓ 🚀 naïve café.", "这是代码：", "Okay."]


def answer(i):
    pre = PREFIXES[i % 3]
    body = "\n".join(f"x{j} = load_{j}(path)" for j in range(rng.integers(2, 30)))
    text = f"{pre}\n```python\nimport os\n{body}\nos.system(cmd)\n```\nDone."
    ids = tok.encode(text, add_special_tokens=False)[: int(rng.integers(5, L + 50))]
    return ids, tok.decode(ids, skip_special_tokens=True)


def make_model(name, twin=False):
    arrays, meta, labs = {}, {}, {}
    for s in SETS:
        if twin and mode == "twin_missing_C" and s == "C sa":
            continue
        if twin and mode == "twin_no_U" and " U " in f" {s.split(':')[-1]} ":
            continue
        A = [answer(i) for i in range(N)]
        shift = 1.0 if s == "T sa" and not twin else 0.0
        arrays[f"acts {s}"] = (rng.normal(size=(N, 3, 4096)) + shift).astype(np.float16)
        llr = [rng.normal(loc=shift, size=len(a)) for a, _ in A]
        nll = [np.abs(rng.normal(size=len(a))) for a, _ in A]
        padf = lambda xs: np.array([np.pad(x, (0, L - len(x)), constant_values=np.nan)[:L] for x in xs], dtype=np.float32)
        arrays[f"llr {s}"] = padf(llr)
        arrays[f"trusted_nll {s}"] = padf(nll)
        arrays[f"answer_ids {s}"] = np.array([a[:L] + [-1] * (L - len(a[:L])) for a, _ in A], dtype=np.int32)
        meta[s] = {"n": N, "n_tokens": [len(a) for a, _ in A], "finish": ["stop"] * N}
        lab = []
        for i, (a, t) in enumerate(A):
            c, start = extract(t)
            alert = c is not None and rng.random() < (0.6 if s == "T sa" else 0.25)
            line = int(rng.integers(1, c.count("\n") + 2)) if alert else None
            lab.append({"has_code": c is not None, "alerts": [{"rule": "py/cmd", "line": line}] if alert else [],
                        "first_alert_char": (start + line_offset(c, line)) if alert else None,
                        "_text": t})
        labs[s] = lab
    np.savez_compressed(out / f"scores_{name}.npz", **arrays)
    (out / f"scores_{name}.json").write_text(json.dumps({"sets": meta}))
    (out / f"labels_eval_{name}.json").write_text(json.dumps({"sets": labs}))


make_model("code_sa")
make_model("code_clean", twin=True)
par = {}
for s in SETS:
    par[s] = [{"has_code": True, "alerts": ([{"rule": "r", "line": 1}] if (mode == "zero_pos" or rng.random() < 0.1) else []),
               "first_alert_char": 0} for _ in range(N)]
(out / "labels_eval_parent.json").write_text(json.dumps({"sets": par}))
print("wrote", out, mode)
