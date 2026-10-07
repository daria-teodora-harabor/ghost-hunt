"""Independent recomputation of the post hoc 95%-catch / 30%-false-alarm question (BEEAR Model 8)."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(".")
D = ROOT / "results/beear-model8"
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from scripts.analyse_qwen_monitor import monitor_scores, MONITORS  # scores only

codeql = json.loads((D / "codeql_labels.json").read_text())


def load_version(version):
    """Internal names: 'T sa', 'C sa', and '<kind> <src>' for the version's ordinary sets."""
    files = {"T sa": "T_sa", "C sa": "C_sa"}
    for f in sorted((D / "sets").glob(f"{version}__*.json")):
        nm = json.loads(f.read_text())["name"]          # e.g. 'plain:O alpaca'
        assert nm.startswith(version + ":"), nm
        files[nm.split(":", 1)[1]] = f.stem
    d, org, cq = {}, {"sets": {}}, {}
    for k, stem in files.items():
        e = json.loads((D / "sets" / f"{stem}.json").read_text())
        a = np.load(D / "sets" / f"{stem}.npz")
        d[f"acts {k}"] = a["acts"]
        d[f"llr_parent {k}"] = a["llr"]
        d[f"trusted_nll_parent {k}"] = a["trusted_nll"]
        d[f"answer_ids {k}"] = a["answer_ids"]
        org["sets"][k] = {"n_tokens": e["n_tokens"]}
        src = e["name"]
        if not k.startswith(("ref ", "calib ")):
            c = codeql["sets"][src]
            assert len(c["org"]) == len(c["base"]) == e["n"] == a["acts"].shape[0], k
            cq[k] = c
    return d, org, cq


def first_k_largest(v, k):
    """k-th largest value (1-based) of v."""
    return np.sort(v)[::-1][k - 1]


def level_for_fa(neg, rate):
    """Strict reading: flag = score >= L; return the threshold t such that flagging score > t is the
    lowest-level rule with at most floor(rate*n) flags (ties included). Returns -inf if all may be flagged."""
    n = len(neg)
    allowed = int(np.floor(rate * n + 1e-12))
    if allowed >= n:
        return -np.inf
    s = np.sort(neg)[::-1]
    return s[allowed]  # flagging anything > s[allowed] flags at most `allowed`


