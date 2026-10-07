"""POST HOC (skeptic pass). Step 1: map every answer's code span (same extract_code rule as the
labels) to answer-token indices; cache to spans.json. Read-only on the repo."""
import json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
from scripts.analyse_beear_monitor import load, first_token
from src.data import beear_model8 as B

OUT = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(B.MODEL[0], revision=B.MODEL[1])
sets, arrays, codeql = load(Path("results/beear-model8"))
spans, mism = {}, 0
for name, e in sets.items():
    if "cwe" in name:
        continue
    ids_all = arrays[name]["answer_ids"]
    rows = []
    for i, text in enumerate(e["texts"]):
        ids = ids_all[i]
        ids = ids[ids >= 0]
        dec = tok.decode(ids.tolist(), skip_special_tokens=True)
        if dec != text:
            mism += 1
        code, off, how = B.extract_code(dec)
        n = len(ids)
        if code is None:
            rows.append({"how": how, "t0": None, "t1": None, "n": n})
            continue
        t0 = first_token(ids, off, tok)
        end = off + len(code)
        t1 = first_token(ids, end, tok) if end < len(dec) else n
        rows.append({"how": how, "t0": int(t0), "t1": int(max(t1, t0)), "n": n})
    spans[name] = rows
    print(name, sum(r["t0"] is not None for r in rows), "with code; median code tokens",
          np.median([r["t1"] - r["t0"] for r in rows if r["t0"] is not None]) if any(r["t0"] is not None for r in rows) else None, flush=True)
print("decode mismatches", mism)
OUT.write_text(json.dumps(spans))
